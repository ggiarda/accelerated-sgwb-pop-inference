import json
import tqdm
import numpy as np

from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution, BrokenPowerLawPeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from gwpopulation.utils import xp

from popstock.PopulationOmegaGW import PopulationOmegaGW

class Simulator:
    """
    Simulator for OmegaGW population spectra.

    Attributes
    ----------
    models : dict
        Mass and redshift models.
    freqs : np.ndarray
        Frequency array for OmegaGW calculation.
    lambda_samples : dict
        Preloaded Lambda samples (from JSON).
    """

    def __init__(self, freqs, lambda_samples, models):
        self.freqs = freqs
        self.lambda_samples = lambda_samples
        self.models = models or {
            "mass_model": SinglePeakSmoothedMassDistribution(),
            "redshift_model": MadauDickinsonRedshift(z_max=10),
        }
        self.results = []

    def build_lambda_for_model(self, idx, base_params):
        """
        Build Lambda dictionary dynamically for the current model.

        Only include keys present in base_params and in lambda_samples.
        """
        Lambda_new = {}
        for key in base_params.keys():
            if key in self.lambda_samples:
                Lambda_new[key] = self.lambda_samples[key][idx]
            else:
                # Use default from base_params if not in lambda_samples
                Lambda_new[key] = base_params[key]
        return Lambda_new

    def run_single(
        self,
        base_params: dict,
        n_proposal_samples: int = 100_000,
        wave_approx: str = "IMRPhenomD",
        sampling_frequency: int = 2048,
    ):
        """
        Run a single OmegaGW population calculation with all Lambda samples.

        Returns
        -------
        dict
            Results dictionary with Lambdas, Neff, omega_gw, and freqs.
        """
        pop = PopulationOmegaGW(models=self.models, frequency_array=self.freqs)

        # Proposal samples
        pop.draw_and_set_proposal_samples(
            base_params, N_proposal_samples=n_proposal_samples
        )

        # Initial evaluation
        pop.calculate_omega_gw(
            waveform_approximant=wave_approx,
            Lambda=base_params,
            multiprocess=False,
        )

        result = {"Lambdas": [], "Neff": [], "omega_gw": []}
        n_trials = len(next(iter(self.lambda_samples.values())))  # get length from any key

        print("Running trials...")
        for idx in tqdm.tqdm(range(n_trials)):
            Lambda_new = self.build_lambda_for_model(idx, base_params)
            result["Lambdas"].append(Lambda_new)

            pop.calculate_omega_gw(
                sampling_frequency=sampling_frequency,
                Lambda=Lambda_new,
                multiprocess=False,
            )

            result["Neff"].append(
                float((xp.sum(pop.weights) ** 2) / (xp.sum(pop.weights ** 2)))
            )
            result["omega_gw"].append(pop.omega_gw.tolist())

        result["freqs"] = pop.frequency_array.tolist()
        return result

    def run_multiple(
        self,
        base_params: dict,
        n_runs: int = 1,
        outfile_prefix: str = "spectra_run",
        **kwargs,
    ):
        """
        Run multiple OmegaGW simulations and save results to JSON files.

        Returns
        -------
        list
            List of results for each run.
        """
        self.results = []
        for run_number in range(n_runs):
            run_result = self.run_single(base_params, **kwargs)
            filename = f"{outfile_prefix}_{run_number}.json"
            with open(filename, "w") as f:
                json.dump(run_result, f, indent=4)
            print(f"Completed run {run_number}, saved to {filename}")
            self.results.append(run_result)
        return self.results
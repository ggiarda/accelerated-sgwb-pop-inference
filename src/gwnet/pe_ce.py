import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d

import bilby
from bilby.core.prior import Uniform, LogUniform
from bilby.core.prior import Gaussian, LogNormal

import torch
import torch.nn as nn

from pygwb.constants import H0
from pycbc.psd.analytical import EinsteinTelescopeP1600143, CosmicExplorerP1600143

def make_input_format_torch(lambdas, freqs=None):

    if freqs is None:
        freqs = np.linspace(10, 2000, num=100)
    elif np.isscalar(freqs):
        freqs = np.array([freqs])
    elif not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)
    
    log_freqs = np.log(freqs)
    num_freqs = len(freqs)
    
    lambdas = np.array(lambdas)
    
    if lambdas.ndim == 1:
        lambdas = lambdas.reshape(1, -1)
        
    num_lambdas = lambdas.shape[0]
    
    input_array = np.empty((num_freqs * num_lambdas, lambdas.shape[1] + 1))
    
    input_array[:, 1:] = np.repeat(lambdas, num_freqs, axis=0)
    input_array[:, 0] = np.tile(log_freqs, num_lambdas)
    
    return input_array

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_size)

        self.input_mean = None
        self.input_std = None

        self.output_mean = None
        self.output_std = None

    def normalize_input(self, x):
        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean) / self.input_std
        return x
    
    def denormalize_output(self, x):
        if self.output_mean is not None and self.output_std is not None:
            x = (x * self.output_std) + self.output_mean
        return x

    def forward(self, x):

        x = self.normalize_input(x)

        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)

        x = self.denormalize_output(x)

        return x
    
    def compute_input_normalization_params(self, x):
        self.input_mean = torch.mean(x, dim=0)
        self.input_std = torch.std(x, dim=0)

    def compute_output_normalization_params(self, x):
        self.output_mean = torch.mean(x, dim=0)
        self.output_std = torch.std(x, dim=0)

gwbnet = torch.load('../model/mlp_plpp.pth')
gwbnet.eval()

with open('../model/sigma_rel_interpolator.pkl', 'rb') as f:
    sigma_rel = pickle.load(f)

with open('../../data/inference/omega_plpp.json', 'r') as f:
    data_dict = json.load(f)


freqs = np.array(data_dict['Frequency'])
omega = np.array(np.squeeze(data_dict['Omega']))
lambdas = np.array([value for value in data_dict['Lambdas'].values()])
labels = np.array([key for key in data_dict['Lambdas'].keys()])
injection_parameters = data_dict['Lambdas']
print(injection_parameters)
del(data_dict)

delta_f = 1.0 / 10000
flen = int(2048 / delta_f)
low_frequency_cutoff = 10

psd_et = EinsteinTelescopeP1600143(flen, delta_f, low_frequency_cutoff)
psd_ce = CosmicExplorerP1600143(flen, delta_f, low_frequency_cutoff)
psd_freqs = psd_et.sample_frequencies
S_alpha = 3 * H0.si.value ** 2 / (10 * np.pi ** 2) / freqs ** 3

psd_et_func = interp1d(psd_freqs, psd_et, kind='linear', bounds_error=False, fill_value=0)
psd_ce_func = interp1d(psd_freqs, psd_ce, kind='linear', bounds_error=False, fill_value=0)
psd_et_at_freqs = psd_et_func(freqs)/S_alpha
psd_ce_at_freqs = psd_ce_func(freqs)/S_alpha

sigma_et = psd_et_at_freqs/1e5
sigma_ce = psd_ce_at_freqs/1e5

error_et = np.random.normal(scale=sigma_et)
error_ce = np.random.normal(scale=sigma_ce)

omega_noisy_et = omega + error_et
omega_noisy_ce = omega + error_ce

def model_gwb(freqs, **params):

    if len(params) != 12:
        raise ValueError(f"Expected 12 additional arguments, but got {len(params)}")

    args = np.array([params[key] for key in params.keys()])

    nn_input  = torch.tensor(make_input_format_torch(args, freqs=freqs), dtype=torch.float32)
    nn_output = np.exp(gwbnet(nn_input).detach().numpy())
    
    return nn_output

mask_ce = freqs < 1000

label = "test"
outdir = "../../data/inference/outdir_ce"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

priors = {
            'alpha': Uniform(2., 5., '$\\alpha$'),
            'beta': Gaussian(1.2, 0.9, '$\\beta$'),
            'delta_m': Gaussian(3.5, 0.4, '$\\delta_{m}$'),
            'lam': Uniform(0, 0.25, '$\\lambda$'),
            'mmax': Uniform(60, 100, '$m_{max}$'),
            'mmin': Uniform(2, 7.5, '$m_{min}$'),
            'mpp': Uniform(22, 34, '$\\mu_{pp}$'),
            'sigpp': Uniform(1, 10, '$\\sigma_{pp}$'),
            'rate': LogUniform(0.1, 100, '$\\mathcal{R}_0$'),
            'gamma': Uniform(0.5, 6, '$\\gamma$') ,
            'kappa': Gaussian(5.6, 0.1, '$\\kappa$'),
            'z_peak': Gaussian(1.9, 0.3, '$z_{peak}$')

}

class CustomLikelihood(bilby.Likelihood):
    def __init__(self, model, x, y, sigma):
        super().__init__(parameters=dict.fromkeys(priors.keys()))
        self.model = model
        self.x = x
        self.y = y
        self.sigma = sigma
        

    def log_likelihood(self):
        nn_output = self.model(self.x, **self.parameters)
        predictions = np.squeeze(nn_output)
        sigma_y = sigma_rel(self.x)*predictions

        residuals = self.y - predictions

        sigma_eff = np.sqrt(self.sigma**2 + sigma_y**2)

        chi_squared = (residuals/sigma_eff)**2
        log_det_sigma = 2*np.log(sigma_eff) + np.log(2 * np.pi)

        log_l = -0.5*np.sum(chi_squared + log_det_sigma)

        return log_l
    
likelihood = CustomLikelihood(model= model_gwb, 
                              x=freqs[mask_ce], 
                              y=omega_noisy_ce[mask_ce], 
                              sigma= sigma_ce[mask_ce]
                             )
    
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=2500,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    dlogz=1.,
    resume=False,
    npool=4
)

result.plot_corner(priors=True)
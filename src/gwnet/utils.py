import glob
import json
import numpy as np
from tqdm import tqdm
import os
import torch
from torch.utils.data import random_split

def load_lambda_samples(filepath: str):
    """Helper to load Lambda samples from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def summarize_omega_gw_spectra(
    input_pattern: str,
    output_file: str = "omega_summary.json"
):
    """
    Compute mean and std of omega_gw across multiple JSON runs and save summary.

    Parameters
    ----------
    input_pattern : str
        Glob pattern to match input JSON files (e.g., "../data/spectra/ppl/ppl_run*.json").
    output_file : str
        Path to save the summary JSON file (e.g., "../data/spectra/ppl/omega_summary_ppl.json").

    Returns
    -------
    dict
        Dictionary containing Lambdas, mean_omega_gw, sigma_omega_gw, and optionally freqs.
    """

    # Get all matching JSON files
    spectra_files = sorted(glob.glob(input_pattern))
    if not spectra_files:
        raise FileNotFoundError(f"No files matched pattern: {input_pattern}")

    omega_gw = []
    Lambdas = None
    freqs = None

    for file in tqdm(spectra_files, desc="Processing spectra", unit="file"):
        with open(file, 'r') as json_file:
            data = json.load(json_file)

        omega_gw.append(np.array(data["omega_gw"]))

        if Lambdas is None:
            Lambdas = data["Lambdas"]
            freqs = data.get("freqs")  # Optional

    omega_gw = np.array(omega_gw)
    mean_omega = np.mean(omega_gw, axis=0)
    sigma_omega = np.std(omega_gw, axis=0, ddof=1)

    output_data = {
        "Lambdas": Lambdas,
        "mean_omega_gw": mean_omega.tolist(),
        "sigma_omega_gw": sigma_omega.tolist(),
    }

    if freqs is not None:
        output_data["freqs"] = freqs

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Summary file saved as '{output_file}'")

    return output_data  

def extract_features(Lambdas, keys):
    """
    Extract selected keys from a list of dicts as numerical features.
    """
    return [
        [entry[key] for key in keys if key in entry and isinstance(entry[key], (int, float))]
        for entry in Lambdas
    ]

def split_summary_for_training(
    summary_data: dict,
    labels,
    test_size=0.2,
    random_seed=42,
    save_dir=None,
    prefix="split"
):
    """
    Split loaded summary data into training and test sets using PyTorch.

    Parameters
    ----------
    summary_data : dict
        Already loaded summary dictionary (from JSON or elsewhere).
    labels : list of str
        Keys from Lambdas to use as features.
    test_size : float
        Fraction of data to reserve for testing.
    random_seed : int
        Seed for reproducibility.
    save_dir : str or None
        Optional path to save the split data.
    prefix : str
        Prefix for saved files (e.g., "ppl", "bpl").

    Returns
    -------
    dict
        Dictionary containing X_train, X_test, y_train, y_test, labels.
    """

    # Extract features and targets
    X = extract_features(summary_data["Lambdas"], labels)
    omega_mean = summary_data["mean_omega_gw"]
    omega_sigma = summary_data["sigma_omega_gw"]
    y = list(zip(omega_mean, omega_sigma))

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    total_size = len(X_tensor)
    test_size_abs = int(total_size * test_size)
    train_size_abs = total_size - test_size_abs

    torch.manual_seed(random_seed)

    # Split tensors (returns subsets with .indices)
    X_train_subset, X_test_subset = random_split(X_tensor, [train_size_abs, test_size_abs])
    y_train_subset, y_test_subset = random_split(y_tensor, [train_size_abs, test_size_abs])

    # Recover the actual split tensors
    X_train = X_train_subset.dataset[X_train_subset.indices].tolist()
    X_test = X_test_subset.dataset[X_test_subset.indices].tolist()
    y_train = y_train_subset.dataset[y_train_subset.indices].tolist()
    y_test = y_test_subset.dataset[y_test_subset.indices].tolist()

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "labels": labels,
    }

    # Optionally save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, f"train_{prefix}.json"), "w") as f:
            json.dump({"X_train": X_train, "y_train": y_train, "labels": labels}, f, indent=4)

        with open(os.path.join(save_dir, f"test_{prefix}.json"), "w") as f:
            json.dump({"X_test": X_test, "y_test": y_test, "labels": labels}, f, indent=4)

        print(f"Train/test splits saved in '{save_dir}' with prefix '{prefix}'")

    return result
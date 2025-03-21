#!/usr/bin/env python
"""
Script: 2d_second_order_load_model.py
---------------------------------------

This script loads a SafeSDE model (and its snapshots) from a specified results directory,
extracts key hyperparameters (regularization λ, kernel gamma parameter, confidence parameters βₛ and βᵣ),
prints these values to the console, and creates a visual table of the full configuration. The table
is saved as an image (config_table.png) in the results directory.

Dependencies:
  - methods.safe_sde_estimation: Provides the SafeSDE class.
  - matplotlib: For plotting the configuration table.
  - pickle, argparse, os: For file and argument handling.

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import pickle
import argparse
import matplotlib.pyplot as plt


def plot_config_table(config, results_dir):
    """
    Create and save a table plot showing the configuration parameters.

    Parameters:
        config (dict): Dictionary of configuration parameters.
        results_dir (str): Path to the directory where the table image will be saved.
    """
    # Convert the config dictionary to a sorted list of [key, value] rows.
    rows = [[str(k), str(v)] for k, v in sorted(config.items())]

    # Create a figure for the table.
    fig, ax = plt.subplots(figsize=(6, len(rows) * 0.35 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Create the table with two columns: Parameter and Value.
    table = ax.table(cellText=rows,
                     colLabels=["Parameter", "Value"],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("Configuration Parameters", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save the table as an image in the results directory.
    save_path = os.path.join(results_dir, "config_table.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Configuration table plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a SafeSDE model, extract its hyperparameters (λ, kernel gamma, βₛ, βᵣ), "
                    "and save a configuration table plot."
    )
    # Default values so the script runs without requiring command-line arguments.
    parser.add_argument("--results_dir", type=str, default="results_save_1000_27_02",
                        help="Path to the result folder (default: 'results_save_1000_27_02')")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Value for epsilon (ξ will be set equal to ε) (default: 0.1)")
    args = parser.parse_args()

    results_dir = args.results_dir
    epsilon = args.epsilon
    xi = epsilon  # Using ξ = ε

    # Delay import to avoid circular import issues.
    # import methods.safe_sde_estimation as sde_mod
    # SafeSDE = sde_mod.SafeSDE

    # Build the path to the pickle file that stores the models and snapshots.
    pickle_filename = os.path.join(results_dir, "safe_models_snapshots.pkl")
    if not os.path.exists(pickle_filename):
        print(f"Pickle file not found: {pickle_filename}")
        return

    with open(pickle_filename, "rb") as f:
        data = pickle.load(f)

    models = data.get("models", {})
    key = (epsilon, xi)
    if key not in models:
        print(f"Model with (ε, ξ) = ({epsilon}, {xi}) not found in {pickle_filename}")
        return

    model = models[key]

    # Get the regularization parameter (λ).
    lambda_param = getattr(model, "regularization", "Attribute not found")

    # Extract the gamma parameter from the kernel function.
    gamma_value = "Not available"
    if hasattr(model, "kernel_func"):
        kernel_func = model.kernel_func
        if hasattr(kernel_func, "keywords") and "gamma" in kernel_func.keywords:
            gamma_value = kernel_func.keywords["gamma"]
        else:
            gamma_value = "Kernel function does not use a numeric gamma parameter"

    # Get the confidence parameters for safety (βₛ) and reset (βᵣ).
    beta_s = getattr(model, "beta_s", "Attribute not found")
    beta_r = getattr(model, "beta_r", "Attribute not found")

    print(f"Model with ε = ξ = {epsilon}:")
    print(f"  Regularization (λ): {lambda_param}")
    print(f"  Kernel gamma parameter: {gamma_value}")
    print(f"  Confidence parameter for safety (βₛ): {beta_s}")
    print(f"  Confidence parameter for reset (βᵣ): {beta_r}")

    # Load and plot the configuration.
    config_filename = os.path.join(results_dir, "config.pkl")
    if os.path.exists(config_filename):
        with open(config_filename, "rb") as cf:
            config = pickle.load(cf)
        print("\nConfiguration:")
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
        # Save the configuration table plot.
        plot_config_table(config, results_dir)
    else:
        print(f"Configuration file not found: {config_filename}")


if __name__ == "__main__":
    main()

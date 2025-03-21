#!/usr/bin/env python
"""
Script: 2d_second_order_plot_model_observations.py
----------------------------------------

This script loads a SafeSDE model (from a snapshots pickle file) from a specified results directory,
extracts the true safety and reset observations stored in the model data, and creates scatter
plots of these observations over a defined control range. The plots are saved as images in a
"plots" subdirectory of the results directory.

The safe model is expected to store data in rows formatted as:
    [θ₁, …, θₘ, t, safety, reset]
For a 2D control (m = 2), the first two columns represent control parameters.

Dependencies:
  - matplotlib, seaborn, numpy, pickle, argparse, os

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_safe_model_observations(safe_model, control_range, plots_dir):
    """
    Plot true safety and reset observations from a safe model.

    The safe model is expected to store data as rows:
        [θ₁, …, θₘ, t, safety, reset]
    For a 2D control (m = 2) the first two columns represent control parameters.

    Parameters:
        safe_model: An instance of SafeSDE with a 'data' attribute (numpy array).
        control_range: Tuple of ((xmin, xmax), (ymin, ymax)) for plotting.
        plots_dir: Directory where the plots will be saved.
    """
    if safe_model.data.size == 0:
        print("No data available in safe_model.data.")
        return

    # Determine control dimension from the safe model.
    control_dim = safe_model.control_dim

    # Extract control parameters.
    if control_dim >= 2:
        thetas = safe_model.data[:, :2]  # first two columns: control parameters
    else:
        thetas = safe_model.data[:, :1]

    # In the stored data:
    #   - Column at index control_dim is time,
    #   - Column at index control_dim+1 is safety,
    #   - Column at index control_dim+2 is reset.
    safety_vals = safe_model.data[:, control_dim + 1]
    reset_vals = safe_model.data[:, control_dim + 2]

    # Set a plot style.
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # ---------------------------
    # Plot True Safety Observations
    # ---------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    xlim, ylim = control_range[0], control_range[1]
    cmap_safety = sns.color_palette("flare", as_cmap=True)
    sc1 = ax.scatter(thetas[:, 0], thetas[:, 1], c=safety_vals, cmap=cmap_safety,
                     edgecolor='w', s=50)
    fig.colorbar(sc1, ax=ax)
    ax.set_title("True Safety Observations")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    safety_plot_path = os.path.join(plots_dir, "true_observations_safety.png")
    fig.tight_layout()
    fig.savefig(safety_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved true safety plot to {safety_plot_path}")

    # ---------------------------
    # Plot True Reset Observations
    # ---------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap_reset = sns.color_palette("crest", as_cmap=True)
    sc2 = ax.scatter(thetas[:, 0], thetas[:, 1], c=reset_vals, cmap=cmap_reset,
                     edgecolor='w', s=50)
    fig.colorbar(sc2, ax=ax)
    ax.set_title("True Reset Observations")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    reset_plot_path = os.path.join(plots_dir, "true_observations_reset.png")
    fig.tight_layout()
    fig.savefig(reset_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved true reset plot to {reset_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Safe Model Observations")
    default_results_dir = "kernel_1.0_lam_1e-05_betas_0.7"

    parser.add_argument("--results_dir", type=str, default=default_results_dir,
                        help="Directory containing experiment results and safe model snapshots")
    args = parser.parse_args()

    RESULT_DIR = args.results_dir
    plots_dir = os.path.join(RESULT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load the safe model snapshots.
    snapshots_file = os.path.join(RESULT_DIR, "safe_models_snapshots.pkl")
    if not os.path.exists(snapshots_file):
        print(f"Safe model snapshots file not found at {snapshots_file}.")
        return

    with open(snapshots_file, "rb") as f:
        data = pickle.load(f)
    # Expecting the pickle file to contain a dict with keys 'models' and 'snapshots'.
    models = data.get("models", {})
    if not models:
        print("No safe models found in the snapshots file.")
        return

    # Select one safe model (for example, choose the first model in the dictionary).
    key = list(models.keys())[0]
    safe_model = models[key]
    print(f"Using safe model for (ε, ξ) = {key} with {safe_model.data.shape[0]} observations.")

    # Define control range for plotting (default for 2D controls).
    global_control_range = ((-np.pi, np.pi), (-np.pi / 2, np.pi / 2))

    # Plot observations (true safety and reset) from the safe model.
    plot_safe_model_observations(safe_model, global_control_range, plots_dir)


if __name__ == "__main__":
    main()

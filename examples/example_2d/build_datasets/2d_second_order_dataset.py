#!/usr/bin/env python
"""
Script: 2d_second_order_dataset.py
-----------------------------------------
This script computes Monte Carlo (MC) estimates for the true safety and reset probabilities
across a set of candidate control points, and then saves these estimates to a pickle file.
It also generates scatter plots of the true safety and reset observations over a defined
control range. The computed "true values" can be used later for evaluation and comparison.

Workflow:
  1. Sample N candidate control points from a given control range.
  2. For each candidate, simulate trajectories using Euler–Maruyama and compute MC estimates of:
       - Cumulative safety probability (minimum safety over time)
       - Reset probability at time T
  3. Save the candidate controls and their corresponding MC estimates to a pickle file.
  4. Plot true safety and reset observations (scatter plots) and save the plots.

Dependencies:
  - utils.data: Provides generate_one_control, euler_maruyama, second_order_coefficients,
   a_func, and plot_paths_with_turbulence_list.
  - utils.utils: Provides sample_random_candidates_from_grid.
  - methods.safe_sde_estimation: Provides the SafeSDE class.
  - numpy, matplotlib, seaborn, pickle, argparse, os, tqdm

Author: [Your Name]
Date: [Date]
"""

import os
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data import (
    generate_one_control,
    euler_maruyama,
    second_order_coefficients,
    a_func,
)
from utils.utils import sample_random_candidates_from_grid
from methods.safe_sde_estimation import SafeSDE

# ---------------------------
# Global Plot Style
# ---------------------------
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "axes.titlesize": 16,
    "axes.labelsize": 28,
    "axes.labelweight": "bold",
    "axes.linewidth": 2,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.size": 20,
    "font.weight": "bold",
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 1,
    "ytick.minor.width": 1,
})

# ---------------------------
# Load Global Simulation Parameters
# ---------------------------
RESULT_DIR = "../kernel_1.0_lam_1e-05_betas_0.7"
config_path = os.path.join(RESULT_DIR, "config.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

T = config["T"]
n_steps = config["n_steps"]
n_paths = config["n_paths"]
sigma_0 = config["sigma_0"]
dim = config["dim"]
mu_0 = np.array(config["mu_0"])
safe_bounds = tuple(config["safe_bounds"])
reset_radius = config["reset_radius"]
fixed_velocity = config["fixed_velocity"]
damping_factor = config["damping_factor"]
exploration_fraction = config["exploration_fraction"]
n_control_steps = config["n_control_steps"]


# Log configuration (for debugging purposes).
def log_config(cfg):
    print("Simulation Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")


log_config(config)

# Define output folder for plots.
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def compute_and_save_true_values(safe_model, control_range, N=1000, n_paths=1000,
                                 filename="results/true_values_dataset.pkl"):
    """
    Sample N candidate control points and compute Monte Carlo (MC) estimates of safety and reset,
    then save a dataset with entries: [theta, t, T, true_safety, true_reset].

    Parameters:
        safe_model: An instance of SafeSDE.
        control_range: Tuple of ((xmin, xmax), (ymin, ymax)) defining the control space.
        N (int): Number of candidate control points to sample.
        n_paths (int): Number of trajectories to simulate per candidate.
        filename (str): Path to save the resulting dataset.

    Returns:
        dict: A dictionary with keys "candidates", "estimated_safety", and "estimated_reset".
    """
    time_grid = np.linspace(0, T, n_steps)
    candidates = sample_random_candidates_from_grid(control_range, time_grid, T, num_candidates=N)
    control_dim = safe_model.control_dim

    estimated_cum_safety_list = []
    estimated_safety_list = []
    estimated_reset_list = []

    for candidate in tqdm(candidates, desc="Processing candidates"):
        theta = candidate[:control_dim]
        t_val = candidate[2]
        selected_u_func = generate_one_control(theta, T, num_steps=n_control_steps,
                                               fixed_velocity=fixed_velocity, mu_0=mu_0,
                                               exploration_fraction=exploration_fraction,
                                               bounds=safe_bounds, damping_factor=damping_factor)
        b_func, sigma_func = second_order_coefficients(selected_u_func, a_func, dim)
        init_state = np.concatenate([mu_0, np.zeros(dim)])
        paths = euler_maruyama(b_func, sigma_func, n_steps, n_paths, T, 2 * dim,
                               mu_0=init_state, sigma_0=sigma_0)
        est_safety = safe_model.compute_safety_probability(paths, safe_bounds=safe_bounds,
                                                           t_val=t_val, time_grid=time_grid)
        est_cum_safety = safe_model.compute_cumulative_safety_probability(paths, safe_bounds=safe_bounds,
                                                                          t_val=T, time_grid=time_grid)
        est_reset = safe_model.compute_reset_probability(paths, reset_radius=reset_radius,
                                                         t_val=T, time_grid=time_grid)
        estimated_cum_safety_list.append(est_cum_safety)
        estimated_safety_list.append(est_safety)
        estimated_reset_list.append(est_reset)

    data_dict = {
        "candidates": candidates,  # Each row: [theta_1, theta_2, t, T]
        "estimated_cumulative_safety": np.array(estimated_cum_safety_list),
        "estimated_safety": np.array(estimated_safety_list),
        "estimated_reset": np.array(estimated_reset_list)
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Saved true values dataset to {filename}")
    return data_dict


def plot_true_observations(control_range, true_values_file, plots_dir):
    """
    Plot true safety and reset observations with axis limits matching the learned maps.

    Parameters:
        control_range: Tuple of ((xmin, xmax), (ymin, ymax)) for plotting.
        true_values_file (str): Path to the true values dataset.
        plots_dir (str): Directory to save the plots.
    """
    if not os.path.exists(true_values_file):
        print(f"True values file not found: {true_values_file}. Skipping true observations plot.")
        return

    with open(true_values_file, "rb") as f:
        true_data = pickle.load(f)
    candidates = true_data["candidates"]
    true_safety = true_data["estimated_cumulative_safety"]
    true_reset = true_data["estimated_reset"]

    # Plot true safety observations.
    fig, ax = plt.subplots(figsize=(14, 6))
    xlim, ylim = control_range[0], control_range[1]
    cmap = sns.color_palette("flare", as_cmap=True)
    sc_obs1 = ax.scatter(candidates[:, 0], candidates[:, 1], c=true_safety, cmap=cmap)
    fig.colorbar(sc_obs1, ax=ax)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()
    true_obs_save_path = os.path.join(plots_dir, "true_safety.png")
    fig.savefig(true_obs_save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved true safety plot to {true_obs_save_path}")

    # Plot true reset observations.
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = sns.color_palette("crest", as_cmap=True)
    sc_obs2 = ax.scatter(candidates[:, 0], candidates[:, 1], c=true_reset, cmap=cmap)
    fig.colorbar(sc_obs2, ax=ax)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()
    true_obs_save_path = os.path.join(plots_dir, "true_reset.png")
    fig.savefig(true_obs_save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved true reset plot to {true_obs_save_path}")


# ---------------------------
# Main Script
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot true safety and reset values for candidate controls."
    )
    parser.add_argument("--results_dir", type=str, default=RESULT_DIR,
                        help="Directory containing simulation results and configuration")
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of candidate controls to sample (default: 1000)")
    parser.add_argument("--n_paths", type=int, default=100,
                        help="Number of trajectories per candidate (default: 1000)")
    parser.add_argument("--true_values_file", type=str, default=os.path.join(RESULT_DIR, "true_values_dataset.pkl"),
                        help="Filename to save the true values dataset")
    args = parser.parse_args()

    results_dir = args.results_dir
    true_values_file = args.true_values_file

    # Load configuration.
    config = pickle.load(open(os.path.join(results_dir, "config.pkl"), "rb"))
    global T, n_steps, n_paths, sigma_0, dim, mu_0, safe_bounds, reset_radius, fixed_velocity, damping_factor, \
        exploration_fraction, n_control_steps
    T = config["T"]
    n_steps = config["n_steps"]
    n_paths = config["n_paths"]
    sigma_0 = config["sigma_0"]
    dim = config["dim"]
    mu_0 = np.array(config["mu_0"])
    safe_bounds = tuple(config["safe_bounds"])
    reset_radius = config["reset_radius"]
    fixed_velocity = config["fixed_velocity"]
    damping_factor = config["damping_factor"]
    exploration_fraction = config["exploration_fraction"]
    n_control_steps = config["n_control_steps"]
    seed: int = config["seed"]
    np.random.seed(seed)
    random.seed(seed)

    # Set a default control range for candidate sampling.
    global_control_range = ((-np.pi, np.pi), (-np.pi / 2, np.pi / 2))

    # Create a SafeSDE instance (an empty one for this computation).
    safe_model = SafeSDE(control_dim=2)

    # 1. Compute and save true values.
    _ = compute_and_save_true_values(safe_model, global_control_range, N=args.N,
                                     n_paths=args.n_paths, filename=true_values_file)

    # 2. Plot true observations.
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_true_observations(global_control_range, true_values_file, plots_dir)


if __name__ == "__main__":
    main()

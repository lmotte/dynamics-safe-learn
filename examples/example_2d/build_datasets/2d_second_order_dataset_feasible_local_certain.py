#!/usr/bin/env python3
"""
Script: 2d_second_order_dataset_feasible_local_certain.py
----------------------------------------------
This script builds a dataset from saved safe models by considering only the feasible candidates—
i.e. those with predicted safety above a specified threshold and predictive uncertainty below a
specified threshold. The script uses vectorized computation for filtering candidates and local
sampling (via sample_local_candidates) to generate a candidate set. For each feasible candidate,
it simulates trajectories (via Euler–Maruyama) and computes Monte Carlo (MC) estimates for safety
and reset probabilities.

The resulting dataset is returned as a dictionary containing:
  - "candidates": Array of candidate parameters ([θ, t, T])
  - "estimated_safety": Array of MC-estimated safety probabilities
  - "estimated_reset": Array of MC-estimated reset probabilities

Dependencies:
  - utils.data: Provides generate_one_control, euler_maruyama, second_order_coefficients,
                generate_controls, a_func.
  - utils.utils: Provides sample_local_candidates.
  - methods.safe_sde_estimation: Provides the SafeSDE class.
  - numpy, pickle, argparse, os, time, tqdm, matplotlib, seaborn.

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import random
import pickle
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Set the results directory (adjust as needed)
# ---------------------------
RESULT_DIR = "../kernel_1.0_lam_1e-05_betas_0.7"

# ---------------------------
# Set a global plot style.
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
    "ytick.minor.width": 1
})

# ---------------------------
# Load global simulation parameters
# ---------------------------
config_path = os.path.join(RESULT_DIR, "config.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

# Simulation parameters from config.
T = config["T"]                       # Total simulation time.
n_steps = config["n_steps"]           # Number of time steps.
n_paths = config["n_paths"]           # Number of trajectories per candidate.
sigma_0 = config["sigma_0"]           # Initial noise standard deviation.
dim = config["dim"]                   # Position dimension (full state = 2*dim).
mu_0 = np.array(config["mu_0"])         # Initial (target) position.
safe_bounds = tuple(config["safe_bounds"])  # Safety bounds.
reset_radius = config["reset_radius"]
fixed_velocity = config["fixed_velocity"]
damping_factor = config["damping_factor"]
exploration_fraction = config["exploration_fraction"]
n_control_steps = config["n_control_steps"]  # Dimension of control vector.
seed = config["seed"]
np.random.seed(seed)
random.seed(seed)

# ---------------------------
# Import necessary functions and classes.
# ---------------------------
from utils.data import (
    generate_one_control,
    euler_maruyama,
    second_order_coefficients,
    generate_controls,
    a_func
)
from utils.utils import sample_local_candidates
from methods.safe_sde_estimation import SafeSDE

# ---------------------------
# Global filtering parameters.
# ---------------------------
EPSILON_THRESHOLD = 0.8         # Predicted safety must be >= this threshold.
UNCERTAINTY_THRESHOLD = 0.01      # Predictive uncertainty must be below this threshold.
N_CANDIDATES = 100              # Number of candidate controls to generate initially.


def build_feasible_candidates_dataset_local(
        safe_model,
        epsilon_threshold=EPSILON_THRESHOLD,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        N_candidates=N_CANDIDATES,
        n_steps=n_steps,
        n_paths=n_paths,
        T=T,
        safe_bounds=safe_bounds,
        sigma_0=sigma_0,
        fixed_velocity=fixed_velocity,
        exploration_fraction=exploration_fraction
):
    """
    Build a dataset for a given safe_model by:
      1. Generating a default candidate set using bounded controls and local sampling.
      2. Filtering candidates via vectorized prediction to keep those with predicted safety
         above epsilon_threshold and uncertainty below uncertainty_threshold.
      3. For each feasible candidate, simulating trajectories and computing MC estimates for
         safety and reset probabilities.

    Returns:
      dict: {
          "candidates": array of candidate parameters ([θ, t, T]),
          "estimated_safety": array of MC-estimated safety probabilities,
          "estimated_reset": array of MC-estimated reset probabilities
      }
    """
    # Define the control range (should match your training/experiment setup).
    global_control_range = ((-0.5, 5), (3.5, 7.5))
    time_grid = np.linspace(0, T, n_steps)

    # --- Generate a default candidate set using bounded controls ---
    control_data = generate_controls(K=100, T=T, num_steps=n_control_steps,
                                     fixed_velocity=fixed_velocity,
                                     mu_0=np.zeros(safe_model.control_dim),
                                     exploration_fraction=exploration_fraction,
                                     threshold_radius=2.5,
                                     bounds=(-9.9, 9.9))
    candidate_t_choices = time_grid[:100]
    default_candidate_rows = []
    for (_, angles, step_times) in control_data:
        t_val_candidate = np.random.choice(candidate_t_choices)
        candidate_row = np.hstack((np.array(angles), [t_val_candidate, T]))
        default_candidate_rows.append(candidate_row)
    default_candidate_set = np.array(default_candidate_rows)

    # --- Local sampling of candidates ---
    candidate_set = sample_local_candidates(
        safe_model,
        default_candidate_set,
        time_grid,
        T,
        num_candidates=N_candidates,
        margin_angle=0.5,
        margin_t=0.1
    )

    # --- Vectorized candidate filtering ---
    control_dim = safe_model.control_dim
    theta_candidates = candidate_set[:, :control_dim]
    t_candidates = candidate_set[:, control_dim]

    # Batch predict safety and uncertainty.
    pred_safety, variances = safe_model.predict(theta_candidates, t_candidates, target='safety')
    feasible_mask = (pred_safety >= epsilon_threshold) & (variances < uncertainty_threshold)
    feasible_candidates = candidate_set[feasible_mask]

    if feasible_candidates.shape[0] == 0:
        print(f"[INFO] No feasible candidates found with predicted safety threshold {epsilon_threshold} "
              f"and uncertainty threshold {uncertainty_threshold}.")
        return None

    # --- Monte Carlo evaluation for each feasible candidate ---
    estimated_safety_list = []
    estimated_reset_list = []
    mu_0_local = np.zeros(control_dim)  # initial position for evaluation
    reset_radius_local = reset_radius

    for candidate in tqdm(feasible_candidates, desc="Evaluating feasible candidates"):
        theta = candidate[:control_dim]
        t_val = candidate[control_dim]
        selected_u_func = generate_one_control(
            theta, T,
            num_steps=n_control_steps,
            fixed_velocity=fixed_velocity,
            mu_0=mu_0_local,
            exploration_fraction=exploration_fraction,
            bounds=safe_bounds
        )
        b_func, sigma_func = second_order_coefficients(selected_u_func, a_func, control_dim)
        init_state = np.concatenate([mu_0_local, np.zeros(control_dim)])  # positions & velocities
        paths = euler_maruyama(
            b_func, sigma_func,
            n_steps, n_paths, T,
            2 * control_dim,
            mu_0=init_state, sigma_0=sigma_0
        )
        est_safety = safe_model.compute_cumulative_safety_probability(paths, safe_bounds=safe_bounds,
                                                                      t_val=T, time_grid=time_grid)
        est_reset = safe_model.compute_reset_probability(paths,
                                                         reset_radius=reset_radius_local,
                                                         t_val=T, time_grid=time_grid)
        estimated_safety_list.append(est_safety)
        estimated_reset_list.append(est_reset)

    data_dict = {
        "candidates": feasible_candidates,
        "estimated_safety": np.array(estimated_safety_list),
        "estimated_reset": np.array(estimated_reset_list)
    }
    return data_dict


def main():
    models_file = os.path.join(RESULT_DIR, "safe_models_snapshots.pkl")
    if not os.path.exists(models_file):
        print(f"[ERROR] Models file not found: {models_file}")
        return

    with open(models_file, "rb") as f:
        saved_data = pickle.load(f)
    models = saved_data.get("models", {})

    if not models:
        print("[ERROR] No models found in the loaded file.")
        return

    # Process each saved model (keyed by (ε, ξ)).
    for key, safe_model in models.items():
        eps, xi = key
        print(f"[INFO] Processing model with ε = {eps} and ξ = {xi} ...")
        start_time = time.time()
        dataset = build_feasible_candidates_dataset_local(
            safe_model,
            epsilon_threshold=EPSILON_THRESHOLD,
            uncertainty_threshold=UNCERTAINTY_THRESHOLD,
            N_candidates=N_CANDIDATES,
            n_steps=n_steps,
            n_paths=n_paths,
            T=T,
            safe_bounds=safe_bounds,
            sigma_0=sigma_0,
            fixed_velocity=fixed_velocity,
            exploration_fraction=exploration_fraction
        )
        elapsed = time.time() - start_time
        if dataset is None:
            print(f"[INFO] No feasible candidates found for model with ε = {eps} and ξ = {xi}.")
            continue

        output_filename = os.path.join(RESULT_DIR, f"feasible_candidates_dataset_eps{eps}_xi{xi}.pkl")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "wb") as f_out:
            pickle.dump(dataset, f_out)
        print(f"[INFO] Saved feasible candidates dataset to {output_filename} (processing time: {elapsed:.2f} sec).")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Script: 2d_second_order_predict_control.py
-----------------------------------
This script loads a SafeSDE model from a specified results directory, then predicts the cumulative safety
(probability) over time and the reset probability at time T for a single control candidate. The candidate
control is provided as a comma-separated string. Optionally, the script also evaluates "true" values via a
Monte Carlo simulation of trajectories.

The safe model is expected to store data in rows formatted as:
    [θ₁, …, θₘ, t, safety, reset]

Dependencies:
  - utils.data: Provides generate_one_control, euler_maruyama, second_order_coefficients, a_func.
  - numpy, matplotlib, pickle, argparse, os.

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import random
import pickle
import argparse
import numpy as np
from utils.data import (
    generate_one_control,
    euler_maruyama,
    second_order_coefficients,
    a_func
)

# ---------------------------
# Default Values
# ---------------------------
DEFAULT_RESULTS_DIR = "kernel_1.0_lam_1e-05_betas_0.7"
DEFAULT_CONTROL = "1.45005022, -0.95123739"
INITIAL_CONTROL = "-1.05, 1.05"  # initial safe control


def load_config(results_dir):
    """Load simulation configuration from config.pkl."""
    config_path = os.path.join(results_dir, "config.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return config


def load_safe_model(results_dir):
    """
    Load safe model snapshots from safe_models_snapshots.pkl.
    Returns the first safe model found.
    """
    snapshots_file = os.path.join(results_dir, "safe_models_snapshots.pkl")
    if not os.path.exists(snapshots_file):
        raise FileNotFoundError(f"Safe model snapshots file not found at {snapshots_file}")
    with open(snapshots_file, "rb") as f:
        data = pickle.load(f)
    models = data.get("models", {})
    if not models:
        raise ValueError("No safe models found in the snapshots file.")
    key = list(models.keys())[0]
    safe_model = models[key]
    print(f"Using safe model for (ε, ξ) = {key} with {safe_model.data.shape[0]} observations.")
    return safe_model


def parse_control(control_str):
    """
    Parse a comma-separated string into a NumPy array.
    E.g., "0.2,0.8" becomes array([0.2, 0.8]).
    """
    try:
        control = np.array([float(x) for x in control_str.split(",")])
    except Exception as e:
        raise ValueError("Control candidate must be provided as comma-separated floats (e.g. '0.2,0.8').") from e
    return control


def main():
    parser = argparse.ArgumentParser(
        description="Predict cumulative safety and reset probabilities for a single control candidate."
    )
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Directory with safe model snapshots and config")
    parser.add_argument("--control", type=str, default=INITIAL_CONTROL,
                        help="Control candidate as comma-separated floats (e.g. '0.2,0.8')")
    parser.add_argument("--time_samples", type=int, default=100,
                        help="Number of time samples between 0 and T for cumulative safety evaluation")
    args = parser.parse_args()

    results_dir = args.results_dir
    control_candidate = parse_control(args.control)

    # Load simulation configuration.
    config = load_config(results_dir)
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

    # Load the safe model.
    safe_model = load_safe_model(results_dir)

    # --- Predict cumulative safety probability over time for the control candidate ---
    time_samples = args.time_samples
    time_grid_vals = np.linspace(0, T, time_samples)
    safety_predictions = []
    for t_val in time_grid_vals:
        pred_safety, _ = safe_model.predict(control_candidate, t_val, target='safety')
        # Clip predictions to [0, 1].
        pred_safety = np.clip(pred_safety, 0, 1)
        safety_predictions.append(pred_safety)
    safety_predictions = np.array(safety_predictions)
    cumulative_safety = np.min(safety_predictions)

    # --- Predict reset probability at time T for the control candidate ---
    pred_reset, _ = safe_model.predict(control_candidate, T, target='reset')
    pred_reset = np.clip(pred_reset, 0, 1)

    print(f"\nFor control candidate {control_candidate}:")
    print(f"  Cumulative Safety Probability (min over time [0, T]): {cumulative_safety:.4f}")
    print(f"  Reset Probability at T: {pred_reset:.4f}")

    # --- (Optional) Evaluate true safety/reset via Monte Carlo simulation ---
    selected_u_func = generate_one_control(control_candidate, T, num_steps=n_control_steps,
                                           fixed_velocity=fixed_velocity, mu_0=mu_0,
                                           exploration_fraction=exploration_fraction, bounds=safe_bounds,
                                           damping_factor=damping_factor)
    b_func, sigma_func = second_order_coefficients(selected_u_func, a_func, dim)
    init_state = np.concatenate([mu_0, np.zeros(dim)])
    paths = euler_maruyama(b_func, sigma_func, n_steps, n_paths, T, 2 * dim,
                           mu_0=init_state, sigma_0=sigma_0)
    time_grid = np.linspace(0, T, n_steps)
    true_safety = safe_model.compute_cumulative_safety_probability(paths,
                                                                   safe_bounds=safe_bounds,
                                                                   t_val=T, time_grid=time_grid)
    true_reset = safe_model.compute_reset_probability(paths,
                                                      reset_radius=reset_radius,
                                                      t_val=T, time_grid=time_grid)
    print("\nMonte Carlo (true) estimates:")
    print(f"  True cumulative safety probability: {true_safety:.4f}")
    print(f"  True reset probability:             {true_reset:.4f}")


if __name__ == "__main__":
    main()

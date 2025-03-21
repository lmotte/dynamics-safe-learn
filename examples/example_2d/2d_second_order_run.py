#!/usr/bin/env python
"""
Run Safe Learning Experiments for 2D Second-Order Systems
---------------------------------------------------------

This script conducts safe learning experiments for 2D second-order systems by:
  - Loads candidate controls and runs safe learning for a given (ε, ξ) pair.
  - Simulates trajectories and updates the SafeSDE model.
  - Computes performance metrics (e.g., safety and reset probabilities, information gain).
  - Saves metrics to a CSV file and pickles the safe models, snapshots, and paths data.

Dependencies:
  - methods.safe_sde_estimation (provides the SafeSDE class)
  - utils.data (provides euler_maruyama, second_order_coefficients, generate_controls, generate_one_control, a_func)
  - utils.utils (provides sample_local_candidates)

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import time
import copy
import random
import logging
import argparse
import pickle
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import the safe learning class and associated functions.
from methods.safe_sde_estimation import SafeSDE
from utils.data import (
    euler_maruyama,
    second_order_coefficients,
    generate_one_control,
    a_func,
)
from utils.utils import sample_local_candidates

# Configure logging.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Command-line Arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Safe Learning Experiments")
parser.add_argument("--kernel_func", type=float, default=2.0, help="Kernel function parameter value")
parser.add_argument("--betas", type=float, default=1e-1, help="Betas parameter for safety and reset")
parser.add_argument("--lam", type=float, default=1e-6, help="Regularization (lambda) parameter")
args = parser.parse_args()
kernel_func_value: float = args.kernel_func
betas_value: float = args.betas
lam_values: float = args.lam


# -----------------------------------------------------------------------------
# Safe Learning Runner Function
# -----------------------------------------------------------------------------
def run_safe_learning(
        epsilon: float,
        xi: float,
        max_iterations: int = 100,
        snapshot_iterations: Optional[List[int]] = None,
        regularization: float = 1e-5,
        config: Optional[Dict[str, Any]] = None
) -> Tuple[SafeSDE, Dict[str, Any], Dict[int, SafeSDE], Dict[str, Any]]:
    """
    Run safe learning for a given (ε, ξ) pair.

    Parameters:
        epsilon (float): Safety threshold.
        xi (float): Reset threshold.
        max_iterations (int): Maximum number of iterations.
        snapshot_iterations (list, optional): Iterations at which to save a snapshot of the model.
        regularization (float): Regularization parameter for the SafeSDE model.
        config (dict, optional): Dictionary of simulation parameters.

    Returns:
        tuple: (safe_model, metrics, snapshots, paths_data) where:
          - safe_model: Final SafeSDE model.
          - metrics: Dictionary with performance metrics.
          - snapshots: Dictionary of model snapshots at selected iterations.
          - paths_data: Dictionary with simulation trajectories and associated arrays.
    """
    if snapshot_iterations is None:
        snapshot_iterations = [10, 50, 99]

    # Simulation parameters from config.
    T: float = config["T"]
    n_steps: int = config["n_steps"]
    time_grid: np.ndarray = np.linspace(0, T, n_steps)
    n_paths: int = config["n_paths"]
    sigma_0: float = config["sigma_0"]
    dim: int = config["dim"]
    mu_0: np.ndarray = np.zeros(dim)
    safe_bounds: List[float] = config["safe_bounds"]
    reset_radius: float = config["reset_radius"]
    fixed_velocity: float = config["fixed_velocity"]
    exploration_fraction: float = config["exploration_fraction"]
    seed: int = config["seed"]
    np.random.seed(seed)
    random.seed(seed)
    damping_factor: float = config["damping_factor"]
    n_control_steps: int = config["n_control_steps"]
    control_dim: int = n_control_steps

    # --- Pre-generate candidate controls.
    t_ctrl_start: float = time.time()
    # (Candidate control generation can be expanded here if needed.)
    t_ctrl_end: float = time.time()
    logger.info(f"Candidate control generation took {t_ctrl_end - t_ctrl_start:.3f} sec.")

    # Create initial safe candidates (gamma0) from fixed control parameters.
    gamma0_list = []
    for t in time_grid[::5]:
        safe_candidate = np.hstack(([-np.pi / 3, np.pi / 3], [t, T]))
        gamma0_list.append(safe_candidate)
    gamma0: np.ndarray = np.array(gamma0_list)

    # Initialize the SafeSDE model.
    beta_s: float = betas_value
    beta_r: float = betas_value
    safe_model: SafeSDE = SafeSDE(
        beta_s=beta_s,
        beta_r=beta_r,
        gamma0=gamma0,
        regularization=regularization,
        control_dim=control_dim,
        state_dim=2,
        kernel_func=kernel_func_value
    )

    # Initialize history and storage.
    exploration_history: Dict[int, Tuple[Any, ...]] = {}
    snapshots: Dict[int, SafeSDE] = {}
    iteration_paths: List[np.ndarray] = []  # Each element: (n_paths, n_steps, 2*dim)
    iteration_angles: List[np.ndarray] = []  # Each element: (n_paths, n_steps)
    iteration_step_times: List[np.ndarray] = []  # Each element: (n_paths, n_steps)

    start_time_loop: float = time.time()
    for iteration in tqdm(range(max_iterations), desc="Safe learning iterations"):
        # Candidate sampling.
        t_candidate_start: float = time.time()
        candidate_set: np.ndarray = sample_local_candidates(
            safe_model, gamma0, time_grid, T,
            num_candidates=100, margin_angle=0.5, margin_t=0.1
        )
        t_candidate_end: float = time.time()
        logger.info(f"Iteration {iteration}: Candidate sampling took {t_candidate_end - t_candidate_start:.3f} sec.")

        # Candidate selection.
        t_select_start: float = time.time()
        candidate, safety_pred, uncertainty_pred = safe_model.sample_next(candidate_set, epsilon=epsilon, xi=xi)
        t_select_end: float = time.time()
        logger.info(f"Iteration {iteration}: Candidate selection took {t_select_end - t_select_start:.3f} sec.")

        if candidate is None:
            logger.warning("No feasible candidate found. Terminating early.")
            break

        theta: np.ndarray = candidate[:control_dim]
        t_val_candidate: float = candidate[control_dim]

        # Generate control function using the selected candidate.
        selected_u_func = generate_one_control(
            theta, T, num_steps=n_control_steps,
            fixed_velocity=fixed_velocity, mu_0=mu_0,
            exploration_fraction=exploration_fraction, bounds=safe_bounds,
            damping_factor=damping_factor
        )

        b_func, sigma_func = second_order_coefficients(selected_u_func, a_func, dim)
        init_state: np.ndarray = np.concatenate([mu_0, np.zeros(dim)])

        # Trajectory simulation.
        t_sim_start: float = time.time()
        paths: np.ndarray = euler_maruyama(
            b_func, sigma_func, n_steps, n_paths, T, 2 * dim,
            mu_0=init_state, sigma_0=sigma_0
        )
        t_sim_end: float = time.time()
        logger.info(f"Iteration {iteration}: Trajectory simulation took {t_sim_end - t_sim_start:.3f} sec.")

        # Compute safety probabilities.
        t_safety_start: float = time.time()
        safety_prob: float = safe_model.compute_safety_probability(
            paths, safe_bounds=safe_bounds, t_val=t_val_candidate, time_grid=time_grid
        )
        safety_prob_cum: float = safe_model.compute_cumulative_safety_probability(
            paths, safe_bounds=safe_bounds, t_val=T, time_grid=time_grid
        )
        t_safety_end: float = time.time()
        logger.info(
            f"Iteration {iteration}: Safety probability computation took {t_safety_end - t_safety_start:.3f} sec.")

        # Compute reset probabilities.
        t_reset_start: float = time.time()
        # reset_prob: float = safe_model.compute_reset_probability(
        #     paths, reset_radius=reset_radius, t_val=t_val_candidate, time_grid=time_grid
        # )
        reset_prob_T: float = safe_model.compute_reset_probability(
            paths, reset_radius=reset_radius, t_val=T, time_grid=time_grid
        )
        t_reset_end: float = time.time()
        logger.info(f"Iteration {iteration}: Reset probability computation took {t_reset_end - t_reset_start:.3f} sec.")

        # Update model with new data.
        density_index = np.where(time_grid == t_val_candidate)[0][0]
        density_states = paths[:, density_index, :2]
        t_update_start: float = time.time()
        safe_model.add_data(theta, t_val_candidate, safety_prob, reset_prob_T, density_states=density_states)
        t_update_end: float = time.time()
        logger.info(f"Iteration {iteration}: Model update took {t_update_end - t_update_start:.3f} sec.")

        logger.info(
            f"Iteration {iteration}: (theta, t_val_candidate, safety_prob_cum, reset_prob_T, uncertainty_pred): "
            f"{(theta, t_val_candidate, safety_prob_cum, reset_prob_T, uncertainty_pred)}"
        )
        exploration_history[iteration] = (theta, t_val_candidate, safety_prob_cum, reset_prob_T, uncertainty_pred)
        if iteration in snapshot_iterations:
            snapshots[iteration] = copy.deepcopy(safe_model)

        # Save paths for plotting.
        angles_array = np.full((n_paths, n_steps), theta[0])
        step_times_array = np.tile(time_grid, (n_paths, 1))
        iteration_paths.append(paths)
        iteration_angles.append(angles_array)
        iteration_step_times.append(step_times_array)

    total_time_loop: float = time.time() - start_time_loop

    # Compute performance metrics.
    if exploration_history:
        uncertainty_values = np.array([record[4] for record in exploration_history.values()])
        I_N = 0.5 * np.sum(np.log([
            1 + uncertainty_value / ((i + 1) * regularization)
            for i, uncertainty_value in enumerate(uncertainty_values)
        ]))
        info_gain_rate = I_N / len(exploration_history)
        all_safety = np.array([record[2] for record in exploration_history.values()])
        lowest_safety = float(all_safety.min())
        average_safety = float(all_safety.mean())
        all_reset = np.array([record[3] for record in exploration_history.values()])
        average_reset = float(all_reset.mean())
        average_crash_rate = 1 - average_safety
        lowest_reset = float(all_reset.min())
    else:
        I_N = info_gain_rate = lowest_safety = average_safety = average_reset = average_crash_rate = lowest_reset = None

    metrics: Dict[str, Any] = {
        'epsilon': epsilon,
        'xi': xi,
        'iterations': len(exploration_history),
        'total_time': total_time_loop,
        'lowest_safety': lowest_safety,
        'average_safety': average_safety,
        'average_reset': average_reset,
        'average_crash_rate': average_crash_rate,
        'lowest_reset': lowest_reset,
        'information_gain': I_N,
        'info_gain_rate': info_gain_rate
    }
    safe_model.exploration_history = exploration_history

    paths_data: Dict[str, Any] = {
        "iteration_paths": iteration_paths,
        "iteration_angles": iteration_angles,
        "iteration_step_times": iteration_step_times,
        "plot_iterations": snapshot_iterations
    }

    return safe_model, metrics, snapshots, paths_data


# -----------------------------------------------------------------------------
# Main Loop: Run Safe Learning Experiments
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define simulation configuration.
    config: Dict[str, Any] = {
        "T": 20,
        "n_steps": 500,
        "n_paths": 200,
        "sigma_0": 0.1,
        "dim": 2,
        "mu_0": np.zeros(2).tolist(),  # Converted to list for potential JSON compatibility.
        "safe_bounds": [-10, 10],
        "reset_radius": 2.5,
        "fixed_velocity": 2.0,
        "exploration_fraction": 6 / 20,
        "n_control_steps": 2,
        "seed": 1,
        "damping_factor": 0.5
    }

    # Define epsilon values (safety thresholds) to test.
    epsilons: List[float] = [0.1, 0.3, 0.5, 1e6]
    all_metrics: List[Dict[str, Any]] = []
    all_models: Dict[Tuple[float, float], SafeSDE] = {}
    all_snapshots: Dict[Tuple[float, float], Dict[int, SafeSDE]] = {}
    all_paths_data: Dict[Tuple[float, float], Dict[str, Any]] = {}

    for eps in epsilons:
        xi_val = eps  # In this experiment, reset threshold equals safety threshold.
        logger.info("Running safe learning for ε=%s and ξ=%s", eps, xi_val)
        # safe_model, metrics, snapshots, paths_data = run_safe_learning(
        #     eps, xi_val, max_iterations=1000, snapshot_iterations=[999],
        #     regularization=lam_values, config=config
        # )
        safe_model, metrics, snapshots, paths_data = run_safe_learning(
            eps, xi_val, max_iterations=1000, snapshot_iterations=[999],
            regularization=lam_values, config=config
        )
        all_metrics.append(metrics)
        all_models[(eps, xi_val)] = safe_model
        all_snapshots[(eps, xi_val)] = snapshots
        all_paths_data[(eps, xi_val)] = paths_data
        logger.info("Metrics: %s", metrics)

    # Create results directory.
    results_dir = f"kernel_{kernel_func_value}_lam_{lam_values}_betas_{betas_value}"
    os.makedirs(results_dir, exist_ok=True)

    # Save configuration.
    config_path = os.path.join(results_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    logger.info("Running script with config: %s", config)

    # Save metrics to CSV.
    df = pd.DataFrame(all_metrics)
    csv_filename = os.path.join(results_dir, "2d_second_order_results.csv")
    df.to_csv(csv_filename, index=False)
    logger.info("Saved results to %s", csv_filename)

    # Save models and snapshots.
    snapshots_filename = os.path.join(results_dir, "safe_models_snapshots.pkl")
    with open(snapshots_filename, "wb") as f:
        pickle.dump({'models': all_models, 'snapshots': all_snapshots}, f)
    logger.info("Saved safe models and snapshots to %s", snapshots_filename)

    # Save paths data.
    paths_filename = os.path.join(results_dir, "paths_data.pkl")
    with open(paths_filename, "wb") as f:
        pickle.dump(all_paths_data, f)
    logger.info("Saved simulation paths data to %s", paths_filename)

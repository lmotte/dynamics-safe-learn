"""
utils/utils.py
--------------

This module provides utility functions for simulating controlled stochastic differential equations,
generating candidate controls, and creating a variety of plots (trajectories, control functions,
density maps, etc.) used in safe learning experiments.

Key functionalities:
  - Evaluation of pairwise functions (rho) for data points.
  - Sampling candidate control parameters locally (near the current safe set) or randomly from a grid.
  - Plotting observed histograms for safety and reset probabilities.

Dependencies:
  - numpy
  - matplotlib
  - logging

Author: Luc Brogat-Motte
Date: 2025
"""

import logging
from typing import Tuple, Union, Optional, Any, Dict

import numpy as np
import matplotlib.pyplot as plt

# Configure module-level logger.
logger = logging.getLogger(__name__)


def rho(X: np.ndarray, Y: np.ndarray, mu: float) -> np.ndarray:
    """
    Evaluate the function rho(X_i, Y_i) for each pair (X_i, Y_i) between X and Y.

    Parameters:
        X (np.ndarray): Array of shape (nx, n) containing the points X_i.
        Y (np.ndarray): Array of shape (ny, n) containing the points Y_i.
        mu (float): Parameter of rho.

    Returns:
        np.ndarray: Array of shape (nx, ny) containing the pairwise evaluation.
    """
    _, n = X.shape
    # Expand dimensions to enable pairwise subtraction.
    X_expanded = X[:, np.newaxis, :]  # Shape: (nx, 1, n)
    Y_expanded = Y[np.newaxis, :, :]  # Shape: (1, ny, n)
    # Compute squared Euclidean distances without taking a square root.
    squared_distances = np.sum((X_expanded - Y_expanded) ** 2, axis=2)
    # Evaluate the function rho over each pair.
    rho_xy = mu**n * (2 * np.pi) ** (-n / 2) * np.exp(-0.5 * mu**2 * squared_distances)
    return rho_xy


# =============================================================================
# Candidate Sampling Functions
# =============================================================================

def sample_local_candidates(
    safe_model: Any,
    default_candidate_set: np.ndarray,
    time_grid: np.ndarray,
    T: float,
    num_candidates: int = 30,
    margin_angle: float = 0.2,
    margin_t: float = 0.05,
    angle_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
) -> np.ndarray:
    """
    Sample candidate control parameters from a local region near the current safe set.

    For a control space of dimension m, each candidate is of the form:
        [theta_1, …, theta_m, t, T].

    Parameters:
        safe_model: Instance of SafeSDE.
        default_candidate_set (np.ndarray): Fallback candidate set if no data exists.
        time_grid (np.ndarray): Array of discretized time values.
        T (float): Total simulation time.
        num_candidates (int): Number of candidates to sample.
        margin_angle (float): Margin to expand the control range.
        margin_t (float): Margin to expand the time range.
        angle_range (tuple): Range of candidate angles (radians).

    Returns:
        np.ndarray: Candidate set of shape (num_candidates, m+2).
    """
    if safe_model.data.shape[0] == 0:
        logger.info("No training data; returning default candidate set.")
        return default_candidate_set

    m = safe_model.control_dim
    control_candidates = []
    for d in range(m):
        min_val = np.min(safe_model.data[:, d])
        max_val = np.max(safe_model.data[:, d])
        if d == 0:
            # For the first control parameter, use interval [-pi, pi]
            range_d = (max(min_val - margin_angle, -np.pi), min(max_val + margin_angle, np.pi))
        else:
            range_d = (max(min_val - margin_angle, angle_range[0]),
                       min(max_val + margin_angle, angle_range[1]))
        sampled_values = np.random.uniform(range_d[0], range_d[1], num_candidates)
        control_candidates.append(sampled_values)
    control_candidates = np.column_stack(control_candidates)

    min_t = np.min(safe_model.data[:, m])
    max_t = np.max(safe_model.data[:, m])
    t_range = (max(0, min_t - margin_t), min(time_grid[-1], max_t + margin_t))
    possible_ts = time_grid[(time_grid >= t_range[0]) & (time_grid <= t_range[1])]
    if len(possible_ts) == 0:
        possible_ts = np.array([t_range[0]])
    ts = np.random.choice(possible_ts, size=num_candidates)
    candidate_set = np.column_stack((control_candidates, ts, T * np.ones(num_candidates)))
    return candidate_set


def sample_random_candidates_from_grid(
    control_range: Union[list, tuple],
    time_grid: np.ndarray,
    T: float,
    num_candidates: int = 30
) -> np.ndarray:
    """
    Generate random candidates uniformly from a given control range and time grid.

    For a control space of dimension m, each candidate is of the form:
        [theta_1, …, theta_m, t, T].

    Parameters:
        control_range (list/tuple): List of m tuples, each (min, max) for that control dimension.
        time_grid (np.ndarray): Array of discretized time values.
        T (float): Total simulation time.
        num_candidates (int): Number of candidates.

    Returns:
        np.ndarray: Candidate set of shape (num_candidates, m+2).
    """
    m = len(control_range)
    control_candidates = []
    for d in range(m):
        min_val, max_val = control_range[d]
        samples = np.random.uniform(min_val, max_val, num_candidates)
        control_candidates.append(samples)
    control_candidates = np.column_stack(control_candidates)
    ts = np.random.choice(time_grid, size=num_candidates, replace=True)
    candidate_set = np.column_stack((control_candidates, ts, T * np.ones(num_candidates)))
    return candidate_set


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_observed_histograms(
    exploration_history: Dict[Any, Tuple[Any, ...]],
    epsilon: float,
    xi: float,
    save_path: str
) -> None:
    """
    Plot histograms of observed safety and reset probabilities from the exploration history.

    Parameters:
        exploration_history (dict): History records containing observed probabilities.
        epsilon (float): Safety threshold parameter.
        xi (float): Reset threshold parameter.
        save_path (str): File path to save the plot.
    """
    if not exploration_history:
        logger.warning("No exploration history available to plot histograms.")
        return

    observed_safety = np.array([record[2] for record in exploration_history.values()])
    observed_reset = np.array([record[3] for record in exploration_history.values()])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(observed_safety, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=1 - epsilon, color='red', linestyle='--',
                label=f"Safety threshold: {1 - epsilon:.2f}")
    plt.xlabel("Observed Safety Probability")
    plt.ylabel("Frequency")
    plt.title("Observed Safety Histogram")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(observed_reset, bins=20, color='lightgreen', edgecolor='black')
    plt.axvline(x=1 - xi, color='red', linestyle='--',
                label=f"Reset threshold: {1 - xi:.2f}")
    plt.xlabel("Observed Reset Probability")
    plt.ylabel("Frequency")
    plt.title("Observed Reset Histogram")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved observed histograms to %s", save_path)
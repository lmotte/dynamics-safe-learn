#!/usr/bin/env python
"""
Script: 2d_second_order_plot_trajectories.py
----------------------------------------------

This script simulates trajectories for a set of control policies and aggregates them into a cumulative plot.

Key steps include:
  - Setting simulation parameters (e.g., total simulation time, number of steps, number of trajectories).
  - Generating multiple control policies using the generate_controls function.
  - Simulating trajectories for each control policy via Euler–Maruyama integration.
  - Aggregating the simulated paths, control angles, and time data.
  - Plotting the aggregated trajectories with the plot_paths_with_turbulence_list function.
  - Saving the resulting plot in a designated results directory with a filename that includes the random seed.

Dependencies:
  - utils.data (provides euler_maruyama, second_order_coefficients, generate_controls,
    plot_paths_with_turbulence_list, a_func)
  - numpy, matplotlib, seaborn, argparse, os

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data import (
    euler_maruyama,
    second_order_coefficients,
    generate_controls,
    plot_paths_with_turbulence_list,
    a_func,
)

# Set a global plot style.
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.figsize": (10, 6),  # Default figure size.
    "figure.dpi": 120,  # Resolution.
    "axes.titlesize": 16,  # Title font size.
    "axes.labelsize": 28,  # Axis label font size.
    "axes.labelweight": "bold",  # Axis label font weight.
    "axes.linewidth": 2,  # Thickness of the axis lines.
    "lines.linewidth": 2,  # Default line width.
    "lines.markersize": 8,  # Default marker size.
    "grid.linestyle": "--",  # Dashed grid lines.
    "grid.alpha": 0.7,  # Grid opacity.
    "font.size": 20,  # Base font size.
    "font.weight": "bold",  # Base font weight.
    "xtick.labelsize": 24,  # X tick label size.
    "ytick.labelsize": 24,  # Y tick label size.
    "xtick.major.width": 2,  # Major tick width on x-axis.
    "ytick.major.width": 2,  # Major tick width on y-axis.
    "xtick.minor.width": 1,  # Minor tick width on x-axis.
    "ytick.minor.width": 1  # Minor tick width on y-axis.
})

# ---------------------------
# Configuration
# ---------------------------
T = 20  # Total simulation time
n_steps = 500  # Number of time steps
n_paths = 100  # Number of trajectories per control
sigma_0 = 0.1  # Initial noise standard deviation
dim = 2  # State space dimension (position)
mu_0 = np.zeros(2)  # Initial position
safe_bounds = (-10, 10)  # Bounds of the safe region
reset_radius = 2.5  # Reset threshold radius
fixed_velocity = 2.0  # Fixed velocity for control
exploration_fraction = 0.3  # Exploration time fraction
n_control_steps = 5  # Number of discrete control changes
damping_factor = 0.5  # Damping factor
bounds = safe_bounds  # Boundaries for the safe region

# ---------------------------
# Parse Command-Line Arguments
# ---------------------------
parser = argparse.ArgumentParser(
    description="Run trajectory simulation with a specific random seed."
)
parser.add_argument("--seed", type=int, default=0, help="Random seed for the simulation")
args = parser.parse_args()
np.random.seed(args.seed)

# ---------------------------
# Generate Control Policies
# ---------------------------
K = 3  # Number of different control policies to test
controls = generate_controls(
    K, T, num_steps=n_control_steps, fixed_velocity=fixed_velocity,
    angle_range=(-np.pi / 2, np.pi / 2), mu_0=mu_0,
    exploration_fraction=exploration_fraction, damping_factor=damping_factor,
    bounds=safe_bounds
)

# ---------------------------
# Simulate Trajectories for Each Control
# ---------------------------
time_grid = np.linspace(0, T, n_steps)
iteration_paths, iteration_angles, iteration_step_times = [], [], []

for control_func, angles, step_times in controls:
    # Compute drift and diffusion functions based on the control.
    b_func, sigma_func = second_order_coefficients(control_func, a_func, dim)
    init_state = np.concatenate([mu_0, np.zeros(dim)])

    # Simulate trajectories using the Euler–Maruyama method.
    paths = euler_maruyama(b_func, sigma_func, n_steps, n_paths, T, 2 * dim,
                           mu_0=init_state, sigma_0=sigma_0)

    # Store simulated paths and related data.
    iteration_paths.append(paths)
    angles_array = np.full((n_paths, n_steps), angles[0])
    step_times_array = np.tile(time_grid, (n_paths, 1))
    iteration_angles.append(angles_array)
    iteration_step_times.append(step_times_array)

# ---------------------------
# Plot the Simulated Trajectories
# ---------------------------
results_dir = "true_trajectory_results"
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, f"true_trajectories_seed_{args.seed}.png")
x_range, y_range = (-12, 12), (-12, 12)
x_lim, y_lim = (-12, 12), (-12, 12)

# Concatenate all trajectories and related arrays.
all_paths = np.concatenate(iteration_paths, axis=0)
all_angles = np.concatenate(iteration_angles, axis=0)
all_step_times = np.concatenate(iteration_step_times, axis=0)

plot_paths_with_turbulence_list(
    paths=all_paths,
    angle_list=all_angles,
    step_time_list=all_step_times,
    exploration_fraction=exploration_fraction,
    a_func=a_func,
    x_range=x_range,
    y_range=y_range,
    resolution=200,
    save_path=save_path,
    title="Simulated True Trajectories for Three Controls",
    x_lim=x_lim,
    y_lim=y_lim,
    mu_0=mu_0,
    threshold_radius=reset_radius
)

print(f"Plot saved to: {save_path}")

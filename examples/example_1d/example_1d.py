#!/usr/bin/env python
"""
Script: simulate_and_plot_control_and_paths.py
------------------------------------------------
This script simulates 1D second-order systems under sinusoidal control.
For a given number of control functions (K), the script:
  - Generates a sinusoidal control function with a random angular frequency.
  - Computes and plots the control values over time.
  - Computes drift and diffusion functions using second_order_coefficients.
  - Simulates trajectories using Euler–Maruyama integration.
  - Saves plots of the position and velocity sample paths.

Dependencies:
  - utils.data: Provides euler_maruyama, sinusoidal_control, second_order_coefficients, plot_paths_1d, plot_control.
  - numpy, matplotlib, os

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import random
import numpy as np
from utils.data import (
    euler_maruyama,
    sinusoidal_control,
    second_order_coefficients,
    plot_paths_1d,
    plot_control
)

# ---------------------------
# Simulation Parameters
# ---------------------------
n = 1  # Control dimension (1D control)
n_steps = 100  # Number of time steps
n_paths = 10  # Number of trajectories to simulate
T = 10  # Total simulation time
time = np.linspace(0, T, n_steps)
mu_0 = np.zeros(2 * n)  # Initial state: [position, velocity]
sigma_0 = 0.1  # Initial standard deviation (noise level)
seed = 1
np.random.seed(seed)
random.seed(seed)

# ---------------------------
# Generate and Plot Controls and Trajectories
# ---------------------------
K = 5  # Number of different control functions to test

# Generate K random frequency scaling factors for the sinusoidal control.
# (Adjust the range as needed)
control_params = np.random.uniform(1, 4.0, K)

# Create directories for saving plots if they don't exist.
controls_dir = "plots/controls"
position_dir = "plots/position_paths"
velocity_dir = "plots/velocity_paths"
os.makedirs(controls_dir, exist_ok=True)
os.makedirs(position_dir, exist_ok=True)
os.makedirs(velocity_dir, exist_ok=True)

for k, w in enumerate(control_params):
    # Define the control function using the sinusoidal_control helper.
    u_func = sinusoidal_control(w=w, dim=n)


    # Define a constant noise function.
    def a_func(_):
        return 0.1 * np.ones(n)


    # Compute control values over the simulation time.
    control_values = np.array([u_func(t, np.zeros(n)) for t in time])

    # Save the control plot.
    control_plot_path = os.path.join(controls_dir, f"control_{k}.png")
    plot_control(
        T=time,
        control_values=control_values[:, 0],  # 1D array of control values
        save_path=control_plot_path,
        xlabel="Time",
        ylabel="Control Value",
        title="Control Values Over Time"
    )

    # Compute drift and diffusion functions based on the control function.
    b_func, sigma_func = second_order_coefficients(u_func, a_func, n)

    # Simulate trajectories using Euler–Maruyama.
    paths = euler_maruyama(b_func, sigma_func, n_steps, n_paths, T, 2 * n, mu_0, sigma_0)

    # Save the position plot (first component of state).
    position_plot_path = os.path.join(position_dir, f"position_paths_{k}.png")
    plot_paths_1d(
        T=time,
        paths=paths,
        save_path=position_plot_path,
        xlabel="Time",
        ylabel="Position",
        title="Sample Paths for Position (n = 1)"
    )

    # Save the velocity plot (remaining component of state).
    velocity_plot_path = os.path.join(velocity_dir, f"velocity_paths_{k}.png")
    plot_paths_1d(
        T=time,
        paths=paths[:, :, n:],  # Extract velocity component
        save_path=velocity_plot_path,
        xlabel="Time",
        ylabel="Velocity",
        title="Sample Paths for Velocity (n = 1)"
    )

    print(f"Plots for control function {k} (w = {w:.2f}) saved successfully.")

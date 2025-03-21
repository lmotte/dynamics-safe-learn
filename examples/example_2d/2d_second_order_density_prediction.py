"""
Script: 2d_second_order_density_prediction.py

This script:
  - Loads the previously saved safe model from "RESULT_DIR/safe_models_snapshots.pkl".
  - For each combination of gamma (kernel parameter), lambda (regularization),
    and R (density kernel parameter), it deep-copies the model, updates its hyperparameters,
    and recomputes the Gram matrix.
  - Then it plots the learned safety/reset maps and tests density prediction for 10 random controls,
    saving the plots and logging the computation times.

Dependencies:
  - methods.safe_sde_estimation (provides the SafeSDE class)
  - utils.data (provides a_func, generate_one_control, second_order_coefficients, euler_maruyama)
  - sklearn.metrics.pairwise.rbf_kernel
  - seaborn, matplotlib, numpy, pickle, etc.

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import time
import random
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib import colormaps, colors, cm
from matplotlib.colors import LinearSegmentedColormap
import logging
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel

# Import the SafeSDE class and required helper functions.
from utils.data import a_func, generate_one_control, second_order_coefficients, euler_maruyama

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Set the Results Directory (change this value as needed)
# ---------------------------
# Uncomment and set your desired result directory.
RESULT_DIR = "kernel_1.0_lam_1e-05_betas_0.7"

# ---------------------------
# Set a Global Plot Style
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
config_path = os.path.join(RESULT_DIR, "config.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

T = config["T"]  # Total simulation time
n_steps = config["n_steps"]  # Number of time steps
n_paths = config["n_paths"]  # Number of simulated trajectories per candidate
sigma_0 = config["sigma_0"]  # Initial standard deviation
dim = config["dim"]  # Dimension of the position; full state dim = 2*dim (positions & velocities)
mu_0 = np.array(config["mu_0"])  # Initial (target) position
safe_bounds = tuple(config["safe_bounds"])
reset_radius = config["reset_radius"]
fixed_velocity = config["fixed_velocity"]
damping_factor = config["damping_factor"]
exploration_fraction = config["exploration_fraction"]
n_control_steps = config["n_control_steps"]
seed: int = config["seed"]
np.random.seed(seed)
random.seed(seed)


# ---------------------------
# Define a Normalized Gaussian Kernel for Prediction
# ---------------------------
def normalized_gaussian_kernel(X, Y, gamma=1.0, control_scale=1.0, time_scale=1.0):
    """
    Gaussian (RBF) kernel with normalization applied to control and time features.

    Both X and Y should have shape (n_samples, control_dim+1), where the first columns are
    control parameters and the last column is time.
    """
    X_norm = X.copy()
    Y_norm = Y.copy()
    # Scale control features.
    X_norm[:, :-1] *= control_scale
    Y_norm[:, :-1] *= control_scale
    # Scale time feature.
    X_norm[:, -1] *= time_scale
    Y_norm[:, -1] *= time_scale
    return rbf_kernel(X_norm, Y_norm, gamma=gamma)


def plot_density_prediction(safe_model, theta, time_grid, T_val, chosen_times, n_true_paths=10, n_grid=50, R=1.0,
                            save_path=None):
    """
    For a given control theta, simulate trajectories up to time T and, for each chosen time,
    compute and plot the predicted density over a 2D grid.
    """
    # Generate a control function using the candidate theta.
    selected_u_func = generate_one_control(theta, T, num_steps=n_control_steps, fixed_velocity=fixed_velocity,
                                           mu_0=mu_0, exploration_fraction=exploration_fraction, bounds=safe_bounds,
                                           damping_factor=damping_factor)
    b_func, sigma_func = second_order_coefficients(selected_u_func, a_func, dim=2)
    init_state = np.concatenate([np.zeros(2), np.zeros(2)])
    # Simulate trajectories.
    paths = euler_maruyama(b_func, sigma_func, len(time_grid), n_true_paths, T_val, n=4,
                           mu_0=init_state, sigma_0=sigma_0)

    # Create evaluation grid.
    xs = np.linspace(-10, 10, n_grid)
    ys = np.linspace(-10, 10, n_grid)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    fig, ax = plt.subplots(figsize=(12, 8))
    t_min, t_max = min(chosen_times), max(chosen_times)
    cmap = colormaps['viridis']

    for t_val in chosen_times:
        # Select the closest time in time_grid.
        t_val = time_grid[np.argmin(np.abs(time_grid - t_val))]
        t_index = np.where(time_grid == t_val)[0][0]
        sim_states = paths[:, t_index, :2]
        predicted_density = safe_model.predict_system_density(theta, t_val, grid_points, R=R)
        predicted_density = predicted_density.reshape(grid_x.shape)
        num_levels = 14
        levels = np.linspace(predicted_density.min(), predicted_density.max(), num_levels)
        plot_levels = sorted(levels)[-8:]
        norm_time = (t_val - t_min) / (t_max - t_min) if (t_max - t_min) > 0 else 0.5
        time_color = cmap(norm_time)
        cmap_i = LinearSegmentedColormap.from_list("custom_cmap", [(1, 1, 1, 1), time_color], N=num_levels)
        ax.contourf(grid_x, grid_y, predicted_density, levels=plot_levels, cmap=cmap_i, alpha=0.6)
        ax.contour(grid_x, grid_y, predicted_density, levels=[plot_levels[0], plot_levels[-1]],
                   colors=[time_color], linewidths=1.0, linestyles="dashed")
        ax.scatter(sim_states[:, 0], sim_states[:, 1], color=time_color, s=15, zorder=2)
    # ax.set_title(f"Density prediction for θ={np.round(theta, 2)}\n(ε={eps}, ξ={xi})", fontsize=18)
    ax.set_xlabel(r"$X_1$", fontsize=18)
    ax.set_ylabel(r"$X_2$", fontsize=18)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.5)
    ax.tick_params(axis="both", which="minor", labelsize=12, width=1.0)
    norm = colors.Normalize(t_min, t_max)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Time t")
    cbar.set_label("Time t", fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # Save the figure.
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved density prediction plot to {save_path}")
    plt.close(fig)


def load_feasible_dataset(eps, xi, base_path=None):
    """
    Load the feasible candidates build_datasets for the given (ε, ξ) pair.
    Expected filename: feasible_candidates_dataset_eps{eps}_xi{xi}.pkl
    """
    filename = os.path.join(base_path, f"feasible_candidates_dataset_eps{eps}_xi{xi}.pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded feasible candidates build_datasets from {filename}")
        return data
    else:
        print(f"Feasible candidates build_datasets not found for eps={eps}, xi={xi}.")
        return None


# ---------------------------
# Update and Plot Learned Maps & Density Predictions
# ---------------------------
# In this script we:
# 1. Load a saved safe model.
# 2. Deep copy and update its hyperparameters.
# 3. Select a candidate control from a feasible dataset.
# 4. Plot density predictions and log computation times.

# --- Update model hyperparameters ---
# Load the safe models.
model_file = os.path.join(RESULT_DIR, "safe_models_snapshots.pkl")
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")
with open(model_file, "rb") as f:
    data = pickle.load(f)
models = data.get("models", {})

# Define hyperparameter values for this test.
gamma = 20.0  # Kernel gamma parameter.
lam = 1e-2  # Regularization parameter.
R = 5.0  # Density kernel parameter.
control_scale = 1.0
time_scale = 1.0


# --- Main script ---
def main():
    # For each model in the loaded models, update parameters if desired.
    for key, m in models.items():
        eps, xi = key
        # Deep copy the model so that changes do not affect the original.
        model = copy.deepcopy(m)
        # Update the model's hyperparameters.
        model.regularization = lam
        model.kernel_func = partial(normalized_gaussian_kernel,
                                    gamma=gamma, control_scale=control_scale, time_scale=time_scale)
        # Recompute the Gram matrix.
        model._update_gram_matrix()

        # Load a feasible dataset for this (ε, ξ) pair.
        # Here, we assume that a feasible dataset has been saved in RESULT_DIR.
        feasible_data = load_feasible_dataset(eps, xi, RESULT_DIR)
        if feasible_data is None:
            print(f"No feasible dataset available for eps={eps}, xi={xi}. Using model data.")
            # As a fallback, use the first candidate from model.gamma0.
            selected_candidate = model.gamma0[0]
        else:
            num_candidates = feasible_data["candidates"].shape[0]
            selected_idx = np.random.randint(0, num_candidates)
            selected_candidate = feasible_data["candidates"][selected_idx]
        selected_theta = selected_candidate[:2]
        print(f"Selected candidate with control {selected_theta} for eps={eps}, xi={xi}.")

        # Set up a time grid for density prediction.
        T_explo = 15
        time_grid = np.linspace(0, T_explo, 100)
        chosen_times = np.linspace(0, T_explo, T_explo + 1)

        # Perform and plot density prediction.
        t0 = time.time()
        density_save_path = os.path.join(RESULT_DIR,
                                         f"plots/density_prediction/"
                                         f"density_prediction_eps{eps}_xi{xi}_theta{selected_theta}.png")

        # Call the density prediction plotting function.
        plot_density_prediction(model, selected_theta, time_grid, T_explo, chosen_times,
                                n_true_paths=10, n_grid=50, R=R, save_path=density_save_path)
        elapsed_time = time.time() - t0
        logger.info(f"Density prediction for (eps, xi) = ({eps}, {xi}) took {elapsed_time:.3f} seconds.")


if __name__ == "__main__":
    main()

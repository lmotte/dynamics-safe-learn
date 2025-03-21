"""
Analysis Script for Safe Exploration Results
---------------------------------------------

This script loads:
  - CSV metrics from "RESULT_DIR/2d_second_order_results.csv"
  - Pickled safe models and snapshots from "RESULT_DIR/safe_models_snapshots.pkl"
  - (Optional) Pickled simulation data (iteration_paths, iteration_angles, iteration_step_times) from
    "RESULT_DIR/paths_data.pkl"

It then:
  1. Produces plots for safety & reset evolution, cumulative information gain,
     control coverage, and control parameter density.
  2. Processes the exploration history stored in the safe model for each (ε, ξ) pair.
  3. If simulation data is available, accumulates the paths up to selected snapshot iterations
     and uses the function plot_paths_with_turbulence_list to generate cumulative paths plots.

All plots are saved (not shown interactively) in the "RESULT_DIR/plots" directory.

Dependencies:
  - utils.data (euler_maruyama, second_order_coefficients, generate_one_control, plot_paths_with_turbulence_list,
   a_func)
  - utils.utils (sample_random_candidates_from_grid)

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import pickle
import logging
import argparse
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps, colors, cm
from matplotlib.colors import LinearSegmentedColormap

# Import functions from our safe learning modules.
from utils.data import (
    euler_maruyama,
    second_order_coefficients,
    generate_one_control,
    plot_paths_with_turbulence_list,
    a_func,
)
from utils.utils import sample_random_candidates_from_grid

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set a global plot style.
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
# Set the Results Directory
# ---------------------------
parser = argparse.ArgumentParser(description="Analyze Safe Exploration Results")
parser.add_argument("--results_dir", type=str, default="results", help="Directory with experiment results")
args = parser.parse_args()
RESULT_DIR: str = args.results_dir

# ---------------------------
# Load Global Simulation Parameters
# ---------------------------
config_path = os.path.join(RESULT_DIR, "config.pkl")
with open(config_path, "rb") as f:
    config: Dict[str, Any] = pickle.load(f)

T: float = config["T"]
n_steps: int = config["n_steps"]
n_paths: int = config["n_paths"]
sigma_0: float = config["sigma_0"]
dim: int = config["dim"]
mu_0: np.ndarray = np.array(config["mu_0"])
safe_bounds: Tuple[float, float] = tuple(config["safe_bounds"])
reset_radius: float = config["reset_radius"]
fixed_velocity: float = config["fixed_velocity"]
damping_factor: float = config["damping_factor"]
exploration_fraction: float = config["exploration_fraction"]
n_control_steps: int = config["n_control_steps"]


def log_config(cfg: Dict[str, Any]) -> None:
    """Logs the simulation configuration parameters."""
    logger.info("Simulation Configuration:")
    for key, value in cfg.items():
        logger.info(f"  {key}: {value}")


log_config(config)

# Define output folder for plots inside the results directory.
PLOTS_DIR: str = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------
# Data-Loading Functions
# ---------------------------
def load_metrics(csv_filename: str) -> pd.DataFrame:
    """Load CSV metrics into a pandas DataFrame."""
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Metrics file not found: {csv_filename}")
    return pd.read_csv(csv_filename)


def load_models(pickle_filename: str) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """Load safe models and snapshots from a pickle file."""
    if not os.path.exists(pickle_filename):
        raise FileNotFoundError(f"Pickle file not found: {pickle_filename}")
    with open(pickle_filename, "rb") as f:
        data = pickle.load(f)
    models = data.get("models", {})
    snapshots = data.get("snapshots", {})
    return models, snapshots


def load_paths_data(paths_filename: str) -> Optional[Dict[str, Any]]:
    """
    Load simulation data (iteration_paths, iteration_angles, iteration_step_times, plot_iterations)
    from a pickle file.
    """
    if not os.path.exists(paths_filename):
        logger.warning(f"Paths data file not found: {paths_filename}. Skipping cumulative paths plotting.")
        return None
    with open(paths_filename, "rb") as f:
        data = pickle.load(f)
    return data


def load_feasible_dataset(eps: float, xi: float, base_path: str) -> Optional[Any]:
    """
    Load the feasible candidates dataset for the given (ε, ξ) pair.
    Expected filename: feasible_candidates_dataset_eps{eps}_xi{xi}.pkl
    """
    filename = os.path.join(base_path, f"feasible_candidates_dataset_eps{eps}_xi{xi}.pkl")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded feasible candidates dataset from {filename}")
        return data
    else:
        print(f"Feasible candidates dataset not found for eps={eps}, xi={xi}.")
        return None


def process_exploration_history(safe_model: Any) -> Tuple[
    List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract exploration history from a safe_model.
    Expected history: {iteration: (theta, t_val, safety_prob, reset_prob, uncertainty_pred)}.

    Returns:
        tuple: (iterations, thetas, t_vals, safety_probs, reset_probs, uncertainties)
    """
    history = safe_model.exploration_history
    iterations = sorted(history.keys())
    thetas, t_vals, safety_probs, reset_probs, uncertainties = [], [], [], [], []
    for i in iterations:
        theta, t_val, safety_prob, reset_prob, uncertainty = history[i]
        thetas.append(theta)
        t_vals.append(t_val)
        safety_probs.append(safety_prob)
        reset_probs.append(reset_prob)
        uncertainties.append(uncertainty)
    return iterations, np.array(thetas), np.array(t_vals), np.array(safety_probs), np.array(reset_probs), np.array(
        uncertainties)


# ---------------------------
# Plotting Helper Functions
# ---------------------------
def save_plot(fig: plt.Figure, filename: str) -> None:
    """Save the figure to the specified filename and close it."""
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {filename}")


def plot_safety_reset(iterations: List[int], safety: np.ndarray, reset: np.ndarray, eps: float, xi: float) -> None:
    """Plot safety and reset probabilities over iterations with threshold indicators."""
    # Safety plot.
    fig, ax = plt.subplots(figsize=(10, 4))
    reset_color = sns.color_palette("deep")[2]
    crash_color = sns.color_palette("deep")[3]
    ax.plot(iterations, safety, marker="o", label="Safety probability", color=crash_color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Probability")
    ax.grid(True)
    ax.set_ylim(0, 1)
    if eps <= 1:
        ax.axhline(1 - eps, color='black', linestyle='--', label=f'Threshold: {1 - eps:.2f}')
    else:
        ax.axhline(1, color='black', linestyle='--', label='Threshold: ∞')
    os.makedirs(os.path.join(PLOTS_DIR, "constraints"), exist_ok=True)
    save_plot(fig, os.path.join(PLOTS_DIR, f"constraints/safety_eps{eps}_xi{xi}.png"))

    # Reset plot.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iterations, reset, marker="s", label="Reset probability", color=reset_color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Probability")
    ax.grid(True)
    ax.set_ylim(0, 1)
    if xi <= 1:
        ax.axhline(1 - xi, color='black', linestyle='--', label=f'Threshold: {1 - xi:.2f}')
    else:
        ax.axhline(1, color='black', linestyle='--', label='Threshold: ∞')
    os.makedirs(os.path.join(PLOTS_DIR, "constraints"), exist_ok=True)
    save_plot(fig, os.path.join(PLOTS_DIR, f"constraints/reset_eps{eps}_xi{xi}.png"))


def plot_cumulative_info_gain(iterations: List[int], uncertainties: np.ndarray, eps: float, xi: float) -> None:
    """
    Compute and plot cumulative information gain over iterations.
    I_i = 0.5 * sum_{j=0}^{i} log(1 + uncertainty_j / ((j+1)*REGULARIZATION))
    """
    REGULARIZATION = 1e-5
    log_terms = 0.5 * np.log([1 + u / ((i + 1) * REGULARIZATION) for i, u in enumerate(uncertainties)])
    cumulative_info_gain = np.cumsum(log_terms)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, cumulative_info_gain, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Information Gain")
    ax.set_title(f"Cumulative Information Gain (ε={eps}, ξ={xi})")
    ax.grid(True)
    os.makedirs(os.path.join(PLOTS_DIR, "information_gain"), exist_ok=True)
    save_plot(fig, os.path.join(PLOTS_DIR, f"information_gain/cumulative_info_gain_eps{eps}_xi{xi}.png"))


def plot_all_cumulative_info_gain(models: Dict[Tuple[float, float], Any]) -> None:
    """
    Plot cumulative information gain curves for all models on the same plot.
    For each safe model, the cumulative information gain is computed as:
      I_i = 0.5 * sum_{j=0}^{i} log(1 + uncertainty_j / ((j+1)*REGULARIZATION))
    The color of each curve is determined by its epsilon value using the 'crest' colormap.
    """
    REGULARIZATION = 1e-4
    fig, ax = plt.subplots(figsize=(10, 6))
    epsilons = [eps for (eps, xi) in models.keys()]
    min_eps, max_eps = min(epsilons), max(epsilons)
    norm = plt.Normalize(vmin=min_eps, vmax=max_eps)
    cmap = sns.color_palette("crest", as_cmap=True)

    for (eps, xi), safe_model in models.items():
        try:
            iterations, _, _, _, _, uncertainties = process_exploration_history(safe_model)
        except Exception:
            print(f"Skipping model (ε={eps}, ξ={xi}) due to missing exploration history.")
            continue
        log_terms = 0.5 * np.log([1 + u / ((i + 1) * REGULARIZATION) for i, u in enumerate(uncertainties)])
        cum_info_gain = np.cumsum(log_terms)
        if eps < 2:
            color = cmap(norm(eps))
            ax.plot(iterations, cum_info_gain, linewidth=2, color=color, label=f"$\\epsilon=\\xi={eps}$")
        else:
            color = 'black'
            ax.plot(iterations, cum_info_gain, linewidth=2, linestyle='--', color=color,
                    label=r"$\epsilon=\xi=+\infty$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Information gain")
    ax.grid(True)
    ax.legend()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("ε = ξ")
    out_path = os.path.join(PLOTS_DIR, "information_gain", "cumulative_info_gain_all_models.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cumulative information gain comparison plot to {out_path}")


def plot_control_coverage(thetas: np.ndarray, eps: float, xi: float) -> None:
    """Scatter plot of control parameters."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(thetas[:, 0], thetas[:, 1], alpha=0.6)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    margin = 0.2
    ax.set(xlim=(-np.pi - margin, np.pi + margin), ylim=(-np.pi / 2 - margin, np.pi / 2 + margin))
    plt.tight_layout()
    os.makedirs(os.path.join(PLOTS_DIR, "control_cover"), exist_ok=True)
    save_plot(fig, os.path.join(PLOTS_DIR, f"control_cover/control_coverage_eps{eps}_xi{xi}.png"))


def plot_thetas_with_time(safe_model: Any, eps: float, xi: float) -> None:
    """
    Plot theta values with a colormap indicating their associated time.
    Produces both a 2D and a 3D scatter plot.
    """
    control_dim = safe_model.control_dim
    thetas = safe_model.data[:, :control_dim]
    times = safe_model.data[:, control_dim]

    # 2D Scatter Plot.
    fig2d, ax2d = plt.subplots(figsize=(6, 6))
    sc = ax2d.scatter(thetas[:, 0], thetas[:, 1], c=times, cmap='viridis', s=50, edgecolor='k')
    ax2d.set_xlabel("θ[0]")
    ax2d.set_ylabel("θ[1]")
    ax2d.set_title(f"Theta Values Colored by Time (ε={eps}, ξ={xi})")
    cbar = fig2d.colorbar(sc, ax=ax2d)
    cbar.set_label("Time")
    save_plot(fig2d, os.path.join(PLOTS_DIR, f"control_cover/control_coverage_with_time_eps{eps}_xi{xi}_2d.png"))

    # 3D Scatter Plot.
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    p = ax3d.scatter(thetas[:, 0], thetas[:, 1], times, c=times, cmap='viridis', s=50, edgecolor='k')
    ax3d.set_xlabel("θ[0]")
    ax3d.set_ylabel("θ[1]")
    ax3d.set_zlabel("Time")
    ax3d.set_title(f"3D Theta Values with Time (ε={eps}, ξ={xi})")
    cbar3d = fig3d.colorbar(p, ax=ax3d, pad=0.1)
    cbar3d.set_label("Time")
    save_plot(fig3d, os.path.join(PLOTS_DIR, f"control_cover/control_coverage_with_time_eps{eps}_xi{xi}_3d.png"))


def plot_control_density(thetas: np.ndarray, eps: float, xi: float) -> None:
    """Plot 2D density (heat map) of control parameters."""
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(x=thetas[:, 0], y=thetas[:, 1], cmap="viridis", fill=True, thresh=0.05, ax=ax)
    ax.set_xlabel("θ[0]")
    ax.set_ylabel("θ[1]")
    ax.set_title(f"Control Density (ε={eps}, ξ={xi})")
    margin = 0
    ax.set(xlim=(-np.pi - margin, np.pi + margin), ylim=(-np.pi / 2 - margin, np.pi / 2 + margin))
    os.makedirs(os.path.join(PLOTS_DIR, "control_cover"), exist_ok=True)
    save_plot(fig, os.path.join(PLOTS_DIR, f"control_cover/control_density_eps{eps}_xi{xi}.png"))


def plot_selected_thetas(safe_model: Any, chosen_times: List[float], tolerance: float = 0.1,
                         save_path: Optional[str] = None) -> None:
    """
    Plot theta values for samples with t-values close to each chosen time.
    """
    thetas = safe_model.data[:, :safe_model.control_dim]
    t_vals = safe_model.data[:, safe_model.control_dim]
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(chosen_times)))
    for ct, color in zip(chosen_times, colors):
        indices = np.where(np.abs(t_vals - ct) < tolerance)[0]
        if indices.size > 0:
            ax.scatter(thetas[indices, 0], thetas[indices, 1], color=color, label=f"t ≈ {ct:.2f}")
    ax.set_xlabel("θ[0]")
    ax.set_ylabel("θ[1]")
    ax.set_title("Selected (θ, t) Candidates by Time")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_cumulative_paths(paths_data: Dict[Tuple[float, float], Any], eps: float, xi: float,
                          n_plot_paths: int = 1) -> None:
    """
    For each snapshot iteration stored in paths_data, accumulate simulation data and plot cumulative paths.
    Only the first n_plot_paths trajectories from each control are plotted.
    """
    x_range, y_range = (-12, 12), (-12, 12)
    resolution = 200
    x_lim, y_lim = (-12, 12), (-12, 12)
    iteration_paths = paths_data[(eps, xi)]["iteration_paths"]
    iteration_angles = paths_data[(eps, xi)]["iteration_angles"]
    iteration_step_times = paths_data[(eps, xi)]["iteration_step_times"]
    plot_iterations = paths_data[(eps, xi)]["plot_iterations"]
    for idx in plot_iterations:
        if idx >= len(iteration_paths):
            print(f"Iteration index {idx} exceeds available data. Skipping.")
            continue
        cum_paths = np.concatenate([p[:n_plot_paths] for p in iteration_paths[:idx + 1]], axis=0)
        cum_angles = np.concatenate([p[:n_plot_paths] for p in iteration_angles[:idx + 1]], axis=0)
        cum_step_times = np.concatenate([p[:n_plot_paths] for p in iteration_step_times[:idx + 1]], axis=0)
        title = f"Combined Paths Up To Iteration {idx + 1} (ε={eps}, ξ={xi})"
        save_path_full = os.path.join(PLOTS_DIR, f"paths/combined_paths_up_to_iteration_{idx + 1}_eps{eps}_xi{xi}.png")
        plot_paths_with_turbulence_list(
            paths=cum_paths,
            angle_list=cum_angles,
            step_time_list=cum_step_times,
            exploration_fraction=exploration_fraction,
            a_func=a_func,
            x_range=x_range,
            y_range=y_range,
            resolution=resolution,
            save_path=save_path_full,
            title=title,
            x_lim=x_lim,
            y_lim=y_lim,
            mu_0=mu_0,
            threshold_radius=reset_radius
        )
        print(f"Saved cumulative paths plot up to iteration {idx + 1}.")


def plot_learned_maps(safe_model: Any, control_range: Tuple[Tuple[float, float], Tuple[float, float]],
                      T: float, n_grid: int = 50, time_samples: int = 100,
                      save_path_safety_2d: Optional[str] = None, save_path_reset_2d: Optional[str] = None,
                      save_path_safety_3d: Optional[str] = None, save_path_reset_3d: Optional[str] = None) -> None:
    """
    Plot learned safety and reset maps as 2D contour plots and 3D surface plots.
    """
    theta0_range = np.linspace(control_range[0][0], control_range[0][1], n_grid)
    theta1_range = np.linspace(control_range[1][0], control_range[1][1], n_grid)
    Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
    grid_points = np.column_stack([Theta0.ravel(), Theta1.ravel()])
    time_grid_vals = np.linspace(0, T, time_samples)
    safety_values = np.zeros((time_samples, grid_points.shape[0]))
    for i, t_val in enumerate(time_grid_vals):
        pred_safety, _ = safe_model.predict(grid_points, t_val, target='safety')
        pred_safety = np.clip(pred_safety, 0, 1)
        safety_values[i, :] = pred_safety.ravel()
    safety_map = np.min(safety_values, axis=0).reshape((n_grid, n_grid))
    vmin_safety, vmax_safety = safety_map.min(), safety_map.max()
    fig_safety_2d, ax = plt.subplots(figsize=(14, 6))
    cmap_safety = sns.color_palette("flare", as_cmap=True)
    c1 = ax.contourf(Theta0, Theta1, safety_map, cmap=cmap_safety, vmin=vmin_safety, vmax=vmax_safety)
    fig_safety_2d.colorbar(c1, ax=ax)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(control_range[0])
    ax.set_ylim(control_range[1])
    plt.tight_layout()
    if save_path_safety_2d is not None:
        fig_safety_2d.savefig(save_path_safety_2d, bbox_inches="tight")
        print(f"Saved Safety Map 2D plot to {save_path_safety_2d}")
    plt.close(fig_safety_2d)
    fig_reset_2d, ax = plt.subplots(figsize=(14, 6))
    cmap_reset = sns.color_palette("crest", as_cmap=True)
    pred_reset, _ = safe_model.predict(grid_points, T, target='reset')
    pred_reset = np.clip(pred_reset, 0, 1)
    reset_map = pred_reset.reshape((n_grid, n_grid))
    vmin_reset, vmax_reset = reset_map.min(), reset_map.max()
    c2 = ax.contourf(Theta0, Theta1, reset_map, cmap=cmap_reset, vmin=vmin_reset, vmax=vmax_reset)
    fig_reset_2d.colorbar(c2, ax=ax)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(control_range[0])
    ax.set_ylim(control_range[1])
    plt.tight_layout()
    if save_path_reset_2d is not None:
        fig_reset_2d.savefig(save_path_reset_2d, bbox_inches="tight")
        print(f"Saved Reset Map 2D plot to {save_path_reset_2d}")
    plt.close(fig_reset_2d)
    fig_safety_3d = plt.figure(figsize=(7, 6))
    ax3d = fig_safety_3d.add_subplot(111, projection='3d')
    surf1 = ax3d.plot_surface(Theta0, Theta1, safety_map, cmap='viridis', edgecolor='none',
                              vmin=vmin_safety, vmax=vmax_safety)
    ax3d.set_xlabel(r"$\theta_1$")
    ax3d.set_ylabel(r"$\theta_2$")
    ax3d.set_zlabel("Safety")
    ax3d.set_xlim(control_range[0])
    ax3d.set_ylim(control_range[1])
    fig_safety_3d.colorbar(surf1, ax=ax3d, shrink=0.5, aspect=5)
    plt.tight_layout()
    if save_path_safety_3d is not None:
        fig_safety_3d.savefig(save_path_safety_3d, bbox_inches="tight")
        print(f"Saved Safety Map 3D plot to {save_path_safety_3d}")
    plt.close(fig_safety_3d)
    fig_reset_3d = plt.figure(figsize=(7, 6))
    ax3d = fig_reset_3d.add_subplot(111, projection='3d')
    surf2 = ax3d.plot_surface(Theta0, Theta1, reset_map, cmap='plasma', edgecolor='none',
                              vmin=vmin_reset, vmax=vmax_reset)
    ax3d.set_xlabel(r"$\theta_1$")
    ax3d.set_ylabel(r"$\theta_2$")
    ax3d.set_zlabel("Reset")
    ax3d.set_xlim(control_range[0])
    ax3d.set_ylim(control_range[1])
    fig_reset_3d.colorbar(surf2, ax=ax3d, shrink=0.5, aspect=5)
    plt.tight_layout()
    if save_path_reset_3d is not None:
        fig_reset_3d.savefig(save_path_reset_3d, bbox_inches="tight")
        print(f"Saved Reset Map 3D plot to {save_path_reset_3d}")
    plt.close(fig_reset_3d)


def plot_true_observations(control_range: Tuple[Tuple[float, float], Tuple[float, float]], true_values_file: str,
                           save_path: str) -> None:
    """
    Plot true safety and reset observations with axis limits matching the learned maps.
    """
    if not os.path.exists(true_values_file):
        print(f"True values file not found: {true_values_file}. Skipping true observations plot.")
        return
    with open(true_values_file, "rb") as f:
        true_data = pickle.load(f)
    candidates = true_data["candidates"]
    true_safety = true_data["estimated_safety"]
    true_reset = true_data["estimated_reset"]
    fig, (ax_obs1, ax_obs2) = plt.subplots(1, 2, figsize=(14, 6))
    xlim, ylim = control_range[0], control_range[1]
    sc_obs1 = ax_obs1.scatter(candidates[:, 0], candidates[:, 1], c=true_safety, cmap='viridis', edgecolor='w', s=50)
    fig.colorbar(sc_obs1, ax=ax_obs1)
    ax_obs1.set_title("True Safety Observations")
    ax_obs1.set_xlabel("θ[0]")
    ax_obs1.set_ylabel("θ[1]")
    ax_obs1.set_xlim(xlim)
    ax_obs1.set_ylim(ylim)
    sc_obs2 = ax_obs2.scatter(candidates[:, 0], candidates[:, 1], c=true_reset, cmap='plasma', edgecolor='w', s=50)
    fig.colorbar(sc_obs2, ax=ax_obs2)
    ax_obs2.set_title("True Reset Observations")
    ax_obs2.set_xlabel("θ[0]")
    ax_obs2.set_ylabel("θ[1]")
    ax_obs2.set_xlim(xlim)
    ax_obs2.set_ylim(ylim)
    fig.suptitle("True Observations of Safety and Reset", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved true observations plot to {save_path}")


def plot_density_prediction(safe_model: Any, theta: np.ndarray, time_grid: np.ndarray, T: float,
                            chosen_times: List[float], eps: float, xi: float, n_true_paths: int = 100,
                            n_grid: int = 50, R: float = 1.0, save_path: Optional[str] = None) -> None:
    """
    For a given control theta, simulate trajectories up to time T and, for each chosen time,
    compute and plot the predicted density over a 2D grid.
    """
    selected_u_func = generate_one_control(theta, T, num_steps=n_control_steps,
                                           fixed_velocity=fixed_velocity, mu_0=mu_0,
                                           exploration_fraction=exploration_fraction, bounds=safe_bounds,
                                           damping_factor=damping_factor)
    b_func, sigma_func = second_order_coefficients(selected_u_func, a_func, dim)
    init_state = np.concatenate([mu_0, np.zeros(dim)])
    paths = euler_maruyama(b_func, sigma_func, n_steps, n_true_paths, T, 2 * dim,
                           mu_0=init_state, sigma_0=sigma_0)
    xs = np.linspace(safe_bounds[0], safe_bounds[1], n_grid)
    ys = np.linspace(safe_bounds[0], safe_bounds[1], n_grid)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    fig, ax = plt.subplots(figsize=(8, 8))
    t_min, t_max = min(chosen_times), max(chosen_times)
    cmap = colormaps['viridis']
    for t_val in chosen_times:
        if t_val not in time_grid:
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
        ax.contourf(grid_x, grid_y, predicted_density, levels=plot_levels, cmap=cmap_i)
        ax.contour(grid_x, grid_y, predicted_density, levels=[plot_levels[0], plot_levels[-1]],
                   colors=[time_color], linewidths=1.0, linestyles="dashed")
        ax.scatter(sim_states[:, 0], sim_states[:, 1], color=time_color, s=15, label=f"t={t_val:.2f}")
    ax.set_title(f"Density prediction for θ={np.round(theta)}\n(ε={eps}, ξ={xi})", fontsize=18)
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_xlim(safe_bounds)
    ax.set_ylim(safe_bounds)
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
    save_filename = os.path.join(PLOTS_DIR, f"density_prediction/density_prediction_theta{theta}_eps{eps}_xi{xi}.png") \
        if save_path is None else os.path.join(save_path,
                                               f"density_prediction/"
                                               f"density_prediction_theta{theta}_eps{eps}_xi{xi}.png")
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved density prediction plot to {save_filename}")


def evaluate_model_accuracy(safe_model: Any, N: int = 100, T: float = 12,
                            n_control_steps: int = 2, n_steps: int = 500, n_paths: int = 100,
                            sigma_0: float = 0.05, dim: int = 2, mu_0: Optional[np.ndarray] = None,
                            fixed_velocity: float = 2.0, exploration_fraction: float = 0.5,
                            safe_bounds: Tuple[float, float] = (-10, 10), reset_radius: float = 2.5,
                            control_range: Tuple[Tuple[float, float], Tuple[float, float]] = (
                                    (-np.pi, np.pi), (-np.pi / 2, np.pi / 2)),
                            true_values_file: Optional[str] = None) -> Tuple[float, float]:
    """
    Evaluate model accuracy by comparing safe_model.predict outputs to Monte Carlo estimates.
    """
    if mu_0 is None:
        mu_0 = np.zeros(dim)
    time_grid = np.linspace(0, T, n_steps)
    control_dim = safe_model.control_dim
    if true_values_file is not None and os.path.exists(true_values_file):
        with open(true_values_file, "rb") as f:
            data = pickle.load(f)
        candidates = data["candidates"]
        estimated_safety_arr = data["estimated_safety"]
        estimated_reset_arr = data["estimated_reset"]
        print("Loaded true values dataset from file.")
    else:
        candidates = sample_random_candidates_from_grid(control_range, time_grid, T, num_candidates=N)
        estimated_safety_list, estimated_reset_list = [], []
        for candidate in candidates:
            theta = candidate[:control_dim]
            t_val = candidate[2]
            selected_u_func = generate_one_control(theta, T, num_steps=n_control_steps,
                                                   fixed_velocity=fixed_velocity, mu_0=mu_0,
                                                   exploration_fraction=exploration_fraction, bounds=safe_bounds,
                                                   damping_factor=damping_factor)
            b_func, sigma_func = second_order_coefficients(selected_u_func, a_func, dim)
            init_state = np.concatenate([mu_0, np.zeros(dim)])
            paths = euler_maruyama(b_func, sigma_func, n_steps, n_paths, T, 2 * dim,
                                   mu_0=init_state, sigma_0=sigma_0)
            est_safety = safe_model.compute_safety_probability(paths, safe_bounds=safe_bounds,
                                                               t_val=t_val, time_grid=time_grid)
            est_reset = safe_model.compute_reset_probability(paths, reset_radius=reset_radius,
                                                             t_val=T, time_grid=time_grid)

            estimated_safety_list.append(est_safety)
            estimated_reset_list.append(est_reset)
        estimated_safety_arr = np.array(estimated_safety_list)
        estimated_reset_arr = np.array(estimated_reset_list)
    theta_feas = candidates[:, :control_dim]
    t_feas = candidates[:, 2]
    pred_safety, _ = safe_model.predict(theta_feas, t_feas, target='safety')
    pred_reset, _ = safe_model.predict(theta_feas, t_feas, target='reset')
    pred_safety = np.clip(pred_safety, 0, 1)
    pred_reset = np.clip(pred_reset, 0, 1)
    errors_safety = pred_safety.ravel() - estimated_safety_arr
    errors_reset = pred_reset.ravel() - estimated_reset_arr
    mse_safety = np.mean(errors_safety ** 2)
    mse_reset = np.mean(errors_reset ** 2)
    std_safety = np.std(errors_safety ** 2)
    std_reset = np.std(errors_reset ** 2)
    num_candidates = candidates.shape[0]
    print("Number of feasible candidates: ", num_candidates)
    print("Safety prediction MSE: {:.4f} (std: {:.4f})".format(mse_safety, std_safety))
    print("Reset prediction MSE:  {:.4f} (std: {:.4f})".format(mse_reset, std_reset))
    return mse_safety, mse_reset


# ---------------------------
# Main Loop: Run Analysis
# ---------------------------
def main() -> None:
    csv_filename = os.path.join(RESULT_DIR, "2d_second_order_results.csv")
    df_metrics = load_metrics(csv_filename)
    pickle_filename = os.path.join(RESULT_DIR, "safe_models_snapshots.pkl")
    models, snapshots = load_models(pickle_filename)

    print("==== Metrics Summary from CSV ====")
    print(df_metrics)

    max_eps = -np.inf
    model_max_eps = None
    for key, safe_model in models.items():
        eps, xi = key
        if eps > max_eps:
            max_eps = eps
            model_max_eps = safe_model
    if model_max_eps is not None:
        m = model_max_eps.control_dim
        global_control_range = tuple(
            (np.min(model_max_eps.data[:, d]), np.max(model_max_eps.data[:, d]))
            for d in range(m)
        )
        print(f"Global control range (from ε = {max_eps}): {global_control_range}")
    else:
        print("No models found!")
        return
    global_control_range = ((-np.pi, np.pi), (-np.pi / 2, np.pi / 2))

    plot_all_cumulative_info_gain(models)

    for key, safe_model in models.items():
        eps, xi = key
        print(f"\n==== Results for ε = {eps}, ξ = {xi} ====")
        df_match = df_metrics[(df_metrics["epsilon"] == eps) & (df_metrics["xi"] == xi)]
        if not df_match.empty:
            print(df_match.to_string(index=False))
        else:
            print("No matching CSV metrics found for this (ε, ξ) pair.")
        try:
            iterations, thetas, t_vals, safety, reset, uncertainties = process_exploration_history(safe_model)
        except AttributeError:
            print("Safe model does not have exploration_history. Skipping...")
            continue
        plot_safety_reset(iterations, safety, reset, eps, xi)
        plot_cumulative_info_gain(iterations, uncertainties, eps, xi)
        plot_control_coverage(thetas, eps, xi)
        plot_control_density(thetas, eps, xi)
        plot_thetas_with_time(safe_model, eps, xi)
        if not df_match.empty:
            total_time = df_match["total_time"].values[0]
            iters = df_match["iterations"].values[0]
            print(f"Total iterations: {iters}")
            print(f"Total computation time: {total_time:.2f} seconds")
        else:
            print("No computational efficiency data available from CSV.")
        print(f"\n==== Evaluating model accuracy for ε = {eps}, ξ = {xi} ====")
        true_values_file = os.path.join(RESULT_DIR, "true_values_dataset.pkl")
        _ = evaluate_model_accuracy(safe_model, N=100, control_range=global_control_range,
                                    true_values_file=true_values_file)
        if safe_model.control_dim == 2:
            os.makedirs(os.path.join(PLOTS_DIR, "learned_maps"), exist_ok=True)
            save_path_safety_2d = os.path.join(PLOTS_DIR, f"learned_maps/2d_learned_safety_map_eps{eps}_xi{xi}.png")
            save_path_reset_2d = os.path.join(PLOTS_DIR, f"learned_maps/2d_learned_reset_map_eps{eps}_xi{xi}.png")
            save_path_safety_3d = os.path.join(PLOTS_DIR, f"learned_maps/3d_learned_safety_map_eps{eps}_xi{xi}.png")
            save_path_reset_3d = os.path.join(PLOTS_DIR, f"learned_maps/3d_learned_reset_map_eps{eps}_xi{xi}.png")
            plot_learned_maps(safe_model, global_control_range, T,
                              save_path_safety_2d=save_path_safety_2d,
                              save_path_reset_2d=save_path_reset_2d,
                              save_path_safety_3d=save_path_safety_3d,
                              save_path_reset_3d=save_path_reset_3d)
        # Uncomment the following to plot density predictions or cumulative paths if desired.
        # n_chosen_times = 10
        # chosen_times = np.linspace(0, T, n_chosen_times)
        # plot_density_prediction(safe_model, thetas[0], np.linspace(0, T, n_steps), T, chosen_times, eps, xi,
        #                         n_true_paths=10, n_grid=50, R=1.0, save_path=PLOTS_DIR)
        # if paths_data is not None:
        #     os.makedirs(os.path.join(PLOTS_DIR, "paths"), exist_ok=True)
        #     plot_cumulative_paths(paths_data, eps, xi, n_plot_paths=10)
        #     print(f"Saved cumulative paths plot for (ε={eps}, ξ={xi}).")
        # else:
        #     print("No cumulative paths data available. Skipping cumulative paths plotting.")


if __name__ == "__main__":
    main()

"""
utils/data.py
--------------

This module provides a suite of utility functions for simulating controlled stochastic
differential equations (SDEs), generating control functions, and visualizing simulation results. It is
designed to support safe learning experiments for controlled stochastic systems.

Key functionalities include:

1. SDE Simulation:
    - Implementation of the Euler–Maruyama method for simulating multi-dimensional SDEs.
    - Generation of drift and diffusion coefficients for second-order systems.
    - Construction of sinusoidal and other control functions.

2. Control Generation:
    - Generation of multiple control functions with step-wise angle changes and exploration/return
      phases.
    - Utility functions for generating both families of controls and single control functions based on
      candidate parameters.

3. Plotting Utilities:
    - 1D and 2D trajectory visualization.
    - Visualization of control functions and their evolution over time (including control angles).
    - Plotting trajectories with markers at control change points and with overlaid turbulence heatmaps.
    - Shading of unsafe regions and visualization of threshold boundaries.

4. Additional Utilities:
    - Computation of trajectory lengths.
    - Generation of grid-based sigma (noise) fields for turbulence visualization.

Usage Example:
    from utils import data
    # Simulate sample paths using the Euler–Maruyama method.
    paths = data.euler_maruyama(b, sigma, n_steps, n_paths, T, n, mu_0, sigma_0)
    # Generate control functions.
    controls = data.generate_control_functions(K, T, num_steps=3)
    # Plot 1D sample paths.
    data.plot_paths_1d(T_array, paths, save_path="sample_paths.png")

Dependencies:
    - numpy
    - seaborn
    - matplotlib
    - logging

Author: Luc Brogat-Motte
Date: 2025
"""

import logging
from typing import Callable, Tuple, Union, Optional, List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

# Set a global plot style.
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.figsize": (10, 6),      # Default figure size.
    "figure.dpi": 120,              # Resolution.
    "axes.titlesize": 16,           # Title font size.
    "axes.labelsize": 28,           # Axis label font size.
    "axes.labelweight": "bold",     # Axis label font weight.
    "axes.linewidth": 2,            # Thickness of the axis lines.
    "lines.linewidth": 2,           # Default line width.
    "lines.markersize": 8,          # Default marker size.
    "grid.linestyle": "--",         # Dashed grid lines.
    "grid.alpha": 0.7,              # Grid opacity.
    "font.size": 20,                # Base font size.
    "font.weight": "bold",          # Base font weight.
    "xtick.labelsize": 24,          # X tick label size.
    "ytick.labelsize": 24,          # Y tick label size.
    "xtick.major.width": 2,         # Major tick width on x-axis.
    "ytick.major.width": 2,         # Major tick width on y-axis.
    "xtick.minor.width": 1,         # Minor tick width on x-axis.
    "ytick.minor.width": 1          # Minor tick width on y-axis.
})

# =============================================================================
# Simulation Functions
# =============================================================================

def euler_maruyama(
    b: Callable[[float, np.ndarray], np.ndarray],
    sigma: Callable[[float, np.ndarray], np.ndarray],
    n_steps: int,
    n_paths: int,
    T: float,
    n: int,
    mu_0: np.ndarray,
    sigma_0: float,
    coef_save: bool = False,
    time_display: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simulate sample paths of a multidimensional SDE using the Euler–Maruyama method.

    Args:
        b: Drift coefficient function, b(t, x).
        sigma: Diffusion coefficient function, sigma(t, x).
        n_steps: Number of time steps.
        n_paths: Number of sample paths to simulate.
        T: Total duration of the simulation.
        n: Dimension of the state variable.
        mu_0: Initial mean state (length n).
        sigma_0: Standard deviation for the initial state.
        coef_save: If True, also return coefficient arrays.
        time_display: If True, print simulation progress.

    Returns:
        If coef_save is False, returns an array of shape (n_paths, n_steps, n)
        containing the simulated paths. Otherwise, returns a tuple:
            (paths, B_save, S_save) where:
                - paths: Simulated paths.
                - B_save: Drift coefficients for each path and time step.
                - S_save: Diffusion coefficients for each path and time step.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    paths = np.zeros((n_paths, n_steps, n))
    B_save = np.zeros((n_paths, n_steps, n)) if coef_save else None
    S_save = np.zeros((n_paths, n_steps, 1)) if coef_save else None

    for i in range(n_paths):
        x = np.zeros((n_steps, n))
        cov_0 = sigma_0 ** 2 * np.eye(n)
        x[0] = np.random.multivariate_normal(mu_0, cov_0)
        b_save = np.zeros((n_steps, n)) if coef_save else None
        s_save = np.zeros((n_steps, 1)) if coef_save else None

        for j in range(n_steps - 1):
            dW = np.random.normal(0, 1, size=n) * sqrt_dt
            t_val = dt * j
            b_t = b(t_val, x[j])
            s_t = sigma(t_val, x[j])
            x[j + 1] = x[j] + b_t * dt + s_t * dW
            if coef_save:
                b_save[j + 1] = b_t
                s_save[j + 1] = s_t

        paths[i] = x
        if coef_save:
            B_save[i] = b_save
            S_save[i] = s_save

        if time_display:
            print(f"{int(i / n_paths * 100)}%", end=" ", flush=True)

    return (paths, B_save, S_save) if coef_save else paths


def second_order_coefficients(
    u_func: Callable[[float, np.ndarray], np.ndarray],
    a_func: Callable[[np.ndarray], np.ndarray],
    dim: int
) -> Tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[float, np.ndarray], np.ndarray]]:
    """
    Returns the drift and diffusion coefficients for a second-order system:

        dX(t) = V(t) dt,
        dV(t) = u(t, x) dt + a(X(t)) dW(t).

    Args:
        u_func: Control function u(t, x) returning a vector (length dim).
        a_func: Noise amplitude function a(x) returning a vector (length dim).
        dim: Dimension of the position space.

    Returns:
        Tuple (b, sigma_func) where:
            - b(t, x): Drift function.
            - sigma_func(t, x): Diffusion function (applied only to velocity components).
    """
    def b(t: float, x: np.ndarray) -> np.ndarray:
        position = x[:dim]
        velocity = x[dim:]
        dx = velocity
        dv = u_func(t, x)
        return np.concatenate([dx, dv])

    def sigma_func(t: float, x: np.ndarray) -> np.ndarray:
        position = x[:dim]
        noise = a_func(position)
        return np.concatenate([np.zeros(dim), noise])

    return b, sigma_func


def sinusoidal_control(w: float, dim: int) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Create a sinusoidal control function.

    Args:
        w: Frequency scaling factor.
        dim: Dimension of the control vector.

    Returns:
        A function u(t, x) returning a vector of length 'dim' with sinusoidal values.
    """
    return lambda t, x: np.array([np.sin(w * t / 10 * np.pi) for _ in range(dim)])


# =============================================================================
# Control Generation Functions
# =============================================================================

def generate_control_functions(
    K: int,
    T: float,
    num_steps: int = 3,
    fixed_velocity: float = 1.0,
    angle_range: Tuple[float, float] = (0, 2 * np.pi)
) -> List[Tuple[Callable[[float, np.ndarray], np.ndarray], np.ndarray, np.ndarray]]:
    """
    Generate a family of control functions with fixed velocity and step-wise angle changes.

    Args:
        K: Number of control functions to generate.
        T: Total simulation time.
        num_steps: Number of control steps (angle changes).
        fixed_velocity: Constant velocity magnitude.
        angle_range: Range of angles (in radians).

    Returns:
        A list of tuples (control_func, angles, step_times).
    """
    control_data = []
    for _ in range(K):
        angles = np.random.uniform(*angle_range, num_steps)
        step_times = np.linspace(0, T, num_steps + 1)

        def u_func_factory(fixed_velocity: float, angles: np.ndarray, step_times: np.ndarray) -> Callable:
            def control_func(t: float, x: np.ndarray) -> np.ndarray:
                idx = np.searchsorted(step_times, t, side='right') - 1
                idx = max(0, min(len(angles) - 1, idx))
                angle = angles[idx]
                return fixed_velocity * np.array([np.cos(angle), np.sin(angle)])
            return control_func

        control_data.append((u_func_factory(fixed_velocity, angles, step_times), angles, step_times))
    return control_data


def generate_controls(
    K: int,
    T: float,
    num_steps: int = 3,
    fixed_velocity: float = 1.0,
    angle_range: Tuple[float, float] = (0, 2 * np.pi),
    mu_0: np.ndarray = np.zeros(2),
    exploration_fraction: float = 0.1,
    threshold_radius: float = 2.5,
    damping_factor: float = 1.0,
    bounds: Tuple[float, float] = (-10, 10)
) -> List[Tuple[Callable[[float, np.ndarray], np.ndarray], np.ndarray, np.ndarray]]:
    """
    Generate control functions that remain within specified bounds.

    Candidate angles are generated only for the exploration phase [0, exploration_fraction * T].
    For times beyond exploration, the control directs toward the target (mu_0).

    The first candidate angle is sampled uniformly from [-pi, pi] while subsequent angles are sampled
    as increments from the provided angle_range, then converted to effective angles via cumulative sum.

    Args:
        K: Number of control functions to generate.
        T: Total simulation time.
        num_steps: Number of control steps (candidate angles).
        fixed_velocity: Magnitude for control direction.
        angle_range: Range of candidate angle increments (radians) for steps i>=1.
        mu_0: Target position (2D).
        exploration_fraction: Fraction of time for exploration.
        threshold_radius: (Not used in this version.)
        damping_factor: Damping factor applied to adjust velocity.
        bounds: Bounds (min, max) for each coordinate.

    Returns:
        A list of tuples (control_func, effective_angles, step_times).
    """
    control_data = []
    exploration_T = exploration_fraction * T

    for _ in range(K):
        first_angle = np.random.uniform(-np.pi, np.pi)
        if num_steps > 1:
            increments = np.random.uniform(angle_range[0], angle_range[1], num_steps - 1)
            effective_angles = np.empty(num_steps)
            effective_angles[0] = first_angle
            effective_angles[1:] = first_angle + np.cumsum(increments)
        else:
            effective_angles = np.array([first_angle])

        step_times = np.linspace(0, exploration_T, num_steps + 1)

        def u_func_factory(
            fixed_velocity: float,
            effective_angles: np.ndarray,
            step_times: np.ndarray,
            mu_0: np.ndarray,
            exploration_T: float,
            T: float
        ) -> Callable:
            def control_func(t: float, x: np.ndarray) -> np.ndarray:
                if t <= exploration_T:
                    idx = np.searchsorted(step_times, t, side='right') - 1
                    idx = max(0, min(len(effective_angles) - 1, idx))
                    exploration_direction = np.array([np.cos(effective_angles[idx]), np.sin(effective_angles[idx])])
                else:
                    exploration_direction = np.zeros(2)
                position = np.array(x[:2], dtype=float)
                velocity = np.array(x[2:], dtype=float)
                direction_to_target = np.array(mu_0, dtype=float) - position
                dist = np.linalg.norm(direction_to_target)
                direction_to_target = direction_to_target / dist if dist > 1e-6 else np.array([1.0, 0.0])
                exploration_weight = 1.0 if t <= exploration_T else 0.0
                return_weight = 0.0 if t <= exploration_T else 1.0
                control_direction = exploration_weight * exploration_direction + return_weight * direction_to_target
                projected = position + control_direction * fixed_velocity
                for i in range(len(projected)):
                    if projected[i] < bounds[0]:
                        control_direction[i] = max(0, control_direction[i])
                    elif projected[i] > bounds[1]:
                        control_direction[i] = min(0, control_direction[i])
                norm_val = np.linalg.norm(control_direction)
                control_direction = control_direction / norm_val if norm_val > 1e-6 else np.array([1.0, 0.0])
                control_velocity = fixed_velocity * control_direction
                damping_term = damping_factor * (control_velocity - velocity)
                return damping_term
            return control_func

        control_data.append((u_func_factory(fixed_velocity, effective_angles, step_times, mu_0, exploration_T, T),
                             effective_angles, step_times))
    return control_data


def generate_one_control(
    angles: np.ndarray,
    T: float,
    num_steps: int = 3,
    fixed_velocity: float = 1.0,
    mu_0: np.ndarray = np.zeros(2),
    exploration_fraction: float = 0.1,
    damping_factor: float = 1.0,
    bounds: Tuple[float, float] = (-10, 10)
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Generate a control function for a 2D system from candidate control parameters.

    For t <= exploration_fraction * T, the candidate angle increments are converted into cumulative angles,
    then to a 2D unit vector. For t > exploration_fraction * T, the function directs toward the target (mu_0).

    Args:
        angles: 1D array (length=num_steps) of candidate angle increments (radians).
        T: Total simulation time.
        num_steps: Number of control steps.
        fixed_velocity: Scalar for control magnitude.
        mu_0: Target (mean) position (2D).
        exploration_fraction: Fraction of time dedicated to exploration.
        damping_factor: Damping factor.
        bounds: Bounds for each coordinate (min, max).

    Returns:
        A control function u(t, x) returning a 2D control vector.
    """
    # Convert candidate increments to cumulative angles.
    angles = np.cumsum(angles)
    exploration_T = exploration_fraction * T
    step_times = np.linspace(0, exploration_T, num_steps + 1)

    def control_func(t: float, x: np.ndarray) -> np.ndarray:
        if t <= exploration_T:
            idx = np.searchsorted(step_times, t, side='right') - 1
            idx = max(0, min(num_steps - 1, idx))
            exploration_direction = np.array([np.cos(angles[idx]), np.sin(angles[idx])])
        else:
            exploration_direction = np.zeros(2)
        position = np.array(x[:2], dtype=float)
        velocity = np.array(x[2:], dtype=float)
        direction_to_target = np.array(mu_0, dtype=float) - position
        norm_distance = np.linalg.norm(direction_to_target)
        direction_to_target = direction_to_target / norm_distance if norm_distance > 1e-6 else np.array([1.0, 0.0])
        exploration_weight = 1.0 if t <= exploration_T else 0.0
        return_weight = 0.0 if t <= exploration_T else 1.0
        control_direction = exploration_weight * exploration_direction + return_weight * direction_to_target
        projected_position = position + control_direction * fixed_velocity
        for i in range(len(projected_position)):
            if projected_position[i] < bounds[0]:
                control_direction[i] = max(0, control_direction[i])
            elif projected_position[i] > bounds[1]:
                control_direction[i] = min(0, control_direction[i])
        norm_dir = np.linalg.norm(control_direction)
        control_direction = control_direction / norm_dir if norm_dir > 1e-6 else np.array([1.0, 0.0])
        control_velocity = fixed_velocity * control_direction
        damping_term = damping_factor * (control_velocity - velocity)
        return damping_term

    return control_func


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_paths_1d(
    T: np.ndarray,
    paths: np.ndarray,
    save_path: str,
    xlabel: str = "t",
    ylabel: str = "X(t)",
    title: str = "Sample Paths"
) -> None:
    """
    Plot 1D sample paths over a time interval.

    Args:
        T: 1D array of time points.
        paths: Array of shape (n_paths, n_steps, 1) with simulated paths.
        save_path: File path for saving the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in paths:
        ax.plot(T, path[:, 0], color="black", alpha=0.5, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=20)
    ax.set_title(title, fontsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_control(
    T: np.ndarray,
    control_values: np.ndarray,
    save_path: str,
    xlabel: str = "Time",
    ylabel: str = "Control Value",
    title: str = "Control Values Over Time"
) -> None:
    """
    Plot control values over time.

    Args:
        T: Array of time points.
        control_values: Control values corresponding to T.
        save_path: File path for saving the plot.
        xlabel: x-axis label.
        ylabel: y-axis label.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T, control_values, color="blue", linewidth=2, label="Control")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.5)
    ax.legend(fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_2d_trajectories(
    paths: np.ndarray,
    save_path: str,
    title: str = "2D Trajectories",
    x_lim: Tuple[float, float] = (-12, 12),
    y_lim: Tuple[float, float] = (-12, 12)
) -> None:
    """
    Plot 2D trajectories.

    Args:
        paths: Array of shape (n_paths, n_steps, 2).
        save_path: File path for saving the plot.
        title: Plot title.
        x_lim: x-axis limits.
        y_lim: y-axis limits.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(paths.shape[0]):
        trajectory = paths[i, :, :2]
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.6, label=f'Path {i + 1}')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"$X_1$", fontsize=14)
    ax.set_ylabel(r"$X_2$", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_control_function(
    control_func: Callable[[float, np.ndarray], np.ndarray],
    T: float,
    n_steps: int,
    save_path: str,
    title: str = "Control Function"
) -> None:
    """
    Plot the components of a control function over time.

    Args:
        control_func: Control function u(t, x).
        T: Total simulation time.
        n_steps: Number of discrete time points.
        save_path: File path for saving the plot.
        title: Plot title.
    """
    time_points = np.linspace(0, T, n_steps)
    control_values = np.array([control_func(t, np.zeros(2)) for t in time_points])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_points, control_values[:, 0], label="Control X", linewidth=2, alpha=0.8)
    ax.plot(time_points, control_values[:, 1], label="Control Y", linewidth=2, alpha=0.8)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Control Value", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_control_angle(
    control_func: Callable[[float, np.ndarray], np.ndarray],
    T: float,
    n_steps: int,
    save_path: str,
    title: str = "Control Angle Over Time"
) -> None:
    """
    Plot the angle (in radians) of a control function over time.

    Args:
        control_func: Control function.
        T: Total simulation time.
        n_steps: Number of time points.
        save_path: File path for saving the plot.
        title: Plot title.
    """
    time_points = np.linspace(0, T, n_steps)
    control_values = np.array([control_func(t, np.zeros(4)) for t in time_points])
    angles = np.arctan2(control_values[:, 1], control_values[:, 0])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time_points, angles, label="Angle (rad)", color="blue", linewidth=2)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Angle (radians)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_trajectory_lengths(paths: np.ndarray) -> np.ndarray:
    """
    Compute the total Euclidean length of each trajectory.

    Args:
        paths: Array of shape (n_paths, n_steps, 2).

    Returns:
        Lengths of trajectories (shape: n_paths,).
    """
    lengths = []
    for path in paths:
        distances = np.sqrt(np.sum(np.diff(path[:, :2], axis=0) ** 2, axis=1))
        lengths.append(np.sum(distances))
    return np.array(lengths)


def plot_paths_with_angle_changes(
    paths: np.ndarray,
    angles: np.ndarray,
    step_times: np.ndarray,
    save_path: str,
    title: str = "Paths with Angle Change Points",
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot 2D paths with arrows at points where control angles change.

    Args:
        paths: Array of shape (n_paths, n_steps, 2).
        angles: Array of angles for each step.
        step_times: Times at which angle changes occur.
        save_path: File path for saving the plot.
        title: Plot title.
        x_lim: x-axis limits.
        y_lim: y-axis limits.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    arrow_length = 0.5

    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color='black', alpha=0.6)
        for j, t in enumerate(step_times[:-1]):
            idx = int((t / step_times[-1]) * (path.shape[0] - 1))
            dx = arrow_length * np.cos(angles[j])
            dy = arrow_length * np.sin(angles[j])
            ax.quiver(
                path[idx, 0], path[idx, 1],
                dx, dy,
                angles='xy', scale_units='xy', scale=1 / arrow_length,
                color='royalblue', alpha=0.8, width=0.005
            )
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"$X_1$", fontsize=14)
    ax.set_ylabel(r"$X_2$", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paths_with_phased_vectors(
    paths: np.ndarray,
    angles: np.ndarray,
    step_times: np.ndarray,
    exploration_fraction: float,
    save_path: str,
    title: str = "Paths with Phased Vectors",
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    mu_0: Optional[np.ndarray] = None,
    threshold_radius: float = 2.5
) -> None:
    """
    Plot 2D paths with arrows showing exploration (blue) and return (green) phases.
    Stops plotting further arrows once the path enters the threshold region.

    Args:
        paths: Array of shape (n_paths, n_steps, 2).
        angles: Array of angles for each step.
        step_times: Times at which angle changes occur.
        exploration_fraction: Fraction of time for exploration.
        save_path: File path for saving the plot.
        title: Plot title.
        x_lim: x-axis limits.
        y_lim: y-axis limits.
        mu_0: Center of the threshold region.
        threshold_radius: Radius of the threshold region.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    arrow_length = 0.5
    total_time = step_times[-1]
    transition_time = exploration_fraction * total_time

    for path in paths:
        entered_threshold = False
        for i, position in enumerate(path[:, :2]):
            current_time = i / len(path) * total_time
            if current_time > transition_time and mu_0 is not None and np.linalg.norm(position - mu_0) < threshold_radius:
                entered_threshold = True
                ax.plot(path[:i, 0], path[:i, 1], color='black', alpha=0.6)
                ax.scatter(position[0], position[1], color='#228B22', s=10)
                break
        if not entered_threshold:
            ax.plot(path[:, 0], path[:, 1], color='black', alpha=0.6)
        for j, t in enumerate(step_times[:-1]):
            idx = int((t / step_times[-1]) * (path.shape[0] - 1))
            dx = arrow_length * np.cos(angles[j])
            dy = arrow_length * np.sin(angles[j])
            color = 'royalblue' if t <= transition_time else '#228B22'
            if mu_0 is None or np.linalg.norm(path[idx, :2] - mu_0) >= threshold_radius or t <= transition_time:
                ax.quiver(
                    path[idx, 0], path[idx, 1],
                    dx, dy,
                    angles='xy', scale_units='xy', scale=1 / arrow_length,
                    color=color, alpha=0.8, width=0.005
                )
            else:
                break

    circle = plt.Circle(mu_0, threshold_radius, color='#228B22', alpha=0.2)
    ax.add_artist(circle)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"$X_1$", fontsize=14)
    ax.set_ylabel(r"$X_2$", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_sigma_grid(
    a_func: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the sigma values on a 2D grid.

    Args:
        a_func: Function to compute sigma at a given position.
        x_range: Range of x-coordinates as (x_min, x_max).
        y_range: Range of y-coordinates as (y_min, y_max).
        resolution: Number of points in each dimension for the grid.

    Returns:
        A tuple (X, Y, sigma_grid) where X and Y are grid points and sigma_grid contains the computed sigma values.
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    sigma_grid = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            sigma_grid[i, j] = np.max(a_func(point))
    return X, Y, sigma_grid


def plot_paths_with_turbulence(
    paths: np.ndarray,
    angles: np.ndarray,
    step_times: np.ndarray,
    exploration_fraction: float,
    a_func: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int,
    save_path: str,
    title: str = "Paths with Turbulence Heatmap",
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    mu_0: Optional[np.ndarray] = None
) -> None:
    """
    Plot 2D paths overlaid on a turbulence heatmap.

    Args:
        paths: Array of shape (n_paths, n_steps, 2).
        angles: Array of angles corresponding to control steps.
        step_times: Times at which control changes occur.
        exploration_fraction: Fraction of time for exploration.
        a_func: Function to compute sigma at a point.
        x_range: Range of x values.
        y_range: Range of y values.
        resolution: Grid resolution.
        save_path: File path for saving the plot.
        title: Plot title.
        x_lim: Axis limits for x.
        y_lim: Axis limits for y.
        mu_0: Center of threshold region.
    """
    X, Y, sigma_values = compute_sigma_grid(a_func, x_range, y_range, resolution)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_turbulence", ["white", "Goldenrod"])
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = ax.contourf(X, Y, sigma_values, levels=20, cmap=custom_cmap, alpha=0.8)
    plt.colorbar(heatmap, ax=ax, label=r"$\sigma(x)$")
    arrow_length = 0.5
    total_time = step_times[-1]
    transition_time = exploration_fraction * total_time
    for path in paths:
        entered_threshold = False
        for i, position in enumerate(path[:, :2]):
            current_time = i / len(path) * total_time
            if current_time > transition_time and mu_0 is not None and np.linalg.norm(position - mu_0) < 2.5:
                entered_threshold = True
                ax.plot(path[:i, 0], path[:i, 1], color='black', alpha=0.6, zorder=1)
                ax.scatter(position[0], position[1], color='#228B22', s=10, zorder=3)
                break
        if not entered_threshold:
            ax.plot(path[:, 0], path[:, 1], color='black', alpha=0.6, zorder=1)
        for j, t in enumerate(step_times[:-1]):
            idx = int((t / step_times[-1]) * (path.shape[0] - 1))
            dx = arrow_length * np.cos(angles[j])
            dy = arrow_length * np.sin(angles[j])
            color = 'royalblue' if t <= transition_time else '#228B22'
            ax.quiver(
                path[idx, 0], path[idx, 1],
                dx, dy,
                angles='xy', scale_units='xy', scale=1 / arrow_length,
                color=color, alpha=0.8, width=0.005, zorder=2
            )
    if mu_0 is not None:
        circle = plt.Circle(mu_0, 2.5, color='#228B22', alpha=0.2, label="Threshold Region")
        ax.add_artist(circle)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r"$X_1$", fontsize=14)
    ax.set_ylabel(r"$X_2$", fontsize=14)
    ax.grid(True, linestyle="--", alpha=1.0, linewidth=1.5)
    ax.legend(fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paths_with_turbulence_list(
    paths: np.ndarray,
    angle_list: List[np.ndarray],
    step_time_list: List[np.ndarray],
    exploration_fraction: float,
    a_func: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int,
    save_path: str,
    title: str = "Paths with Phased Vectors and Turbulence",
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    mu_0: Optional[np.ndarray] = None,
    threshold_radius: float = 2.5,
    bounds: Tuple[float, float] = (-10, 10)
) -> None:
    """
    Plot 2D paths with arrows (exploration vs. return) overlaid on a turbulence heatmap,
    with shaded areas outside specified bounds.

    Args:
        paths: Array of shape (n_paths, n_steps, 2).
        angle_list: List of angle arrays for each path family.
        step_time_list: List of step time arrays corresponding to each family.
        exploration_fraction: Fraction of time for exploration.
        a_func: Function to compute sigma at a point.
        x_range: Range of x values.
        y_range: Range of y values.
        resolution: Grid resolution.
        save_path: File path for saving the plot.
        title: Plot title.
        x_lim: Axis limits for x.
        y_lim: Axis limits for y.
        mu_0: Center of threshold region.
        threshold_radius: Radius of the threshold region.
        bounds: Boundary limits (min, max).

    """
    X, Y, sigma_values = compute_sigma_grid(a_func, x_range, y_range, resolution)
    noise_color = sns.color_palette("deep")[8]
    reset_color = sns.color_palette("deep")[2]
    crash_color = sns.color_palette("deep")[3]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_turbulence", ["white", noise_color])
    fig, ax = plt.subplots(figsize=(12.8, 10))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=14)
    amplitude_noise = 5
    levels = np.linspace(0, amplitude_noise, 19)
    heatmap = ax.contourf(X, Y, sigma_values, levels=levels, cmap=custom_cmap, alpha=1.0, zorder=-1)
    colorbar = plt.colorbar(heatmap, ax=ax)
    colorbar.set_label(r"Diffusion $\sigma(x)$", size=16)
    colorbar.ax.tick_params(labelsize=14, width=1.5)
    arrow_length = 0.5
    linewidth = 0.1

    add_shaded_area(ax, x_lim, y_lim, bounds)
    start_idx = 0
    for angles, step_times in zip(angle_list, step_time_list):
        n_family_paths = len(angles)
        family_paths = paths[start_idx:start_idx + n_family_paths]
        total_time = step_times[-1]
        transition_time = exploration_fraction * total_time
        for path in family_paths:
            crash_idx = None
            threshold_idx = None
            for i, position in enumerate(path[:, :2]):
                current_time = i / (len(path) - 1) * total_time
                if not (bounds[0] <= position[0] <= bounds[1] and bounds[0] <= position[1] <= bounds[1]):
                    crash_idx = i
                    break
                if current_time > transition_time and mu_0 is not None and np.linalg.norm(position - mu_0) < threshold_radius:
                    threshold_idx = i
                    break
            if crash_idx is not None:
                ax.plot(path[:crash_idx + 1, 0], path[:crash_idx + 1, 1],
                        color='black', alpha=0.6, zorder=1, linewidth=linewidth)
                ax.scatter(path[crash_idx, 0], path[crash_idx, 1],
                           color=crash_color, s=20, zorder=3)
            elif threshold_idx is not None:
                ax.plot(path[:threshold_idx, 0], path[:threshold_idx, 1],
                        color='black', alpha=0.6, zorder=1, linewidth=linewidth)
                ax.scatter(path[threshold_idx, 0], path[threshold_idx, 1],
                           color=reset_color, s=20, zorder=3)
            else:
                ax.plot(path[:, 0], path[:, 1],
                        color='black', alpha=0.6, zorder=1, linewidth=linewidth)
        start_idx += n_family_paths
    if mu_0 is not None:
        circle = plt.Circle(mu_0, threshold_radius, color=reset_color, alpha=0.5, label="Threshold Region")
        ax.add_artist(circle)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r"$X_1$", fontsize=14)
    ax.set_ylabel(r"$X_2$", fontsize=14)
    ax.grid(True, linestyle="--", alpha=1.0, linewidth=1.5)
    legend_elements = [
        Patch(facecolor=noise_color, label='High turbulence region'),
        Patch(facecolor='white', hatch='xx', edgecolor='black', label='Unsafe region'),
        Patch(facecolor=reset_color, alpha=0.5, label='Reset region')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, bbox_to_anchor=(0.1, 0.9))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_shaded_area(ax: plt.Axes, x_lim: Optional[Tuple[float, float]], y_lim: Optional[Tuple[float, float]], bounds: Tuple[float, float]) -> None:
    """
    Add shaded areas outside the specified bounds to the axes.

    Args:
        ax: Axes to add shading.
        x_lim: Overall x-axis limits.
        y_lim: Overall y-axis limits.
        bounds: The unshaded region's bounds (min, max).
    """
    plt.rcParams['hatch.linewidth'] = 1.5
    facecolor = 'white'
    edgecolor = 'black'
    hatch = 'x'

    bound_min, bound_max = bounds
    left_polygon = patches.Polygon(
        [(x_lim[0], y_lim[0]), (bound_min, y_lim[0]), (bound_min, y_lim[1]), (x_lim[0], y_lim[1])],
        closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=1., zorder=0, linewidth=0, hatch=hatch
    )
    right_polygon = patches.Polygon(
        [(bound_max, y_lim[0]), (x_lim[1], y_lim[0]), (x_lim[1], y_lim[1]), (bound_max, y_lim[1])],
        closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=1., zorder=0, linewidth=0, hatch=hatch
    )
    top_polygon = patches.Polygon(
        [(bound_min, bound_max), (bound_max, bound_max), (bound_max, y_lim[1]), (bound_min, y_lim[1])],
        closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=1., zorder=0, linewidth=0, hatch=hatch
    )
    bottom_polygon = patches.Polygon(
        [(bound_min, y_lim[0]), (bound_max, y_lim[0]), (bound_max, bound_min), (bound_min, bound_min)],
        closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=1., zorder=0, linewidth=0, hatch=hatch
    )
    ax.add_patch(left_polygon)
    ax.add_patch(right_polygon)
    ax.add_patch(top_polygon)
    ax.add_patch(bottom_polygon)


def a_func(x: np.ndarray) -> np.ndarray:
    """
    Compute a smooth noise amplitude at position x, centered at (5, 5).

    Args:
        x: 2D point.

    Returns:
        Noise amplitude (same shape as x).
    """
    amplitude = 5.0
    center = np.array([5, 5])
    width = 2.0
    distance = np.linalg.norm(x - center)
    noise_amplitude = amplitude * np.exp(-distance ** 2 / (2 * width ** 2))
    return noise_amplitude * np.ones_like(x)
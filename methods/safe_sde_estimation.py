"""
SafeSDE: A Kernel-Based Model for Safe Estimation of Controlled Stochastic Systems
----------------------------------------------------------------------------------

This module provides an implementation of the SafeSDE class, a kernel-based model designed
to safely estimate and predict the behavior of controlled stochastic systems. The class
stores data in the format [theta_1, …, theta_m, t, safety, reset] and leverages a kernel
function (default is the Gaussian/RBF kernel) computed over control parameters and time to
perform predictions.

Key functionalities include:
    - Adding new observations and updating the Gram matrix.
    - Predicting safety and reset probabilities with uncertainty estimation.
    - Computing lower confidence bounds (LCB) to assess safety.
    - Evaluating feasibility of candidate inputs based on LCB thresholds.
    - Sampling the next candidate from feasible and initial safe sets.
    - Checking stop conditions based on uncertainty thresholds.
    - Estimating system density using both Bessel-based and Gaussian kernel methods.
    - Calculating safety and reset probabilities from simulation trajectories.

Usage Example:
    from safe_sde import SafeSDE
    model = SafeSDE(kernel_func=0.5, control_dim=2, state_dim=3)
    model.add_data(theta=[...], t=..., safety_val=..., reset_val=..., density_states=...)
    safety, variance = model.predict(theta=[...], t=..., target='safety')

Dependencies:
    - numpy
    - scipy (for Bessel functions and gamma computations)
    - scikit-learn (for the RBF kernel)
    - logging
    - time

Author: Luc Brogat-Motte
Date: 2025
"""

import time
import logging
from functools import partial
from typing import Optional, Union, Tuple, Any
import numpy as np
from scipy.special import jv, gamma
from sklearn.metrics.pairwise import rbf_kernel

logger = logging.getLogger(__name__)


class SafeSDE:
    """
    A kernel-based model for safe estimation of controlled stochastic systems.

    Data rows are stored as:
        [theta_1, …, theta_m, t, safety, reset]
    The kernel is computed over the control parameters and time.
    """

    def __init__(
        self,
        kernel_func: Union[float, Any] = 1.0,
        regularization: float = 1e-5,
        beta_s: float = 1.0,
        beta_r: float = 1.0,
        gamma0: Optional[np.ndarray] = None,
        control_dim: int = 1,
        state_dim: int = 2,
    ) -> None:
        """
        Initialize the SafeSDE model.

        Args:
            kernel_func: Either a callable kernel function or a numeric value used as the gamma
                         parameter for the default Gaussian kernel.
            regularization: Regularization parameter for Gram matrix inversion (K + NλI).
            beta_s: Confidence parameter for safety.
            beta_r: Confidence parameter for reset.
            gamma0: Initial safe candidates (shape: (n_candidates, control_dim+2)),
                    each row as [theta_1, …, theta_m, t, T].
            control_dim: Dimension of the control vector.
            state_dim: Dimension of the state.
        """
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.input_dim = control_dim + 1  # control vector + time

        # Regression hyperparameters: if kernel_func is numeric, use it as gamma for the default Gaussian kernel.
        if isinstance(kernel_func, (int, float)):
            self.kernel_func = partial(SafeSDE.gaussian_kernel, gamma=kernel_func)
        else:
            self.kernel_func = kernel_func

        self.regularization = regularization
        self.beta_s = beta_s
        self.beta_r = beta_r

        # Initial safe candidates (gamma0): shape (n_candidates, control_dim+2)
        self.gamma0 = np.asarray(gamma0) if gamma0 is not None else np.empty((0, self.control_dim + 2))

        # Data: each row is [theta_1, …, theta_m, t, safety, reset]
        self.data = np.empty((0, self.input_dim + 2))
        self.K = None       # Gram matrix
        self.K_inv = None   # Inverse of (K + NλI)
        self.density_states = []  # List of arrays for density estimation, each of shape (Q, state_dim)

    def add_data(
        self,
        theta: np.ndarray,
        t: float,
        safety_val: float,
        reset_val: float,
        density_states: Optional[np.ndarray] = None,
    ) -> None:
        """
        Append a new observation and update the Gram matrix.

        Args:
            theta: Control vector (length = control_dim).
            t: Evaluation time.
            safety_val: Observed safety probability.
            reset_val: Observed reset probability.
            density_states: Optional sample states at time t (shape: (Q, state_dim)).
        """
        theta = np.atleast_1d(theta)
        new_row = np.hstack((theta, [t, safety_val, reset_val]))
        self.data = np.vstack((self.data, new_row))
        self.density_states.append(density_states)
        self._update_gram_matrix()

    def _update_gram_matrix(self) -> None:
        """
        Update the Gram matrix and its inverse over the stored inputs (control parameters and time).
        The matrix is computed as: K + (Nλ)I, where N is the number of data points.
        """
        if self.data.shape[0] == 0:
            self.K, self.K_inv = None, None
            return
        X = self.data[:, :self.input_dim]
        self.K = self.kernel_func(X, X)
        self.K += X.shape[0] * self.regularization * np.eye(X.shape[0])
        self.K_inv = np.linalg.inv(self.K)

    def _kernel_vector(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute the kernel vector between given points and the stored data.

        Args:
            pts: Array of shape (n_points, self.input_dim).

        Returns:
            Kernel matrix of shape (n_points, N), where N is the number of stored data points.
        """
        pts = np.atleast_2d(pts)
        if self.data.shape[0] == 0:
            return np.empty((pts.shape[0], 0))
        X = self.data[:, :self.input_dim]
        return self.kernel_func(pts, X)

    def predict(
        self,
        theta: Union[np.ndarray, list],
        t: Union[float, np.ndarray],
        target: str = 'safety',
        variance_only: bool = False,
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], float, np.ndarray]:
        """
        Predict the mean and variance for a given target ('safety' or 'reset') at specified points.

        Args:
            theta: Control vector(s) (shape: (n_points, control_dim)).
            t: Evaluation time(s).
            target: 'safety' (default) or 'reset'.
            variance_only: If True, return only the variance.

        Returns:
            A tuple (mean, variance) for multiple points or scalars for a single point.
            If variance_only is True, only the variance is returned.
        """
        theta = np.atleast_2d(theta)
        t_arr = np.full((theta.shape[0],), t) if np.isscalar(t) else np.asarray(t)
        pts = np.column_stack((theta, t_arr))

        if self.data.shape[0] == 0:
            n = pts.shape[0]
            default_val = 1e5
            if variance_only:
                result = np.ones(n) * default_val
            else:
                result = (np.zeros(n), np.ones(n) * default_val)
            if n == 1:
                return result[0].item() if variance_only else (result[0].item(), result[1].item())
            return result

        col = self.input_dim if target == 'safety' else self.input_dim + 1
        k_vecs = self._kernel_vector(pts)
        means = k_vecs.dot(self.K_inv).dot(self.data[:, col])
        k_self = np.diag(self.kernel_func(pts, pts))
        variances = k_self - np.sum(k_vecs.dot(self.K_inv) * k_vecs, axis=1)
        variances = np.maximum(variances, 1e-6)

        if variance_only:
            return means.item() if pts.shape[0] == 1 else variances
        else:
            return (means.item(), variances.item()) if pts.shape[0] == 1 else (means, variances)

    def compute_lcb(
        self,
        theta: np.ndarray,
        t: Union[float, np.ndarray],
        target: str = 'safety',
        verbose: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Compute the Lower Confidence Bound (LCB) at the given point(s).

        LCB is defined as: mean - beta * sqrt(variance).

        Args:
            theta: Control vector.
            t: Evaluation time(s).
            target: 'safety' (default) or 'reset'.
            verbose: If True, print diagnostic information.

        Returns:
            The LCB value(s).
        """
        t_arr = np.atleast_1d(t)
        pts = np.column_stack((np.tile(theta, (len(t_arr), 1)), t_arr))
        means, variances = self.predict(pts[:, :self.control_dim],
                                        pts[:, self.control_dim],
                                        target=target)
        beta = self.beta_s if target == 'safety' else self.beta_r
        lcbs = means - beta * np.sqrt(variances)
        if verbose:
            print(f"means: {means}, variances: {variances}, lcbs: {lcbs}, beta: {beta}")
        return lcbs.item() if lcbs.ndim == 0 or lcbs.size == 1 else lcbs

    def is_feasible(
        self,
        theta: np.ndarray,
        t: float,
        T: float,
        epsilon: float = 0.01,
        xi: float = 0.01,
        time_grid: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> bool:
        """
        Determine if a candidate [theta, t, T] is feasible.

        Feasibility conditions:
            1. t <= T.
            2. The safety LCB for all times in [0, T] is >= 1 - epsilon.
            3. The reset LCB at time T is >= 1 - xi.

        Args:
            theta: Control vector.
            t: Evaluation time.
            T: Final time.
            epsilon: Safety threshold.
            xi: Reset threshold.
            time_grid: Optional array of times over [0, T].
            verbose: If True, print diagnostic information.

        Returns:
            True if the candidate is feasible, False otherwise.
        """
        if t > T:
            return False
        if time_grid is None:
            time_grid = np.linspace(0, T, num=100)
        safety_lcbs = self.compute_lcb(theta, time_grid, target='safety')
        if np.min(safety_lcbs) < 1 - epsilon:
            return False
        reset_lcb = self.compute_lcb(theta, np.array([T]), target='reset')
        if reset_lcb < 1 - xi:
            return False
        if verbose:
            print(f"Feasibility check for theta={theta}, t={t}, T={T}: "
                  f"min safety LCB = {np.min(safety_lcbs)}, reset LCB = {reset_lcb}")
        return True

    def sample_next(
        self,
        candidate_set: np.ndarray,
        epsilon: float = 0.01,
        xi: float = 0.01,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select the next candidate from the union of feasible candidates in candidate_set
        and the initial safe set (gamma0).

        Args:
            candidate_set: Array of candidate rows ([theta_1, ..., theta_m, t, T]).
            epsilon: Safety threshold.
            xi: Reset threshold.

        Returns:
            A tuple (best_candidate, predicted_safety, uncertainty).
        """
        if candidate_set.size == 0:
            feasibles = np.empty((0, self.control_dim + 2))
        else:
            feasible_mask = np.array([
                self.is_feasible(candidate[:self.control_dim],
                                 candidate[self.control_dim],
                                 candidate[self.control_dim + 1],
                                 epsilon, xi)
                for candidate in candidate_set
            ])
            feasibles = candidate_set[feasible_mask]
        union_set = self.gamma0 if feasibles.size == 0 else np.vstack((feasibles, self.gamma0))
        if self.data.shape[0] == 0:
            variances = np.ones(union_set.shape[0]) * 1e5
            safeties = np.zeros(union_set.shape[0])
        else:
            theta_candidates = union_set[:, :self.control_dim]
            t_candidates = union_set[:, self.control_dim]
            means, variances = self.predict(theta_candidates, t_candidates, target='safety')
            safeties = means
        i_max = np.argmax(variances)
        best_candidate = union_set[i_max]
        best_candidate_safety = safeties[i_max]
        max_uncertainty = variances[i_max]
        logger.info(f"{feasibles.shape[0]} out of {candidate_set.shape[0]} candidates are feasible, "
                    f"with an additional {self.gamma0.shape[0]} initial safe candidates.")
        logger.info("Candidate selected from the initial safe set (gamma0)."
                    if i_max >= feasibles.shape[0]
                    else "Candidate selected from the feasible candidate set.")
        return best_candidate, best_candidate_safety, max_uncertainty

    def stop_condition(self, candidate_set: np.ndarray, eta: float) -> bool:
        """
        Check whether the maximum predictive uncertainty in candidate_set is below eta.

        Args:
            candidate_set: Array of candidate rows.
            eta: Uncertainty threshold.

        Returns:
            True if maximum uncertainty < eta, False otherwise.
        """
        if candidate_set.size == 0:
            return True
        theta_candidates = candidate_set[:, :self.control_dim]
        t_candidates = candidate_set[:, self.control_dim]
        _, variances = self.predict(theta_candidates, t_candidates, target='safety')
        return np.max(variances) < eta

    @staticmethod
    def compute_safety_probability(
        paths: np.ndarray,
        safe_bounds: Tuple[float, float] = (-10, 10),
        t_val: float = 0.0,
        time_grid: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the safety probability at a given time based on simulation paths.

        Args:
            paths: Simulated trajectories, shape (n_paths, n_steps, state_dim).
            safe_bounds: Bounds defining the safe region.
            t_val: Evaluation time.
            time_grid: Array of times corresponding to simulation steps.

        Returns:
            Fraction of trajectories within the safe region at time t_val.
        """
        n_paths, n_steps, _ = paths.shape
        if time_grid is None:
            raise ValueError("time_grid must be provided.")
        if t_val not in time_grid:
            raise ValueError(f"t_val={t_val} is not in the predefined time grid!")
        t_index = np.where(time_grid == t_val)[0][0]
        positions_at_t = paths[:, t_index, :2]
        is_safe = ((safe_bounds[0] <= positions_at_t[:, 0]) & (positions_at_t[:, 0] <= safe_bounds[1]) &
                   (safe_bounds[0] <= positions_at_t[:, 1]) & (positions_at_t[:, 1] <= safe_bounds[1]))
        return float(np.mean(is_safe))

    @staticmethod
    def compute_cumulative_safety_probability(
        paths: np.ndarray,
        safe_bounds: Tuple[float, float] = (-10, 10),
        t_val: float = 0.0,
        time_grid: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the minimum safety probability over the time grid up to t_val based on simulation paths.

        For each time in the grid from 0 to t_val, the fraction of trajectories that are within the safe
        region is computed. The output is the minimum safety probability over these time steps.

        Args:
            paths: Simulated trajectories, shape (n_paths, n_steps, state_dim).
            safe_bounds: Bounds defining the safe region.
            t_val: Final evaluation time.
            time_grid: Array of times corresponding to simulation steps.

        Returns:
            The minimum safety probability over all time steps in [0, t_val].
        """
        n_paths, n_steps, _ = paths.shape
        if time_grid is None:
            raise ValueError("time_grid must be provided.")
        try:
            t_index = np.where(time_grid == t_val)[0][0]
        except IndexError:
            raise ValueError(f"t_val={t_val} is not in the predefined time grid!")

        positions_up_to_t = paths[:, :t_index + 1, :2]  # shape: (n_paths, t_index+1, 2)
        safe_boolean = ((positions_up_to_t[..., 0] >= safe_bounds[0]) &
                        (positions_up_to_t[..., 0] <= safe_bounds[1]) &
                        (positions_up_to_t[..., 1] >= safe_bounds[0]) &
                        (positions_up_to_t[..., 1] <= safe_bounds[1]))
        safety_over_time = np.mean(safe_boolean, axis=0)
        return float(np.min(safety_over_time))

    @staticmethod
    def compute_reset_probability(
        paths: np.ndarray,
        reset_radius: float = 2.5,
        t_val: float = 0.0,
        time_grid: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the reset probability at a given time based on simulation paths.

        Args:
            paths: Simulated trajectories, shape (n_paths, n_steps, state_dim).
            reset_radius: Radius defining the reset region.
            t_val: Evaluation time.
            time_grid: Array of times corresponding to simulation steps.

        Returns:
            Fraction of trajectories within the reset region at time t_val.
        """
        n_paths, n_steps, _ = paths.shape
        if time_grid is None:
            raise ValueError("time_grid must be provided.")
        if t_val not in time_grid:
            raise ValueError(f"t_val={t_val} is not in the predefined time grid!")
        t_index = np.where(time_grid == t_val)[0][0]
        positions_at_t = paths[:, t_index, :2]
        norms = np.linalg.norm(positions_at_t, axis=1)
        is_reset = norms <= reset_radius
        return float(np.mean(is_reset))

    @staticmethod
    def _rho(x: np.ndarray, R: float) -> np.ndarray:
        """
        Evaluate the kernel density function:
            ρ_R(x) = R^(n/2) ||x||^(-n/2) J_{n/2}(2πR||x||)

        Args:
            x: Input array with shape (..., state_dim).
            R: Kernel parameter.

        Returns:
            Kernel density evaluations.
        """
        n = x.shape[-1]
        r = np.linalg.norm(x, axis=-1)
        eps = 1e-8
        constant = (np.pi * R ** 2) ** (n / 2) / gamma(n / 2 + 1)
        return np.where(
            r < eps,
            constant,
            (R ** (n / 2)) * (r ** (-n / 2)) * jv(n / 2, 2 * np.pi * R * r)
        )

    def predict_system_density_bessel(
        self,
        theta: np.ndarray,
        t: float,
        xs: np.ndarray,
        R: float
    ) -> np.ndarray:
        """
        Predict the system density at evaluation points xs for a new input (theta, t)
        using a Bessel-based kernel density.

        The density is computed as:
            p̂_θ(t, x) = û(x) · (K + NλI)^{-1} · k(theta, t)
        where û(x) is evaluated using the stored density states.

        Args:
            theta: New control vector (1D array of length control_dim).
            t: New evaluation time.
            xs: Points at which to estimate density (shape: (M, state_dim)).
            R: Kernel parameter used in the density function ρ_R.

        Returns:
            Density estimates at the evaluation points (shape: (M,)).
        """
        start_total = time.time()
        logger.info("Starting density prediction (Bessel kernel).")

        N = self.data.shape[0]
        if N == 0:
            logger.warning("No data available for density prediction. Returning zeros.")
            return np.zeros(xs.shape[0])

        t0 = time.time()
        try:
            density_states_all = np.stack(self.density_states, axis=0)  # shape: (N, Q, state_dim)
        except Exception as e:
            raise ValueError("All density_states must be provided and have the same shape.") from e
        logger.debug(f"Stacking density_states took {time.time() - t0:.3f} seconds. Shape: {density_states_all.shape}")

        t0 = time.time()
        diff = xs[None, :, None, :] - density_states_all[:, None, :, :]  # shape: (N, M, Q, state_dim)
        logger.debug(f"Computing differences took {time.time() - t0:.3f} seconds. Shape: {diff.shape}")

        t0 = time.time()
        kernel_vals = self._rho(diff, R)  # shape: (N, M, Q)
        logger.debug(f"Computing kernel values took {time.time() - t0:.3f} seconds. Shape: {kernel_vals.shape}")

        t0 = time.time()
        u_hat = np.mean(kernel_vals, axis=2)  # shape: (N, M)
        logger.debug(f"Computing u_hat took {time.time() - t0:.3f} seconds. Shape: {u_hat.shape}")

        t0 = time.time()
        new_input = np.hstack((theta, t)).reshape(1, -1)
        logger.debug(f"Constructed new_input. Shape: {new_input.shape}")
        k_new = self._kernel_vector(new_input).flatten()  # shape: (N,)
        logger.debug(f"Computing kernel vector took {time.time() - t0:.3f} seconds. Shape: {k_new.shape}")

        t0 = time.time()
        w = self.K_inv.dot(k_new)  # shape: (N,)
        logger.debug(f"Computing weights (w) took {time.time() - t0:.3f} seconds. Shape: {w.shape}")

        t0 = time.time()
        density_pred = u_hat.T.dot(w)  # shape: (M,)
        logger.debug(f"Computing density prediction took {time.time() - t0:.3f} seconds. Shape: {density_pred.shape}")

        total_time = time.time() - start_total
        logger.info(f"Total prediction time: {total_time:.3f} seconds.")

        return density_pred

    def predict_system_density(
        self,
        theta: np.ndarray,
        t: float,
        xs: np.ndarray,
        R: float,
        chunk_size: int = 500
    ) -> np.ndarray:
        """
        Predict the system density at evaluation points xs for a new input (theta, t)
        using a Gaussian kernel (with squared distance formulation) in a vectorized and chunked manner.

        The density is computed as:
            p̂_θ(t, x) = û(x) · (K + NλI)^{-1} · k(theta, t)
        where û(x) is evaluated using the stored density states.

        Args:
            theta: New control vector (1D array of length control_dim).
            t: New evaluation time.
            xs: Points at which to estimate density (shape: (M, state_dim)).
            R: Kernel parameter used in the density function.
            chunk_size: Number of evaluation points to process at once.

        Returns:
            Density estimates at the evaluation points (shape: (M,)).
        """
        start_total = time.time()
        logger.info("Starting density prediction (Gaussian kernel).")

        N = self.data.shape[0]
        if N == 0:
            logger.warning("No data available for density prediction. Returning zeros.")
            return np.zeros(xs.shape[0], dtype=np.float32)

        t0 = time.time()
        try:
            density_states_all = np.stack(self.density_states, axis=0).astype(np.float32)
        except Exception as e:
            raise ValueError("All density_states must be provided and have the same shape.") from e
        logger.debug(f"Stacking density_states took {time.time() - t0:.3f} seconds. Shape: {density_states_all.shape}")

        N, Q, d = density_states_all.shape
        M = xs.shape[0]
        u_hat_all = np.empty((N, M), dtype=np.float32)
        const_factor = R ** d * (2 * np.pi) ** (-d / 2)

        t0 = time.time()
        for i in range(0, M, chunk_size):
            xs_chunk = xs[i:i + chunk_size]  # shape: (chunk_size, d)
            diff = xs_chunk[None, :, None, :] - density_states_all[:, None, :, :]  # shape: (N, chunk_size, Q, d)
            squared_distances = np.sum(diff ** 2, axis=-1)  # shape: (N, chunk_size, Q)
            kernel_vals_chunk = const_factor * np.exp(-0.5 * R ** 2 * squared_distances)  # shape: (N, chunk_size, Q)
            u_hat_chunk = np.mean(kernel_vals_chunk, axis=2)  # shape: (N, chunk_size)
            u_hat_all[:, i:i + chunk_size] = u_hat_chunk
        logger.debug(f"Vectorized (chunked) kernel computation took {time.time() - t0:.3f} seconds. Shape: {u_hat_all.shape}")

        t0 = time.time()
        new_input = np.hstack((theta, t)).reshape(1, -1)
        logger.debug(f"Constructed new_input. Shape: {new_input.shape}")
        k_new = self._kernel_vector(new_input).flatten()  # shape: (N,)
        logger.debug(f"Computing kernel vector took {time.time() - t0:.3f} seconds. Shape: {k_new.shape}")

        t0 = time.time()
        w = self.K_inv.dot(k_new)  # shape: (N,)
        logger.debug(f"Computing weights (w) took {time.time() - t0:.3f} seconds. Shape: {w.shape}")

        t0 = time.time()
        density_pred = u_hat_all.T.dot(w)  # shape: (M,)
        logger.debug(f"Computing density prediction took {time.time() - t0:.3f} seconds. Shape: {density_pred.shape}")

        total_time = time.time() - start_total
        logger.info(f"Total prediction time: {total_time:.3f} seconds.")

        return density_pred

    @staticmethod
    def gaussian_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Compute the Gaussian (RBF) kernel between X and Y.

        Args:
            X: First input array.
            Y: Second input array.
            gamma: Kernel coefficient.

        Returns:
            The RBF kernel matrix.
        """
        return rbf_kernel(X, Y, gamma=gamma)
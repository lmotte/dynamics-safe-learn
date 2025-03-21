import time
import unittest
import numpy as np
import logging
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel

# Assume SafeSDE is imported from your module.
from methods.safe_sde_estimation import SafeSDE


class TestSafeSDEEstimation(unittest.TestCase):

    def setUp(self):
        # Initialize the SafeSDE model.
        # Here, if kernel_func is numeric, it is used as the gamma parameter.
        self.model = SafeSDE(kernel_func=1.0, regularization=1e-5,
                             beta_s=1.0, beta_r=1.0,
                             gamma0=np.array([[0.0, 0.0, 0.5]]),
                             control_dim=1, state_dim=2)
        # Add dummy data points with high safety/reset values.
        for t in np.linspace(0, 0.5, 6):
            self.model.add_data(0.0, t, 1.0, 1.0)

    def test_gram_matrix_shape(self):
        """After adding data, the Gram matrix should be square with size equal to the number of data points."""
        N = self.model.data.shape[0]
        self.assertEqual(self.model.K.shape, (N, N))
        self.assertEqual(self.model.K_inv.shape, (N, N))

    def test_predict_safety_single_point(self):
        """Test that predict returns float mean and variance for a single point (safety)."""
        mean, variance = self.model.predict(0.0, 0.0, target='safety')
        self.assertIsInstance(mean, float)
        self.assertIsInstance(variance, float)
        self.assertGreaterEqual(variance, 0.0)

    def test_predict_reset_single_point(self):
        """Test that predict returns float mean and variance for a single point (reset)."""
        mean, variance = self.model.predict(0.0, 0.0, target='reset')
        self.assertIsInstance(mean, float)
        self.assertIsInstance(variance, float)
        self.assertGreaterEqual(variance, 0.0)

    def test_predict_multiple_points(self):
        """
        Test predict method with multiple time points for a single control candidate.
        We pass theta as a 2D array of shape (n, 1) to match the time array.
        """
        t_vals = np.array([0.0, 0.1, 0.2])
        # Create theta as a 2D array with one control candidate per time value.
        theta = np.full((t_vals.shape[0], 1), 0.0)  # shape (3, 1)
        means, variances = self.model.predict(theta, t_vals, target='safety')
        self.assertEqual(means.shape, (3,))
        self.assertEqual(variances.shape, (3,))


    def test_compute_lcb_single_point(self):
        """
        Test that compute_lcb returns mean - beta * sqrt(variance) for a single point.
        """
        mean, variance = self.model.predict(0.0, 0.0, target='safety')
        lcb = self.model.compute_lcb(0.0, 0.0, target='safety')
        expected_lcb = mean - self.model.beta_s * np.sqrt(variance)
        self.assertAlmostEqual(lcb, expected_lcb)

    def test_compute_lcb_multiple_points(self):
        """
        Test compute_lcb for multiple time points.
        Pass theta as a one-dimensional array (the single control candidate)
        and t as an array.
        """
        t_vals = np.array([0.0, 0.1, 0.2])
        theta = np.array([0.0])  # 1D array for a single candidate
        lcb = self.model.compute_lcb(theta, t_vals, target='safety')
        # Compute expected LCB using predict.
        # For predict, we need theta tiled to match t.
        theta_tiled = np.tile(theta, (t_vals.shape[0], 1))
        means, variances = self.model.predict(theta_tiled, t_vals, target='safety')
        expected_lcb = means - self.model.beta_s * np.sqrt(variances)
        np.testing.assert_allclose(lcb, expected_lcb)

    def test_is_feasible(self):
        """
        Test the feasibility check.
        A candidate (theta, t, T) is considered feasible if the safety LCB and the reset LCB
        are above (1-epsilon) and (1-xi) respectively.
        """
        candidate = (0.0, 0.0, 0.5)
        self.assertTrue(self.model.is_feasible(*candidate, epsilon=0.05, xi=0.05))
        self.assertFalse(self.model.is_feasible(*candidate, epsilon=0.0, xi=0.0))

    def test_sample_next(self):
        """
        Test that sample_next returns a candidate from the provided candidate set.
        """
        candidate_set = np.array([
            [0.0, 0.0, 0.5],
            [0.1, 0.1, 0.6],
            [0.2, 0.2, 0.7]
        ])
        candidate, safety_pred, uncertainty_pred = self.model.sample_next(candidate_set, epsilon=0.05, xi=0.05)
        self.assertIsNotNone(candidate, "sample_next returned None despite available candidates")
        candidate_tuple = tuple(candidate)
        candidate_set_tuples = [tuple(row) for row in candidate_set]
        self.assertIn(candidate_tuple, candidate_set_tuples,
                      "Returned candidate is not in the provided candidate set")

    def test_stop_condition(self):
        """
        Test that stop_condition returns a boolean.
        The method should return True if the maximum predictive uncertainty in the candidate set is below eta.
        """
        candidate_set = np.array([
            [0.0, 0.0, 0.5],
            [0.5, 0.3, 0.8],
            [-0.5, 0.2, 0.7]
        ])
        stop = self.model.stop_condition(candidate_set, eta=1.0)
        # Accept both Python bool and numpy.bool_
        self.assertTrue(isinstance(stop, (bool, np.bool_)))

    def test_no_data_prediction(self):
        """
        Test behavior when the model has no data.
        In such cases, predict should return default values.
        """
        empty_model = SafeSDE(kernel_func=1.0, regularization=1e-5,
                              beta_s=1.0, beta_r=1.0, gamma0=None,
                              control_dim=1, state_dim=2)
        result = empty_model.predict(0.0, 0.0, target='safety', variance_only=False)
        if isinstance(result, tuple):
            mean, variance = result
            self.assertEqual(mean, 0)
            self.assertTrue(variance > 1e4)
        else:
            self.fail("Prediction output format not as expected for empty model.")

    def test_predict_system_density_bessel(self):
        """
        Test the Bessel-based density prediction.
        This verifies that the output has the expected shape.
        """
        model = SafeSDE(kernel_func=1.0, regularization=1e-5,
                        beta_s=1.0, beta_r=1.0,
                        gamma0=np.array([[0.0, 0.0, 0.5]]),
                        control_dim=1, state_dim=2)
        # Create dummy density states (Q=5, state_dim=2).
        density_states = np.random.randn(5, 2)
        for t in np.linspace(0, 0.5, 6):
            model.add_data(0.0, t, 1.0, 1.0, density_states=density_states)
        xs = np.random.randn(10, 2)  # evaluation points
        R = 1.0
        density_pred = model.predict_system_density_bessel(np.array([0.0]), 0.0, xs, R)
        self.assertEqual(density_pred.shape, (xs.shape[0],))

    def test_predict_system_density(self):
        """
        Test the Gaussian (chunked) density prediction.
        Verifies that the density prediction output has the expected shape.
        """
        model = SafeSDE(kernel_func=1.0, regularization=1e-5,
                        beta_s=1.0, beta_r=1.0,
                        gamma0=np.array([[0.0, 0.0, 0.5]]),
                        control_dim=1, state_dim=2)
        density_states = np.random.randn(5, 2)
        for t in np.linspace(0, 0.5, 6):
            model.add_data(0.0, t, 1.0, 1.0, density_states=density_states)
        xs = np.random.randn(20, 2)  # more evaluation points
        R = 1.0
        density_pred = model.predict_system_density(np.array([0.0]), 0.0, xs, R, chunk_size=5)
        self.assertEqual(density_pred.shape, (xs.shape[0],))

    def test_compute_reset_probability(self):
        """
        Test that compute_reset_probability returns a valid probability (between 0 and 1)
        given simulated trajectories.
        """
        n_paths = 10
        n_steps = 5
        state_dim = 2
        # Create dummy paths uniformly within [-1, 1].
        paths = np.random.uniform(low=-1.0, high=1.0, size=(n_paths, n_steps, state_dim))
        time_grid = np.linspace(0, 1, n_steps)
        t_val = time_grid[2]
        reset_prob = SafeSDE.compute_reset_probability(paths, reset_radius=1.5, t_val=t_val, time_grid=time_grid)
        self.assertGreaterEqual(reset_prob, 0.0)
        self.assertLessEqual(reset_prob, 1.0)

    def test_compute_cumulative_safety_probability(self):
        """
        Test that compute_cumulative_safety_probability returns a valid probability (between 0 and 1)
        and that it raises an error when t_val is not in the provided time grid or when time_grid is None.
        """
        n_paths = 10
        n_steps = 5
        state_dim = 2
        # Create paths that are safely within [-0.5, 0.5].
        paths = np.random.uniform(low=-0.5, high=0.5, size=(n_paths, n_steps, state_dim))
        time_grid = np.linspace(0, 1, n_steps)
        t_val = time_grid[-1]
        cum_safety_prob = SafeSDE.compute_cumulative_safety_probability(paths, safe_bounds=(-1, 1), t_val=t_val,
                                                                        time_grid=time_grid)
        self.assertGreaterEqual(cum_safety_prob, 0.0)
        self.assertLessEqual(cum_safety_prob, 1.0)

        # Test error when t_val not in time_grid.
        with self.assertRaises(ValueError):
            SafeSDE.compute_cumulative_safety_probability(paths, safe_bounds=(-1, 1), t_val=2.0, time_grid=time_grid)

        # Test error when time_grid is None.
        with self.assertRaises(ValueError):
            SafeSDE.compute_cumulative_safety_probability(paths, safe_bounds=(-1, 1), t_val=t_val, time_grid=None)

    def test_custom_kernel_function(self):
        """
        Test that a custom kernel function (here, a simple linear kernel) works with the model.
        """

        def linear_kernel(X, Y, gamma=1.0):
            return np.dot(X, Y.T)

        model = SafeSDE(kernel_func=linear_kernel, regularization=1e-5,
                        beta_s=1.0, beta_r=1.0,
                        gamma0=np.array([[0.0, 0.0, 0.5]]),
                        control_dim=1, state_dim=2)
        # Add some data.
        for t in np.linspace(0, 0.5, 6):
            model.add_data(0.0, t, 1.0, 1.0)
        mean, variance = model.predict(0.0, 0.0, target='safety')
        self.assertIsInstance(mean, float)
        self.assertIsInstance(variance, float)

    def test_density_states_inconsistency(self):
        """
        Test that if density_states of inconsistent shapes are provided,
        the density prediction method raises a ValueError.
        """
        model = SafeSDE(kernel_func=1.0, regularization=1e-5,
                        beta_s=1.0, beta_r=1.0,
                        gamma0=np.array([[0.0, 0.0, 0.5]]),
                        control_dim=1, state_dim=2)
        # Add data with consistent density_states shape.
        density_states_consistent = np.random.randn(5, 2)
        model.add_data(0.0, 0.0, 1.0, 1.0, density_states=density_states_consistent)
        # Then add data with an inconsistent density_states shape.
        density_states_inconsistent = np.random.randn(6, 2)
        model.add_data(0.0, 0.1, 1.0, 1.0, density_states=density_states_inconsistent)
        xs = np.random.randn(10, 2)
        R = 1.0
        with self.assertRaises(ValueError):
            model.predict_system_density_bessel(np.array([0.0]), 0.0, xs, R)


if __name__ == '__main__':
    unittest.main()

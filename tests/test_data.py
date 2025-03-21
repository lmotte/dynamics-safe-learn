import os
import tempfile
import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils.data import (
    euler_maruyama,
    second_order_coefficients,
    sinusoidal_control,
    generate_control_functions,
    generate_controls,
    generate_one_control,
    plot_paths_1d,
    plot_control,
    plot_2d_trajectories,
    plot_control_function,
    plot_control_angle,
    compute_trajectory_lengths,
    plot_paths_with_angle_changes,
    plot_paths_with_turbulence,
    plot_paths_with_turbulence_list,
    compute_sigma_grid,
    add_shaded_area,
    a_func
)


class TestUtilsData(unittest.TestCase):

    def test_euler_maruyama_without_coef(self):
        """Test euler_maruyama without coefficient saving returns correct shape."""
        b = lambda t, x: 0.5 * x
        sigma = lambda t, x: 0.1 * np.ones_like(x)
        n_steps = 10
        n_paths = 5
        T = 1.0
        n = 2
        mu_0 = np.array([0.0, 0.0])
        sigma_0 = 1.0

        paths = euler_maruyama(b, sigma, n_steps, n_paths, T, n, mu_0, sigma_0)
        self.assertEqual(paths.shape, (n_paths, n_steps, n))

    def test_euler_maruyama_with_coef(self):
        """Test euler_maruyama with coefficient saving returns tuple with correct shapes.
        Use n=1 so that diffusion outputs match the pre-allocated shape (n_steps, 1)."""
        b = lambda t, x: 0.5 * x
        sigma = lambda t, x: 0.1 * np.ones_like(x)
        n_steps = 10
        n_paths = 5
        T = 1.0
        n = 1  # simulate a 1D SDE so that sigma returns shape (1,)
        mu_0 = np.array([0.0])
        sigma_0 = 1.0

        paths, B_save, S_save = euler_maruyama(b, sigma, n_steps, n_paths, T, n, mu_0, sigma_0, coef_save=True)
        self.assertEqual(paths.shape, (n_paths, n_steps, n))
        self.assertEqual(B_save.shape, (n_paths, n_steps, n))
        self.assertEqual(S_save.shape, (n_paths, n_steps, 1))

    def test_second_order_coefficients(self):
        """Test second_order_coefficients returns functions with proper output shapes."""
        dim = 2
        u_func = lambda t, x: np.array([1.0, -1.0])
        a_func_local = lambda x: np.array([0.5, 0.5])
        b_func, sigma_func = second_order_coefficients(u_func, a_func_local, dim)
        x = np.concatenate([np.array([0.0, 0.0]), np.array([0.0, 0.0])])
        drift = b_func(0, x)
        diffusion = sigma_func(0, x)
        self.assertEqual(drift.shape[0], 2 * dim)
        self.assertEqual(diffusion.shape[0], 2 * dim)
        self.assertTrue(np.allclose(diffusion[:dim], 0))

    def test_sinusoidal_control(self):
        """Test that sinusoidal_control returns a control function with correct output shape."""
        w = 2.0
        dim = 3
        control_func = sinusoidal_control(w, dim)
        t = 1.0
        x = np.zeros(dim)
        output = control_func(t, x)
        self.assertEqual(output.shape, (dim,))
        expected = np.array([np.sin(w * t / 10 * np.pi)] * dim)
        np.testing.assert_allclose(output, expected)

    def test_generate_control_functions(self):
        """Test generate_control_functions returns expected tuple structure and output shape."""
        K = 3
        T = 10.0
        num_steps = 3
        fixed_velocity = 1.0
        controls = generate_control_functions(K, T, num_steps=num_steps, fixed_velocity=fixed_velocity)
        self.assertEqual(len(controls), K)
        for control_func, angles, step_times in controls:
            self.assertEqual(angles.shape[0], num_steps)
            self.assertEqual(step_times.shape[0], num_steps + 1)
            output = control_func(0.0, np.zeros(2))
            self.assertEqual(output.shape, (2,))

    def test_generate_controls(self):
        """Test generate_controls returns a list with proper structure and control outputs."""
        K = 3
        T = 10.0
        num_steps = 3
        fixed_velocity = 1.0
        mu_0 = np.array([0, 0])
        exploration_fraction = 0.5
        threshold_radius = 2.5
        damping_factor = 1.0
        bounds = (-10, 10)
        controls = generate_controls(K, T, num_steps=num_steps, fixed_velocity=fixed_velocity,
                                     mu_0=mu_0, exploration_fraction=exploration_fraction,
                                     threshold_radius=threshold_radius, damping_factor=damping_factor,
                                     bounds=bounds)
        self.assertEqual(len(controls), K)
        for control_func, angles, step_times in controls:
            self.assertEqual(angles.shape[0], num_steps)
            self.assertEqual(step_times.shape[0], num_steps + 1)
            output = control_func(0.0, np.array([5, 5, 0, 0]))
            self.assertEqual(output.shape, (2,))

    def test_generate_one_control_output_variation(self):
        """Test that generate_one_control returns different outputs for different time intervals."""
        T = 10.0
        num_steps = 2
        angles_candidate = np.array([1.0, 2.5])
        fixed_velocity = 2.0
        mu_0 = np.array([0, 0])
        exploration_fraction = 0.5
        damping_factor = 1.0
        bounds = (-10, 10)
        control_func = generate_one_control(angles_candidate, T, num_steps=num_steps,
                                            fixed_velocity=fixed_velocity, mu_0=mu_0,
                                            exploration_fraction=exploration_fraction,
                                            damping_factor=damping_factor, bounds=bounds)
        x = np.array([5, 5, 0, 0])
        t1 = 2.0  # likely in exploration phase
        t2 = 7.0  # likely in return phase
        out1 = control_func(t1, x)
        out2 = control_func(t2, x)
        self.assertFalse(np.allclose(out1, out2), "Control outputs should differ between phases")

    def test_plot_control_function_file(self):
        """Test that plot_control_function creates a file.
        Wrap the control function to ensure it receives a 4-element state vector."""
        T = 10.0
        num_steps = 2
        angles_candidate = np.array([1.3948, 5.4710])
        fixed_velocity = 2.0
        mu_0 = np.array([0, 0])
        exploration_fraction = 0.5
        damping_factor = 1.0
        bounds = (-10, 10)
        control_func = generate_one_control(angles_candidate, T, num_steps=num_steps,
                                            fixed_velocity=fixed_velocity, mu_0=mu_0,
                                            exploration_fraction=exploration_fraction,
                                            damping_factor=damping_factor, bounds=bounds)
        # Wrap control_func so that it always uses a state vector of length 4.
        wrapped_control_func = lambda t, x: control_func(t, np.zeros(4))
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.close()
        try:
            plot_control_function(wrapped_control_func, T, 50, temp_file.name, title="Test Control Function")
            self.assertTrue(os.path.exists(temp_file.name), "Plot file was not created.")
        finally:
            os.remove(temp_file.name)

    def test_plot_paths_1d_file(self):
        """Test that plot_paths_1d creates a file given dummy 1D paths."""
        n_paths = 3
        n_steps = 20
        T_vals = np.linspace(0, 1, n_steps)
        paths = np.random.randn(n_paths, n_steps, 1)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.close()
        try:
            plot_paths_1d(T_vals, paths, temp_file.name, xlabel="t", ylabel="X(t)", title="1D Paths")
            self.assertTrue(os.path.exists(temp_file.name), "1D paths plot file was not created.")
        finally:
            os.remove(temp_file.name)

    def test_plot_2d_trajectories_file(self):
        """Test that plot_2d_trajectories creates a file given dummy 2D paths."""
        n_paths = 4
        n_steps = 30
        paths = np.cumsum(np.random.randn(n_paths, n_steps, 2), axis=1)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.close()
        try:
            plot_2d_trajectories(paths, temp_file.name, title="2D Trajectories", x_lim=(-10, 10), y_lim=(-10, 10))
            self.assertTrue(os.path.exists(temp_file.name), "2D trajectories plot file was not created.")
        finally:
            os.remove(temp_file.name)

    def test_compute_trajectory_lengths(self):
        """Test that compute_trajectory_lengths computes the correct Euclidean lengths."""
        path = np.array([[0, 0], [3, 4]])
        paths = path.reshape(1, 2, 2)
        lengths = compute_trajectory_lengths(paths)
        np.testing.assert_allclose(lengths, [5.0], rtol=1e-5)

    def test_compute_sigma_grid(self):
        """Test compute_sigma_grid with a constant a_func."""
        const_a_func = lambda x: np.array([2.0, 2.0])
        x_range = (0, 10)
        y_range = (0, 10)
        resolution = 50
        X, Y, sigma_grid = compute_sigma_grid(const_a_func, x_range, y_range, resolution)
        self.assertEqual(X.shape, (resolution, resolution))
        self.assertEqual(Y.shape, (resolution, resolution))
        self.assertEqual(sigma_grid.shape, (resolution, resolution))
        self.assertTrue(np.allclose(sigma_grid, 2.0))

    def test_add_shaded_area(self):
        """Test add_shaded_area adds four patches to the axes."""
        fig, ax = plt.subplots()
        initial_patches = len(ax.patches)
        x_lim = (-15, 15)
        y_lim = (-15, 15)
        bounds = (-10, 10)
        add_shaded_area(ax, x_lim, y_lim, bounds)
        self.assertEqual(len(ax.patches), initial_patches + 4)
        plt.close(fig)

    def test_a_func(self):
        """Test that a_func returns a noise amplitude with the same shape as input."""
        x = np.array([3.0, 4.0])
        out = a_func(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(np.allclose(out, out[0]))


if __name__ == '__main__':
    unittest.main()
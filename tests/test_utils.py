import os
import tempfile
import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import (
    rho,
    sample_local_candidates,
    sample_random_candidates_from_grid,
    plot_observed_histograms
)

# A dummy safe model to simulate the SafeSDE interface needed for candidate sampling.
class DummySafeModel:
    pass


class TestUtilsDataModule(unittest.TestCase):

    def test_rho(self):
        """
        Test that rho returns an array of shape (nx, ny) with values computed as expected.
        """
        # For a 2D space (n=2)
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([[0, 0], [1, 1], [2, 2]])
        mu = 1.0
        result = rho(X, Y, mu)
        self.assertEqual(result.shape, (2, 3))
        # For (0,0) and (0,0): squared distance = 0
        expected_00 = mu**2 * (2 * np.pi) ** (-1) * np.exp(-0.5 * mu**2 * 0)
        self.assertAlmostEqual(result[0, 0], expected_00, places=5)
        # For (0,0) and (1,1): squared distance = 2
        expected_01 = mu**2 * (2 * np.pi) ** (-1) * np.exp(-0.5 * mu**2 * 2)
        self.assertAlmostEqual(result[0, 1], expected_01, places=5)

    def test_sample_local_candidates(self):
        """
        Test that sample_local_candidates returns a candidate set of expected shape.
        """
        safe_model = DummySafeModel()
        safe_model.control_dim = 1
        # Create dummy training data: each row is [theta, t, T]
        safe_model.data = np.array([[0.1, 0.2, 0.3],
                                    [0.15, 0.25, 0.3],
                                    [0.2, 0.3, 0.3],
                                    [0.12, 0.22, 0.3],
                                    [0.18, 0.28, 0.3]])
        default_candidate_set = np.array([[0.0, 0.0, 1.0]])
        time_grid = np.linspace(0, 1, 100)
        T = 1.0
        num_candidates = 20
        candidate_set = sample_local_candidates(safe_model, default_candidate_set, time_grid, T,
                                                num_candidates=num_candidates,
                                                margin_angle=0.1, margin_t=0.05)
        # Candidate set should have shape (num_candidates, control_dim+2) => (20, 3)
        self.assertEqual(candidate_set.shape, (num_candidates, 3))
        # Last column should be T for every candidate.
        self.assertTrue(np.allclose(candidate_set[:, -1], T))

    def test_sample_random_candidates_from_grid(self):
        """
        Test that sample_random_candidates_from_grid returns candidates with the correct shape.
        """
        # For a control space of dimension 2.
        control_range = [(-1, 1), (-np.pi/4, np.pi/4)]
        time_grid = np.linspace(0, 1, 50)
        T = 1.0
        num_candidates = 25
        candidate_set = sample_random_candidates_from_grid(control_range, time_grid, T,
                                                            num_candidates=num_candidates)
        # Expected shape: (num_candidates, m+2) where m = 2  ⇒ (25, 4)
        self.assertEqual(candidate_set.shape, (num_candidates, 4))
        # Last column should equal T.
        self.assertTrue(np.allclose(candidate_set[:, -1], T))

    def test_plot_observed_histograms_nonempty(self):
        """
        Test that plot_observed_histograms creates a file when exploration history is nonempty.
        """
        # Create dummy exploration history where each record is a tuple:
        # (any, any, observed_safety, observed_reset)
        exploration_history = {
            0: (0, 0, 0.95, 0.90),
            1: (0, 0, 0.92, 0.88),
            2: (0, 0, 0.97, 0.85)
        }
        epsilon = 0.1
        xi = 0.15
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.close()
        try:
            plot_observed_histograms(exploration_history, epsilon, xi, temp_file.name)
            self.assertTrue(os.path.exists(temp_file.name))
        finally:
            os.remove(temp_file.name)

    def test_plot_observed_histograms_empty(self):
        """
        Test that plot_observed_histograms does not create a file when exploration history is empty.
        """
        exploration_history = {}
        epsilon = 0.1
        xi = 0.15
        # Use a temporary filename (do not create the file beforehand)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_name = temp_file.name
        temp_file.close()
        # Remove the file if it exists so that we can check later if the function creates it.
        if os.path.exists(temp_name):
            os.remove(temp_name)
        plot_observed_histograms(exploration_history, epsilon, xi, temp_name)
        # When history is empty, the function should log a warning and not create a plot.
        self.assertFalse(os.path.exists(temp_name))


if __name__ == '__main__':
    unittest.main()
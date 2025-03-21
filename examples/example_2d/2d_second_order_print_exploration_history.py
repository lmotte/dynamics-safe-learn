#!/usr/bin/env python
"""
Script: 2d_second_order_print_exploration_history.py
--------------------------------------
This script loads safe model snapshots from a specified results directory and prints the exploration
history stored in each safe model. The exploration history is expected to be a dictionary with keys
corresponding to iteration numbers and values as tuples of the form:
    (theta, t_val_candidate, safety_prob, reset_prob, uncertainty)

Usage:
    python print_exploration_history.py --results_dir <RESULTS_DIRECTORY>

Dependencies:
    - pickle, argparse, os
    - The safe model snapshots file ("safe_models_snapshots.pkl") must be present in the results directory.

Author: Luc Brogat-Motte
Date: 2025
"""

import os
import pickle
import argparse


def print_exploration_history(safe_model):
    """
    Print the exploration history stored in the safe model.

    Expected history: {iteration: (theta, t_val_candidate, safety_prob, reset_prob, uncertainty)}.

    Parameters:
        safe_model: A SafeSDE model instance with an 'exploration_history' attribute.
    """
    if not hasattr(safe_model, "exploration_history"):
        print("This safe model does not have an exploration history.")
        return

    history = safe_model.exploration_history
    if not history:
        print("Exploration history is empty.")
        return

    print("Exploration History:")
    for iteration in sorted(history.keys()):
        record = history[iteration]
        # record is expected to be a tuple:
        # (theta, t_val_candidate, safety_prob, reset_prob, uncertainty)
        print(f"Iteration {iteration}:")
        print(f"  Control (theta): {record[0]}")
        print(f"  t value:         {record[1]}")
        print(f"  Safety prob:     {record[2]}")
        print(f"  Reset prob:      {record[3]}")
        print(f"  Uncertainty:     {record[4]}")
        print("")


def main():
    parser = argparse.ArgumentParser(
        description="Print all exploration history from safe model snapshots."
    )
    DEFAULT_RESULTS_DIR = "kernel_1.0_lam_1e-05_betas_0.7"

    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Directory with experiment results and safe model snapshots")
    args = parser.parse_args()

    snapshots_file = os.path.join(args.results_dir, "safe_models_snapshots.pkl")
    if not os.path.exists(snapshots_file):
        print(f"Safe model snapshots file not found: {snapshots_file}")
        return

    with open(snapshots_file, "rb") as f:
        data = pickle.load(f)

    models = data.get("models", {})
    if not models:
        print("No safe models found in the snapshots file.")
        return

    for key, safe_model in models.items():
        print(f"\n--- Exploration history for safe model with (ε, ξ) = {key} ---")
        print_exploration_history(safe_model)


if __name__ == "__main__":
    main()

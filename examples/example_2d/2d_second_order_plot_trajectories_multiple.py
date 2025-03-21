import multiprocessing
import subprocess


def run_simulation(seed):
    """
    Run the trajectory plotting script with a given random seed.

    Parameters:
        seed (int): The random seed to be used by the simulation.
    """
    print(f"Running with seed {seed}")
    subprocess.run(["python", "2d_second_order_plot_trajectories.py", "--seed", str(seed)], check=True)


if __name__ == "__main__":
    # List of seeds for which to run the simulation.
    selected_seeds = [5, 7, 11, 17, 24, 35, 78, 59, 51, 98, 95, 93, 82]
    # Create a multiprocessing pool with a specified number of processes.
    with multiprocessing.Pool(processes=4) as pool:  # Adjust the number of processes as needed.
        pool.map(run_simulation, selected_seeds)

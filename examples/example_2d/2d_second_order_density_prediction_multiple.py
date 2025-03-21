import subprocess
import time

SCRIPT_NAME = "2d_second_order_density_prediction.py"
NUM_RUNS = 30  # Number of times to run the script


def run_script():
    for i in range(NUM_RUNS):
        print(f"Running {SCRIPT_NAME}, iteration {i + 1}/{NUM_RUNS}")
        start_time = time.time()
        subprocess.run(["python", SCRIPT_NAME], check=True)
        elapsed_time = time.time() - start_time
        print(f"Iteration {i + 1} completed in {elapsed_time:.2f} seconds\n")


if __name__ == "__main__":
    run_script()

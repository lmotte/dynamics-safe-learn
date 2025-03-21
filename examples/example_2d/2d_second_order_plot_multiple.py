import subprocess

# List of result directories corresponding to experiment configurations.
result_dirs = [
    "kernel_1.0_lam_1e-05_betas_0.7",
]

# Loop through each result directory and run the plotting script.
for res_dir in result_dirs:
    print(f"Analyzing results from {res_dir}")
    subprocess.run([
        "python", "2d_second_order_plot.py",
        "--results_dir", res_dir
    ], check=True)

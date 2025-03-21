import subprocess

# Define the list of result directories to analyze.
result_dirs = [
    "kernel_1.0_lam_1e-05_betas_0.7"
]

for res_dir in result_dirs:
    print(f"Analyzing results from {res_dir}")
    subprocess.run([
        "python", "2d_second_order_load_model.py",
        "--results_dir", res_dir
    ], check=True)

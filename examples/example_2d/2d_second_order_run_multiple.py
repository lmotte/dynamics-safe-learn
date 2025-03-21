import subprocess

# Define a list of parameter tuples: (kernel_func, lambda, betas)
# Each tuple represents one experiment configuration.
param_list = [(1.0, 1e-5, 0.7)]

# Loop over each parameter combination and run the experiment.
for param in param_list:
    kernel, lam, betas = param
    print(f"Running experiment with kernel_func={kernel}, lambda={lam}, betas={betas}")
    subprocess.run([
        "python", "2d_second_order_run.py",
        "--kernel_func", str(kernel),
        "--lam", str(lam),
        "--betas", str(betas)],
        check=True)

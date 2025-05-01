"""Third experiment: Piecewise constant schedules."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiment1 import create_random_problem, compute_exact_ot, sinkhorn_track

def run_piecewise_constant_experiment(key=jax.random.PRNGKey(0)):
    """Run piecewise constant schedules experiment."""
    print("Running piecewise constant schedules experiment...")

    # Create random problem
    p, q, c = create_random_problem(key)

    # Compute exact OT solution
    _, opt_cost = compute_exact_ot(p, q, c)
    print(f"Exact OT cost: {opt_cost:.4f}")

    # Experiment parameters
    niter = 2000
    beta0 = 10

    # Run configurations
    print("Running configurations...")
    # a. κ=1/2 with plateau_length=1 and 8
    _, err1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=1/2, beta0=beta0, debiased=False, plateau_length=1)
    _, err2 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=1/2, beta0=beta0, debiased=False, plateau_length=8)

    # b. κ=3/4 with plateau_length=1 and 8
    _, errb1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                             kappa=3/4, beta0=beta0, debiased=False, plateau_length=1)
    _, errb2 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                             kappa=3/4, beta0=beta0, debiased=False, plateau_length=8)

    # c. κ=2/3 (debiased) with plateau_length=1
    _, errc1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                             kappa=2/3, beta0=beta0, debiased=True, plateau_length=1)

    # Plotting
    print("Plotting results...")
    plt.figure(figsize=(7, 5))
    plt.rcParams.update({'font.size': 12})  # Match Julia's font size

    # Three pairs of curves:
    # err1/err2 (κ=1/2) in "C0" color
    plt.loglog(err1 + 1e-16, "C0", label=r"$\kappa=1/2$", lw=3)
    plt.semilogy(err2 + 1e-16, "C0--", label=r"$\kappa=1/2$ (piecewise cst)")

    # errb1/errb2 (κ=3/4) in "C1" color
    plt.semilogy(errb1 + 1e-16, "C1", label=r"$\kappa=3/4$", lw=3)
    plt.semilogy(errb2 + 1e-16, "C1--", label=r"$\kappa=3/4$ (piecewise cst)")

    # errc1 (κ=2/3 debiased) in "C3" color
    plt.semilogy(errc1 + 1e-16, "C3", label=r"$\kappa=2/3$ (debiased)", lw=3)

    plt.xlabel(r"Iteration $t$")
    plt.ylabel("OT suboptimality after projection")
    plt.grid("on")
    plt.legend()
    plt.savefig("output/piecewise-constant.png", bbox_inches="tight", dpi=200)
    plt.close()

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)

    # Set random seed
    key = jax.random.PRNGKey(1)  # Same seed as Julia

    # Run experiment
    run_piecewise_constant_experiment(key)

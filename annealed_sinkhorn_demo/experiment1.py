"""First experiment: Random cost matrix case."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple, NamedTuple
from functools import partial

from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from annealed_sinkhorn_demo.projection import project_to_marginals

def create_random_problem(key, size=75):
    """Create random OT problem (exactly as in Julia)."""
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Create random distributions
    p = jax.random.uniform(key1, (size,))
    q = jax.random.uniform(key2, (size,))
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)
    
    # Create random cost matrix
    c = jax.random.normal(key3, (size, size))
    c = (c - jnp.min(c)) / (jnp.max(c) - jnp.min(c))  # Match Julia normalization
    
    return p, q, c

def compute_exact_ot(p, q, c):
    """Compute exact OT cost using high regularization."""
    # Create geometry with low regularization for approximate exact solution
    geom = geometry.Geometry(cost_matrix=c, epsilon=1e-3)
    prob = linear_problem.LinearProblem(geom, a=p, b=q)
    solver = sinkhorn.Sinkhorn(threshold=1e-8)
    result = solver(prob)
    return result.matrix, jnp.sum(c * result.matrix)

def sinkhorn_track(p, q, c, niter=100, OT_cost=0.0, kappa=0.0, beta0=1.0, debiased=True, 
                   iter_stop=None, plateau_length=1):
    """Exact implementation of Julia's Sinkhorn_track."""
    if iter_stop is None:
        iter_stop = niter
    
    m, n = len(p), len(q)
    assert c.shape == (m, n)
    
    # Initialize potentials and parameters (as in Julia)
    u = jnp.zeros(m)
    v = jnp.zeros(n)
    lp = jnp.log(p)
    lq = jnp.log(q)
    beta = beta0
    
    plans = []
    errors = []
    
    for t in range(niter):
        # Update u (left potential)
        if debiased and t <= iter_stop:
            # Julia: temp1 = (lq .+ v .- β*c)  .- ((t^κ-(t-1)^κ)/t^κ) .* (u)
            t1 = t + 1  # Convert to 1-based indexing
            scale = (jnp.power(t1, kappa) - jnp.power(t1-1, kappa)) / jnp.power(t1, kappa)
            temp1 = lq[None, :] + v[None, :] - beta * c
            temp1 = temp1 - scale * u[:, None]  # Julia's broadcasting
        else:
            # Julia: temp1 = v .+ lq .- β*c
            temp1 = v[None, :] + lq[None, :] - beta * c
        
        # Stabilized u update (as in Julia)
        stab1 = jnp.max(temp1, axis=1, keepdims=True)
        u = -stab1.reshape(-1) - jnp.log(jnp.sum(jnp.exp(temp1 - stab1), axis=1))
        
        # Update beta (exactly as in Julia)
        if (t <= iter_stop) and (int(jnp.floor(jnp.sqrt(t+1))) % plateau_length == 0):
            beta = beta0 * jnp.power(t+2, kappa)  # t+2 for 1-based indexing
        
        # Update v (right potential)
        temp2 = u[:, None] + lp[:, None] - beta * c  # Julia's broadcasting
        stab2 = jnp.max(temp2, axis=0, keepdims=True)
        v = -stab2.reshape(-1) - jnp.log(jnp.sum(jnp.exp(temp2 - stab2), axis=0))
        
        # Compute transport plan (exactly as in Julia)
        logpi = u[:, None] + lp[:, None] + v[None, :] + lq[None, :] - beta * c
        pi = jnp.exp(logpi)
        pi = pi / jnp.sum(pi)  # Normalize
        
        # Project and compute error
        pi_proj = project_to_marginals(pi, p, q)
        error = jnp.sum(c * pi_proj) - OT_cost
        
        plans.append(pi)
        errors.append(error)
    
    return jnp.stack(plans), jnp.array(errors)

def run_pareto_front_experiment(key=jax.random.PRNGKey(0)):
    """Run Pareto front experiment from paper."""
    print("Running Pareto front experiment...")
    
    # Create random problem
    p, q, c = create_random_problem(key)
    
    # Compute exact OT solution
    _, opt_cost = compute_exact_ot(p, q, c)
    print(f"Exact OT cost: {opt_cost:.4f}")
    
    # Parameters exactly as in Julia
    niter = 1000
    beta0 = 10.0
    
    # Standard Sinkhorn with different β₀ values
    beta0s = jnp.logspace(1, 3, num=21)  # 10 to 1000
    print(f"Running standard Sinkhorn with {len(beta0s)} different β₀...")
    
    std_results = []
    for beta0_i in beta0s:
        # Run without annealing (κ=0) and no debiasing
        _, errors = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost, 
                                 kappa=0.0, beta0=beta0_i, debiased=False)
        std_results.append(errors)
    
    # Run annealed versions (as in Julia)
    print("Running annealed versions...")
    # Standard annealed (κ=1/2)
    _, err1 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=1/2, beta0=10.0, debiased=False)
    
    # Debiased annealed (κ=2/3)
    _, err2 = sinkhorn_track(p, q, c, niter=niter, OT_cost=opt_cost,
                            kappa=2/3, beta0=10.0, debiased=True)
    
    # Plot results (exactly matching Julia's style)
    plt.figure(figsize=(7, 5))
    
    # Plot standard Sinkhorn results
    for i, errors in enumerate(std_results):
        color = [i/len(beta0s), 0.5, 0.8]
        if i == 0:  # First β₀
            plt.loglog(errors, '--', color=color, alpha=0.8,
                      label=fr"standard Sinkhorn, $\beta={beta0s[0]:.0f}$")
        elif i == len(beta0s)-1:  # Last β₀
            plt.loglog(errors, '--', color=color, alpha=0.8,
                      label=fr"standard Sinkhorn, $\beta={beta0s[-1]:.0f}$")
        else:
            plt.loglog(errors, '--', color=color, alpha=0.5)
    
    # Plot annealed results (matching Julia's style)
    plt.loglog(err1, 'b-', label=r'Annealed Sinkhorn, $\kappa=1/2$', linewidth=4)
    plt.loglog(err2, 'r-', label=r'Debiased Annealed Sinkhorn, $\kappa=2/3$', linewidth=4)
    
    # Add theoretical rates (exactly as in Julia)
    t = jnp.arange(500, 1000, 10)
    plt.plot(t, t**(-1/2)/19, 'k-', linewidth=3)
    plt.text(600, 0.0025, "-1/2", color="black", size=10,
             bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'), rotation=-20)
    
    plt.plot(t, t**(-2/3)/30, 'k-', linewidth=3)
    plt.text(600, 0.00054, "-2/3", color="black", size=10,
             bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'), rotation=-25)
    
    plt.legend(fontsize=10, ncol=1)
    plt.xlabel(r'Iteration $t$')
    plt.ylabel('OT suboptimality after projection')
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    
    # Save plot with same settings as Julia
    plt.savefig('output/random-case.png', bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Set random seed
    key = jax.random.PRNGKey(1)  # Same seed as Julia
    
    # Run experiment
    run_pareto_front_experiment(key)

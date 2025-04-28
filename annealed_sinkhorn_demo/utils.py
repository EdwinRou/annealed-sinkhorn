"""Utility functions for visualization and analysis."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ott.geometry import geometry, pointcloud
from ott.problems.linear import linear_problem

def plot_convergence(errors, labels, title="Convergence Analysis"):
    """Plot convergence curves for different methods."""
    plt.figure(figsize=(10, 6))
    for err, label in zip(errors, labels):
        if err is not None:  # Handle case where solver didn't converge
            plt.loglog(err[err > -1], label=label)
    
    # Add theoretical rates
    t = jnp.arange(100, 1000)
    plt.plot(t, t**(-1/2)/10, 'k--', label=r'$O(t^{-1/2})$')
    plt.plot(t, t**(-2/3)/10, 'k:', label=r'$O(t^{-2/3})$')
    
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)

def run_comparison(data, methods, epsilons=None, n_iter=1000):
    """Run and compare different OT methods.
    
    Args:
        data: Either (x, y, a, b) for geometric case or (c, a, b) for random case
        methods: List of solvers to compare
        epsilons: List of epsilon values or schedulers for each method
        n_iter: Maximum number of iterations
    """
    if len(data) == 4:
        # Geometric case
        x, y, a, b = data
        geoms = [
            pointcloud.PointCloud(x, y, epsilon=eps) 
            for eps in (epsilons or [None] * len(methods))
        ]
    else:
        # Random case
        c, a, b = data
        geoms = [
            geometry.Geometry(cost_matrix=c, epsilon=eps)
            for eps in (epsilons or [None] * len(methods))
        ]
    
    results = []
    errors = []
    
    for method, geom in zip(methods, geoms):
        try:
            # Create OT problem
            prob = linear_problem.LinearProblem(geom, a=a, b=b)
            # Solve
            result = method(prob)
            results.append(result)
            errors.append(result.errors)
        except Exception as e:
            print(f"Error with solver {method.__class__.__name__}: {str(e)}")
            results.append(None)
            errors.append(None)
    
    return results, errors

def compare_transport_plans(plans, costs, titles):
    """Visualize and compare transport plans."""
    n = len(plans)
    plt.figure(figsize=(5*n, 4))
    
    for i, (P, cost, title) in enumerate(zip(plans, costs, titles)):
        if P is not None and cost is not None:
            plt.subplot(1, n, i+1)
            plt.imshow(P, cmap='RdBu')
            plt.colorbar()
            plt.title(f"{title}\nOT Cost: {cost:.4f}")
    
    plt.tight_layout()

def plot_geometric_transport(ot_result, title="Optimal Transport"):
    """Plot transport for geometric case."""
    plt.figure(figsize=(8, 6))
    plotter = plot.Plot()
    plotter.from_sinkhorn(ot_result, title=title)

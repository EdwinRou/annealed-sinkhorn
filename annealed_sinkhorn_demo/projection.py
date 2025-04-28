"""Implementation of exact projection to marginals."""

import jax
import jax.numpy as jnp

def project_to_marginals(gamma, p, q):
    """Project transport plan to match marginals exactly.
    
    This implements Algorithm 3 from Altschuler et al.
    
    Args:
        gamma: Transport plan matrix of shape (m, n)
        p: Source distribution of shape (m,)
        q: Target distribution of shape (n,)
        
    Returns:
        Projected transport plan that exactly matches marginals p and q.
    """
    # Project to first marginal
    gamma_sum_1 = jnp.sum(gamma, axis=1)
    a = jnp.minimum(p / gamma_sum_1, 1.0)
    gamma_temp = a[:, None] * gamma
    
    # Project to second marginal
    gamma_sum_2 = jnp.sum(gamma_temp, axis=0)
    b = jnp.minimum(q / gamma_sum_2, 1.0)
    gamma_temp = gamma_temp * b[None, :]
    
    # Handle remaining mass
    delta_p = p - jnp.sum(gamma_temp, axis=1)
    delta_q = q - jnp.sum(gamma_temp, axis=0)
    gamma_remaining = jnp.outer(delta_p, delta_q) / jnp.sum(delta_p)
    
    return gamma_temp + gamma_remaining

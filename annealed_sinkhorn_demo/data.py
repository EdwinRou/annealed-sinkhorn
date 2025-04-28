"""Data generation utilities for test cases."""

import jax
import jax.numpy as jnp

def create_geometric_case():
    """Create geometric test case matching the original paper."""
    rng = jax.random.key(1)  # Same seed as original
    keys = jax.random.split(rng, 8)
    
    # First measure: Two circles
    m = 150
    # First circle
    ang1 = jax.random.uniform(keys[0], (m,)) * jnp.pi
    rad1 = jax.random.uniform(keys[1], (m,)) * 0.2 + 0.3
    X1 = jnp.stack([rad1 * jnp.cos(ang1), rad1 * jnp.sin(ang1)], axis=1)
    
    # Second circle
    ang2 = jax.random.uniform(keys[2], (m,)) * 2*jnp.pi
    rad2 = jax.random.uniform(keys[3], (m,)) * 0.1
    X2 = jnp.stack([
        rad2 * jnp.cos(ang2) - 0.3,
        rad2 * jnp.sin(ang2) + 0.4
    ], axis=1)
    
    X = jnp.concatenate([X1, X2], axis=0)
    
    # Second measure: Vertical stripes
    n = 150
    # First stripe
    Y1 = jnp.stack([
        jax.random.uniform(keys[4], (n,)) - 0.5,
        jax.random.uniform(keys[5], (n,)) * 0.15
    ], axis=1)
    
    # Second stripe
    Y2 = jnp.stack([
        (jax.random.uniform(keys[6], (n,)) - 0.5) * 0.15,
        jax.random.uniform(keys[7], (n,)) * 0.6
    ], axis=1)
    
    Y = jnp.concatenate([Y1, Y2], axis=0)
    
    # Create uniform weights
    m, n = X.shape[0], Y.shape[0]
    p = jnp.ones(m) / m
    q = jnp.ones(n) / n
    
    # Compute cost matrix
    X_exp = X[:, None, :]  # Shape (m, 1, 2)
    Y_exp = Y[None, :, :]  # Shape (1, n, 2)
    c = jnp.sum((X_exp - Y_exp) ** 2, axis=2)
    
    # Normalize cost
    c = c / (jnp.max(c) - jnp.min(c))
    
    return X, Y, p, q, c

def create_random_case():
    """Create random test case matching the original paper."""
    rng = jax.random.key(1)  # Same seed as original
    keys = jax.random.split(rng, 3)
    
    m, n = 75, 75
    
    # Create random distributions
    p = jax.random.uniform(keys[0], (m,))
    q = jax.random.uniform(keys[1], (n,))
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)
    
    # Create random cost matrix
    c = jax.random.normal(keys[2], (m, n))
    c = c / (jnp.max(c) - jnp.min(c))
    
    return p, q, c

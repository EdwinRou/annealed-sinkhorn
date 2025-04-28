"""Core implementations for annealed Sinkhorn."""

import jax
import jax.numpy as jnp
from ott.geometry import epsilon_scheduler
from ott.solvers import linear

def project_to_marginals(pi, p, q):
    """Project transport plan to match marginals exactly."""
    m, n = pi.shape
    
    # Project to first marginal
    a = jnp.minimum(p / jnp.sum(pi, axis=1), 1.0)
    pi_temp = a[:, None] * pi
    
    # Project to second marginal
    b = jnp.minimum(q / jnp.sum(pi_temp, axis=0), 1.0)
    pi_temp = pi_temp * b[None, :]
    
    # Handle remaining mass
    delta_p = p - jnp.sum(pi_temp, axis=1)
    delta_q = q - jnp.sum(pi_temp, axis=0)
    pi_plan = pi_temp + jnp.outer(delta_p, delta_q) / jnp.sum(delta_p)
    
    return pi_plan

@jax.tree_util.register_pytree_node_class
class SinkhornTracker:
    """Track Sinkhorn iterations with various schedules."""
    
    def __init__(self, p, q, c, niter=1000, kappa=0.5, beta0=10.0, 
                 debiased=False, plateau_length=1):
        self.p = p
        self.q = q
        self.c = c
        self.niter = niter
        self.kappa = kappa
        self.beta0 = beta0
        self.debiased = debiased
        self.plateau_length = plateau_length
        
    def run(self, compute_opt_cost=False):
        """Run Sinkhorn with tracking."""
        m, n = self.p.shape[0], self.q.shape[0]
        
        # Initialize potentials
        u = jnp.zeros(m)
        v = jnp.zeros(n)
        
        # Log of marginals
        log_p = jnp.log(self.p)
        log_q = jnp.log(self.q)
        
        def body_fun(carry, t):
            u, v, beta, opt_cost = carry
            
            # Annealing schedule
            # Use where instead of if/else
            beta = jnp.where(
                (t > 0) & (t % self.plateau_length == 0),
                self.beta0 * ((t+1) ** self.kappa),
                beta
            )
            
            # Debiasing factor
            alpha_t = jnp.where(
                self.debiased & (t > 0),
                ((t+1)**self.kappa - t**self.kappa) / (t+1)**self.kappa,
                0.0
            )
            
            # Update u
            temp1 = v - beta * self.c
            stab1 = jnp.max(temp1, axis=1, keepdims=True)
            u_new = (1 - alpha_t) * u + log_p - stab1.reshape(-1) - jnp.log(
                jnp.sum(jnp.exp(temp1 - stab1), axis=1)
            )
            
            # Update v
            temp2 = u_new - beta * self.c.T
            stab2 = jnp.max(temp2, axis=1, keepdims=True)
            v_new = (1 - alpha_t) * v + log_q - stab2.reshape(-1) - jnp.log(
                jnp.sum(jnp.exp(temp2 - stab2), axis=1)
            )
            
            # Compute plan
            logpi = u_new[:, None] + v_new[None, :] - beta * self.c
            # Stabilize and normalize
            logpi = logpi - jnp.max(logpi)
            pi = jnp.exp(logpi)
            pi = pi / jnp.sum(pi)
            
            # Project to marginals
            pi_proj = project_to_marginals(pi, self.p, self.q)
            
            # Track error based on optimal cost or marginal violation
            error = jnp.where(
                compute_opt_cost,
                jnp.sum(self.c * pi_proj) - opt_cost,
                jnp.sum(jnp.abs(jnp.sum(pi, axis=1) - self.p))
            )
            
            return (u_new, v_new, beta, opt_cost), (pi, error)
        
        # Initialize scan carry
        init_carry = (u, v, self.beta0, jnp.array(0.0))  # Add opt_cost to carry
        
        # Run iterations
        _, (plans, errors) = jax.lax.scan(body_fun, init_carry, jnp.arange(self.niter))
        
        return plans, errors
    
    def tree_flatten(self):
        return (self.p, self.q, self.c), {
            'niter': self.niter,
            'kappa': self.kappa,
            'beta0': self.beta0,
            'debiased': self.debiased,
            'plateau_length': self.plateau_length
        }
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

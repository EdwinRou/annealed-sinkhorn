"""Implementation of annealing schedules for Sinkhorn algorithm."""

import jax
import jax.numpy as jnp

class AnnealingScheduler:
    """Scheduler for annealed Sinkhorn algorithm.
    
    Implements β_t = β_0(t+1)^κ with different κ values:
    - Standard annealed: κ = 1/2
    - Debiased version: κ = 2/3
    
    Matching Julia's implementation:
    - Updates beta only when mod(Int(floor(sqrt(t))),plateau_length)==0
    - Uses correct indexing for debiasing coefficient
    """
    def __init__(self, beta0=10.0, kappa=1/2, debiased=False, plateau_length=1, iter_stop=None):
        """Initialize scheduler.
        
        Args:
            beta0: Initial inverse temperature
            kappa: Exponent for annealing schedule
            debiased: Whether to use debiasing correction
            plateau_length: Length of constant-beta plateaus
            iter_stop: Stop updating beta after this iteration
        """
        self.beta0 = beta0
        self.kappa = kappa
        self.debiased = debiased
        self.plateau_length = plateau_length
        self.iter_stop = iter_stop
        
        # State
        self.current_beta = beta0
    
    def _should_update_beta(self, t):
        """Check if beta should be updated at iteration t."""
        # Julia: mod(Int(floor(sqrt(t))),plateau_length)==0
        sqrt_t = int(jnp.floor(jnp.sqrt(t + 1)))  # t+1 for 1-based indexing
        return sqrt_t % self.plateau_length == 0
    
    def get_beta(self, t):
        """Compute β_t value at iteration t."""
        if self.iter_stop is not None and t > self.iter_stop:
            return self.current_beta
            
        if self._should_update_beta(t):
            # Update using Julia's formula: β0 * ((t+1)^κ)
            self.current_beta = self.beta0 * jnp.power(t + 2, self.kappa)  # t+2 for 1-based indexing
            
        return self.current_beta
    
    def get_alpha(self, t):
        """Get debiasing coefficient α_t at iteration t.
        
        Implements Julia's formula: (t^κ-(t-1)^κ)/t^κ
        """
        if not self.debiased:
            return 0.0
            
        # Match Julia's indexing
        t = t + 1  # Convert to 1-based indexing
        curr_pow = jnp.power(t, self.kappa)
        prev_pow = jnp.power(t - 1, self.kappa)
        return jnp.where(t > 0, (curr_pow - prev_pow) / curr_pow, 0.0)
    
    def step(self, state, iteration):
        """Update state with annealed parameters.
        
        Args:
            state: Current state (potentials, transport plan)
            iteration: Current iteration number (0-based)
            
        Returns:
            Updated state with new parameters
        """
        beta = self.get_beta(iteration)
        alpha = self.get_alpha(iteration)
        return state._replace(beta=beta, alpha=alpha)

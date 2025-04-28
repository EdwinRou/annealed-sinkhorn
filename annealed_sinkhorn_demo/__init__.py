"""Annealed Sinkhorn implementation using ott-jax."""

from annealed_sinkhorn_demo.projection import project_to_marginals
from annealed_sinkhorn_demo.scheduler import AnnealingScheduler

__all__ = [
    'project_to_marginals',
    'AnnealingScheduler',
]

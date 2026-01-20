"""
Cloth Simulation Package

A differentiable cloth simulation using NVIDIA Warp for both forward simulation
and inverse parameter estimation via gradient-based optimization.
"""

from .config import SimConfig
from .geometry import make_grid_positions, make_edges_and_rests, make_pins
from .simulation import ClothSimulator
from .optimization import InverseSolver
from .visualization import animate_particles, plot_trajectories

__all__ = [
    "SimConfig",
    "make_grid_positions",
    "make_edges_and_rests",
    "make_pins",
    "ClothSimulator",
    "InverseSolver",
    "animate_particles",
    "plot_trajectories",
]


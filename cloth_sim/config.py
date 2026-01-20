"""
Configuration dataclass for cloth simulation parameters.
"""

from dataclasses import dataclass, field
from typing import Optional

import warp as wp


@dataclass
class SimConfig:
    """Configuration for the cloth simulation.

    Attributes:
        width: Number of particles in the x direction (grid width).
        height: Number of particles in the y direction (grid height).
        space_delta: Distance between adjacent particles in the grid.
        mass: Mass of each particle in kg.
        k: Spring constant (stiffness). Higher = stiffer cloth.
        c: Damping coefficient. Higher = less bouncy.
        dt: Time step for simulation (seconds per step).
        steps: Total number of simulation steps.
        g: Gravitational acceleration (m/s^2).
        use_diagonals: Whether to include diagonal springs (8-connectivity vs 4).
        device: Warp device to use ('cpu' or 'cuda:0', etc.).
    """

    width: int = 20
    height: int = 20
    space_delta: float = 0.05
    mass: float = 0.02
    k: float = 200.0
    c: float = 0.01
    dt: float = 1.0 / 240.0
    steps: int = 1200
    g: float = 9.81
    use_diagonals: bool = False
    device: Optional[str] = None

    def __post_init__(self):
        """Initialize warp and set default device if not specified."""
        if self.device is None:
            wp.init()
            self.device = str(wp.get_device())

    @property
    def num_particles(self) -> int:
        """Total number of particles in the cloth."""
        return self.width * self.height

    @property
    def wp_device(self):
        """Get the warp device object."""
        return wp.get_device(self.device)


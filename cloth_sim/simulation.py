"""
Cloth simulator class for running forward simulations.
"""

from typing import Optional

import numpy as np
import warp as wp

from .config import SimConfig
from .geometry import make_grid_positions, make_edges_and_rests, make_pins
from . import kernels


class ClothSimulator:
    """Cloth simulator using Warp for GPU-accelerated physics.

    This simulator can be used for both forward simulation and as part of
    an inverse optimization loop (with differentiable parameters).

    Attributes:
        config: Simulation configuration.
        pos: Current particle positions (warp array).
        vel: Current particle velocities (warp array).
        forces: Force accumulator (warp array).
        edges: Edge connectivity (warp array).
        rest: Edge rest lengths (warp array).
        masses: Particle masses (warp array).
        pins: Pin mask (warp array).
    """

    def __init__(self, config: SimConfig):
        """Initialize the cloth simulator.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self._device = config.wp_device
        self._init_arrays()

        # Create parameter arrays (length-1 for gradient support)
        self._k = wp.array(
            [config.k], dtype=wp.float32, device=self._device, requires_grad=False
        )
        self._c = wp.array(
            [config.c], dtype=wp.float32, device=self._device, requires_grad=False
        )

        # Store initial positions for reset
        self._initial_pos = self.pos.numpy().copy()

    def _init_arrays(self):
        """Initialize all simulation arrays from config."""
        config = self.config
        device = self._device

        # Create geometry in NumPy
        pos_np = make_grid_positions(config)
        vel_np = np.zeros_like(pos_np)
        edges_np, rest_np = make_edges_and_rests(config)
        pins_np = make_pins(config)
        masses_np = np.full(config.num_particles, config.mass, dtype=np.float32)

        # Transfer to Warp arrays
        self.pos = wp.array(pos_np, dtype=wp.vec2, device=device)
        self.vel = wp.array(vel_np, dtype=wp.vec2, device=device)
        self.edges = wp.array(edges_np, dtype=wp.int32, device=device)
        self.rest = wp.array(rest_np, dtype=wp.float32, device=device)
        self.pins = wp.array(pins_np, dtype=wp.int32, device=device)
        self.masses = wp.array(masses_np, dtype=wp.float32, device=device)
        self.forces = wp.zeros(config.num_particles, dtype=wp.vec2, device=device)

    def reset(self):
        """Reset simulation to initial state."""
        wp.copy(self.pos, wp.array(self._initial_pos, dtype=wp.vec2, device=self._device))
        wp.launch(
            kernels.zero_vec2,
            dim=self.config.num_particles,
            inputs=[self.vel],
            device=self._device,
        )

    def reset_to(self, positions: wp.array):
        """Reset positions to given state and zero velocities.

        Args:
            positions: Target positions to reset to.
        """
        wp.copy(self.pos, positions)
        wp.launch(
            kernels.zero_vec2,
            dim=self.config.num_particles,
            inputs=[self.vel],
            device=self._device,
        )

    def step(
        self,
        k: Optional[wp.array] = None,
        c: Optional[wp.array] = None,
    ):
        """Perform one simulation step.

        Args:
            k: Spring constant (length-1 array). If None, uses config value.
            c: Damping coefficient (length-1 array). If None, uses config value.
        """
        if k is None:
            k = self._k
        if c is None:
            c = self._c

        config = self.config
        device = self._device
        n_particles = config.num_particles
        n_edges = self.edges.shape[0]

        # Zero forces
        wp.launch(kernels.zero_vec2, dim=n_particles, inputs=[self.forces], device=device)

        # Compute spring forces
        wp.launch(
            kernels.spring_forces,
            dim=n_edges,
            inputs=[self.pos, self.edges, self.rest, k, self.forces],
            device=device,
        )

        # Compute damping forces
        wp.launch(
            kernels.damping_forces,
            dim=n_particles,
            inputs=[self.vel, c, self.forces],
            device=device,
        )

        # Integrate
        wp.launch(
            kernels.integrate,
            dim=n_particles,
            inputs=[
                self.pos,
                self.vel,
                self.forces,
                self.masses,
                self.pins,
                config.g,
                config.dt,
            ],
            device=device,
        )

    def run(
        self,
        steps: Optional[int] = None,
        record: bool = True,
        k: Optional[wp.array] = None,
        c: Optional[wp.array] = None,
    ) -> Optional[np.ndarray]:
        """Run simulation for multiple steps.

        Args:
            steps: Number of steps to run. If None, uses config.steps.
            record: Whether to record trajectory.
            k: Spring constant (length-1 array). If None, uses config value.
            c: Damping coefficient (length-1 array). If None, uses config value.

        Returns:
            If record=True, returns trajectory array of shape (steps, num_particles, 2).
            Otherwise returns None.
        """
        if steps is None:
            steps = self.config.steps

        trajectory = [] if record else None

        for _ in range(steps):
            self.step(k=k, c=c)
            if record:
                trajectory.append(self.pos.numpy().copy())

        if record:
            return np.array(trajectory)
        return None

    def get_positions(self) -> np.ndarray:
        """Get current positions as numpy array."""
        return self.pos.numpy().copy()

    def get_free_mask(self) -> np.ndarray:
        """Get mask for free (non-pinned) particles.

        Returns:
            Array of shape (num_particles,) with 1.0 for free, 0.0 for pinned.
        """
        pins = self.pins.numpy()
        return (1 - pins).astype(np.float32)


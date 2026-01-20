"""
Inverse solver for parameter estimation via gradient-based optimization.
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .config import SimConfig
from .simulation import ClothSimulator
from . import kernels


@dataclass
class OptimizationResult:
    """Result of inverse optimization.

    Attributes:
        k: Estimated spring constant.
        c: Estimated damping coefficient.
        loss_history: List of loss values during optimization.
        k_history: List of k values during optimization.
        c_history: List of c values during optimization.
    """

    k: float
    c: float
    loss_history: List[float] = field(default_factory=list)
    k_history: List[float] = field(default_factory=list)
    c_history: List[float] = field(default_factory=list)


class InverseSolver:
    """Inverse problem solver for cloth parameter estimation.

    Uses gradient-based optimization with wp.Tape() to estimate spring constant
    and damping coefficient from a target trajectory.

    The parameters are optimized in log-space to ensure they remain positive.
    """

    def __init__(
        self,
        simulator: ClothSimulator,
        target_trajectory: np.ndarray,
        k_init: float = 100.0,
        c_init: float = 1e-3,
    ):
        """Initialize the inverse solver.

        Args:
            simulator: The cloth simulator to use.
            target_trajectory: Ground truth trajectory of shape (steps, particles, 2).
            k_init: Initial guess for spring constant.
            c_init: Initial guess for damping coefficient.
        """
        self.simulator = simulator
        self.config = simulator.config
        self._device = simulator._device

        # Store target trajectory as warp arrays (one per frame)
        self.target_trajectory = target_trajectory
        self.target_wp = [
            wp.array(frame.astype(np.float32), dtype=wp.vec2, device=self._device)
            for frame in target_trajectory
        ]

        # Create differentiable parameter arrays in log-space
        self.log_k = wp.array(
            np.array([np.log(k_init)], dtype=np.float32),
            dtype=wp.float32,
            device=self._device,
            requires_grad=True,
        )
        self.log_c = wp.array(
            np.array([np.log(c_init)], dtype=np.float32),
            dtype=wp.float32,
            device=self._device,
            requires_grad=True,
        )

        # Create mask for free particles
        free_mask = simulator.get_free_mask()
        self.mask = wp.array(free_mask, dtype=wp.float32, device=self._device)

        # Create temporary arrays for parameter exponentiation
        self._k_arr = wp.zeros(1, dtype=wp.float32, device=self._device, requires_grad=True)
        self._c_arr = wp.zeros(1, dtype=wp.float32, device=self._device, requires_grad=True)

    def get_params(self) -> Tuple[float, float]:
        """Get current parameter estimates.

        Returns:
            Tuple of (k, c) in original (non-log) space.
        """
        k = float(np.exp(self.log_k.numpy()[0]))
        c = float(np.exp(self.log_c.numpy()[0]))
        return k, c

    def rollout_and_loss(self, frame_stride: int = 8) -> wp.array:
        """Run simulation and compute loss against target.

        Args:
            frame_stride: Compute loss every N frames (for efficiency).

        Returns:
            Total loss as a length-1 warp array (for gradient computation).
        """
        config = self.config
        n_particles = config.num_particles
        steps = len(self.target_wp)

        # Reset simulator to initial position from target
        self.simulator.reset_to(self.target_wp[0])

        # Exponentiate log parameters to get actual k, c
        # Using in-place operations to maintain gradient flow
        k_val = wp.exp(self.log_k)
        c_val = wp.exp(self.log_c)

        # Total loss accumulator
        total = wp.zeros(1, dtype=float, device=self._device, requires_grad=True)

        # Run simulation
        for t in range(1, steps):
            # Perform one step
            self.simulator.step(k=k_val, c=c_val)

            # Compute loss at stride intervals
            if (t % frame_stride) == 0 or t == steps - 1:
                tmp = wp.zeros(
                    n_particles, dtype=float, device=self._device, requires_grad=True
                )
                wp.launch(
                    kernels.mse_frame,
                    dim=n_particles,
                    inputs=[self.simulator.pos, self.target_wp[t], self.mask, tmp],
                    device=self._device,
                )
                wp.launch(
                    kernels.accumulate_loss,
                    dim=1,
                    inputs=[tmp, self.mask, total],
                    device=self._device,
                )

        return total

    def optimize(
        self,
        iterations: int = 200,
        lr: float = 0.1,
        frame_stride: int = 8,
        grad_clip: float = 10.0,
        verbose: bool = True,
        print_every: int = 10,
    ) -> OptimizationResult:
        """Run optimization to estimate parameters.

        Args:
            iterations: Number of optimization iterations.
            lr: Learning rate.
            frame_stride: Compute loss every N frames.
            grad_clip: Maximum gradient norm (for stability).
            verbose: Whether to print progress.
            print_every: Print every N iterations.

        Returns:
            OptimizationResult with estimated parameters and history.
        """
        loss_history = []
        k_history = []
        c_history = []

        for it in range(iterations):
            # Forward pass with tape
            with wp.Tape() as tape:
                loss = self.rollout_and_loss(frame_stride=frame_stride)

            # Backward pass
            tape.backward(loss)

            # Get gradients
            gk = float(tape.gradients[self.log_k].numpy()[0])
            gc = float(tape.gradients[self.log_c].numpy()[0])

            # Gradient clipping
            gnorm = float(np.sqrt(gk**2 + gc**2))
            if gnorm > grad_clip:
                gk *= grad_clip / gnorm
                gc *= grad_clip / gnorm

            # SGD update step
            wp.launch(
                kernels.update_param,
                dim=1,
                inputs=[self.log_k, gk, lr],
                device=self._device,
            )
            wp.launch(
                kernels.update_param,
                dim=1,
                inputs=[self.log_c, gc, lr],
                device=self._device,
            )

            # Record history
            loss_val = float(loss.numpy()[0])
            k, c = self.get_params()
            loss_history.append(loss_val)
            k_history.append(k)
            c_history.append(c)

            if verbose and it % print_every == 0:
                print(f"iter {it:03d}  loss={loss_val:.6f}  k={k:.3f}  c={c:.6e}")

        k_final, c_final = self.get_params()
        if verbose:
            print(f"\nFinal parameters: k={k_final:.3f}, c={c_final:.6e}")

        return OptimizationResult(
            k=k_final,
            c=c_final,
            loss_history=loss_history,
            k_history=k_history,
            c_history=c_history,
        )


def save_trajectory(trajectory: np.ndarray, path: str):
    """Save a trajectory to a numpy file.

    Args:
        trajectory: Array of shape (frames, particles, 2).
        path: Path to save the file.
    """
    np.save(path, trajectory)
    print(f"Saved trajectory to {path} with shape {trajectory.shape}")


def load_trajectory(path: str) -> np.ndarray:
    """Load a trajectory from a numpy file.

    Args:
        path: Path to the trajectory file.

    Returns:
        Trajectory array of shape (frames, particles, 2).
    """
    trajectory = np.load(path)
    print(f"Loaded trajectory from {path} with shape {trajectory.shape}")
    return trajectory


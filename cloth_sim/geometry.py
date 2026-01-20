"""
Cloth geometry creation functions.

These functions create the initial particle positions, edge connectivity,
rest lengths, and pin constraints for the cloth simulation.
"""

from typing import Tuple

import numpy as np

from .config import SimConfig


def make_grid_positions(config: SimConfig) -> np.ndarray:
    """Create initial grid positions for cloth particles.

    Particles are arranged in a regular grid starting from the bottom-left corner.

    Args:
        config: Simulation configuration.

    Returns:
        Array of shape (num_particles, 2) containing 2D positions.
    """
    w, h, dx = config.width, config.height, config.space_delta
    x = np.zeros((w * h, 2), dtype=np.float32)

    for j in range(h):
        for i in range(w):
            idx = j * w + i
            x[idx, 0] = i * dx
            x[idx, 1] = j * dx

    return x


def make_edges_and_rests(config: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Create edge connectivity and rest lengths for the spring system.

    Each edge connects two adjacent particles. The rest length is the
    initial distance between the particles.

    Args:
        config: Simulation configuration.

    Returns:
        Tuple of:
            - edges: Array of shape (num_edges, 2) with particle indices.
            - rest_lengths: Array of shape (num_edges,) with rest lengths.
    """
    w, h, dx = config.width, config.height, config.space_delta
    use_diagonals = config.use_diagonals

    edges = []
    for j in range(h):
        for i in range(w):
            curr = j * w + i
            # Horizontal edge (right neighbor)
            if i + 1 < w:
                edges.append((curr, curr + 1))
            # Vertical edge (top neighbor)
            if j + 1 < h:
                edges.append((curr, (j + 1) * w + i))

            # Diagonal edges (if enabled)
            if use_diagonals:
                # Top-right diagonal
                if i + 1 < w and j + 1 < h:
                    edges.append((curr, (j + 1) * w + (i + 1)))
                # Top-left diagonal
                if i - 1 >= 0 and j + 1 < h:
                    edges.append((curr, (j + 1) * w + (i - 1)))

    edges = np.array(edges, dtype=np.int32)

    # Compute rest lengths from initial positions
    pos = make_grid_positions(config)
    rest_lengths = np.zeros(edges.shape[0], dtype=np.float32)

    for e, (a, b) in enumerate(edges):
        dir_vec = pos[b] - pos[a]
        rest_lengths[e] = np.linalg.norm(dir_vec)

    return edges, rest_lengths


def make_pins(config: SimConfig) -> np.ndarray:
    """Create pin mask for the cloth.

    Pinned particles are fixed in place and don't move during simulation.
    By default, the top-left and top-right corners are pinned.

    Args:
        config: Simulation configuration.

    Returns:
        Array of shape (num_particles,) with 1 for pinned, 0 for free.
    """
    w, h = config.width, config.height
    pins = np.zeros(w * h, dtype=np.int32)

    # Pin top-left corner
    pins[(h - 1) * w] = 1
    # Pin top-right corner
    pins[(h - 1) * w + w - 1] = 1

    return pins


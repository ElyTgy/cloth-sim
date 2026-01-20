"""
Visualization utilities for cloth simulation.
"""

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_particles(trajectory, save_gif=True):
    """Create an animated scatter plot of particle positions"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        positions = trajectory[frame]
        ax.scatter(positions[:, 0], positions[:, 1], s=20, alpha=0.7)
        ax.set_xlim(trajectory[:, :, 0].min() - 0.1, trajectory[:, :, 0].max() + 0.1)
        ax.set_ylim(trajectory[:, :, 1].min() - 0.1, trajectory[:, :, 1].max() + 0.1)
        ax.set_title(f'Cloth Simulation - Frame {frame}/{len(trajectory)}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                                 interval=50, repeat=True)
    
    if save_gif:
        anim.save('cloth_animation.mp4', writer='ffmpeg')

    
    #plt.show()
    return anim


def plot_trajectories(
    trajectory: np.ndarray,
    particle_indices: Optional[List[int]] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot 3D trajectories of selected particles over time.

    Args:
        trajectory: Array of shape (frames, num_particles, 2) containing positions.
        particle_indices: Indices of particles to plot. If None, plots a sample.
        figsize: Figure size.

    Returns:
        The matplotlib figure.
    """
    if particle_indices is None:
        # Sample some particles across the cloth
        num_particles = trajectory.shape[1]
        particle_indices = list(range(0, num_particles, max(1, num_particles // 10)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    frames = np.arange(len(trajectory))

    for idx in particle_indices:
        x = trajectory[:, idx, 0]
        y = trajectory[:, idx, 1]
        ax.plot(x, y, frames, label=f"Particle {idx}", alpha=0.7)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Time (frame)")
    ax.set_title("Particle Trajectories Over Time")
    ax.legend(loc="upper left", fontsize="small")

    return fig


def plot_particle_over_time(
    trajectory: np.ndarray,
    particle_index: int,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot x and y position of a single particle over time.

    Args:
        trajectory: Array of shape (frames, num_particles, 2) containing positions.
        particle_index: Index of the particle to plot.
        figsize: Figure size.

    Returns:
        The matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    frames = np.arange(len(trajectory))
    x = trajectory[:, particle_index, 0]
    y = trajectory[:, particle_index, 1]

    ax1.plot(frames, x)
    ax1.set_xlabel("Time (frame)")
    ax1.set_ylabel("X Position")
    ax1.set_title(f"Particle {particle_index} - X Position")
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames, y)
    ax2.set_xlabel("Time (frame)")
    ax2.set_ylabel("Y Position")
    ax2.set_title(f"Particle {particle_index} - Y Position")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


"""
Cloth Simulation using NVIDIA Warp

This module implements a 2D cloth simulation using spring-mass systems.
The cloth is modeled as a grid of particles connected by springs, with
gravity, damping, and pinned boundary conditions.
"""

import warp as wp
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Cloth grid configuration
W, H = 20, 20  # Grid size (width x height) - creates 400 particles
SPACE_DELTA = 0.05  # Distance between particles (affects cloth detail)
MASS = 0.02  # Mass per particle in kg

# Spring system parameters
K = 200.0  # Spring constant (higher = stiffer, less stretchy cloth)
C = 0.001  # Damping coefficient (higher = less bouncy)

# Simulation control
DT = 1.0 / 240.0  # Time step (240 FPS)
STEPS = 1200  # Total simulation steps

# Environment
G = 9.81  # Gravitational acceleration (m/s²)

# Configuration flags
USE_DIAGONALS = True  # Use diagonal springs (8 neighbors instead of 4)
DEVICE = wp.get_device()  # Warp device (CPU/GPU)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def make_grid_positions(w: int, h: int, dx: float) -> np.ndarray:
    """
    Create initial grid positions for cloth particles.
    
    Args:
        w: Grid width
        h: Grid height  
        dx: Spacing between particles
        
    Returns:
        Array of shape (w*h, 2) containing 2D positions
    """
    positions = np.zeros((w * h, 2), dtype=np.float32)
    
    for j in range(h):
        for i in range(w):
            idx = j * w + i
            positions[idx, 0] = i * dx  # x-coordinate
            positions[idx, 1] = j * dx  # y-coordinate
    
    return positions


def make_edges_and_rests(w: int, h: int, dx: float, diagonals: bool = False) -> tuple:
    """
    Create spring connections between particles and their rest lengths.
    
    Args:
        w: Grid width
        h: Grid height
        dx: Spacing between particles
        diagonals: Whether to include diagonal springs
        
    Returns:
        Tuple of (edges, rest_lengths) where:
        - edges: Array of shape (n_edges, 2) with particle indices
        - rest_lengths: Array of shape (n_edges,) with spring rest lengths
    """
    edges = []
    
    # Create horizontal and vertical connections
    for j in range(h):
        for i in range(w):
            # Horizontal edge (right neighbor)
            if i + 1 < w:
                edges.append((j * w + i, j * w + (i + 1)))
            # Vertical edge (bottom neighbor)  
            if j + 1 < h:
                edges.append((j * w + i, (j + 1) * w + i))
    
    # Add diagonal connections if requested
    if diagonals:
        for j in range(h):
            for i in range(w):
                # Diagonal down-right
                if i + 1 < w and j + 1 < h:
                    edges.append((j * w + i, (j + 1) * w + (i + 1)))
                # Diagonal down-left
                if i > 0 and j + 1 < h:
                    edges.append((j * w + i, (j + 1) * w + (i - 1)))
    
    edges = np.array(edges, dtype=np.int32)
    rest_lengths = np.zeros(edges.shape[0], dtype=np.float32)
    positions = make_grid_positions(w, h, dx)
    
    # Calculate rest lengths for each spring
    for e, (a, b) in enumerate(edges):
        direction_vector = positions[b] - positions[a]
        rest_lengths[e] = np.linalg.norm(direction_vector)
    
    return edges, rest_lengths


def make_pins(w: int, h: int) -> np.ndarray:
    """
    Create pinning mask for boundary particles.
    
    Args:
        w: Grid width
        h: Grid height
        
    Returns:
        Array of shape (w*h,) where 1 indicates pinned particles
    """
    pins = np.zeros((w * h,), dtype=wp.int32)
    # Pin top-left and top-right corners
    pins[0] = 1      # Top-left corner
    pins[w - 1] = 1  # Top-right corner
    return pins


# =============================================================================
# WARP KERNELS
# =============================================================================

@wp.kernel
def zero_vec2(a: wp.array(dtype=wp.vec2)):
    """Zero out a vector array."""
    i = wp.tid()
    a[i] = wp.vec2(0.0, 0.0)


@wp.kernel
def spring_forces(pos: wp.array(dtype=wp.vec2),
                 edges: wp.array2d(dtype=wp.int32),
                 rest: wp.array(dtype=wp.float32),
                 k: wp.array(dtype=wp.float32),
                 f: wp.array(dtype=wp.vec2)):
    """
    Calculate spring forces between connected particles.
    
    Uses Hooke's law: F = -k * (L - L0) * direction
    """
    e = wp.tid()  # Current edge index
    
    # Get connected particle indices
    i = edges[e][0]
    j = edges[e][1]
    
    # Calculate spring force
    direction_vector = pos[i] - pos[j]
    current_length = wp.length(direction_vector) + 1e-6  # Prevent division by zero
    direction_unit = direction_vector / current_length
    spring_force = -k[0] * (current_length - rest[e]) * direction_unit
    
    # Apply forces to both particles (Newton's third law)
    wp.atomic_add(f, i, spring_force)
    wp.atomic_add(f, j, -spring_force)


@wp.kernel
def damping_forces(v: wp.array(dtype=wp.vec2),
                  c: wp.array(dtype=wp.float32),
                  f: wp.array(dtype=wp.vec2)):
    """
    Calculate damping forces proportional to velocity.
    
    Damping force: F = -c * velocity
    """
    i = wp.tid()
    wp.atomic_add(f, i, -c[0] * v[i])


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec2),
             velocities: wp.array(dtype=wp.vec2),
             forces: wp.array(dtype=wp.vec2),
             masses: wp.array(dtype=wp.float32),
             pinned: wp.array(dtype=wp.int32),
             g: wp.float32,
             dt: wp.float32):
    """
    Integrate particle positions and velocities using Euler method.
    
    Args:
        verlet: Whether to use Verlet integration (not implemented)
    """
    i = wp.tid()
    
    # Skip pinned particles
    if pinned[i] == 1:
        velocities[i] = wp.vec2(0.0, 0.0)
        return
    
    # Calculate acceleration: a = F/m + g
    acceleration = forces[i] / masses[i] + wp.vec2(0.0, -g)
    
    # Euler integration: v = v + a*dt, x = x + v*dt
    velocities[i] = velocities[i] + acceleration * dt
    pos[i] = pos[i] + velocities[i] * dt


# =============================================================================
# SIMULATION SETUP AND EXECUTION
# =============================================================================
def setup_data():
    """
    Initialize simulation data structures.
    
    Returns:
        Tuple of (pos, velocity, edges, rest, pins, masses, forces)
    """
    # Create numpy arrays
    pos_np = make_grid_positions(W, H, SPACE_DELTA).astype(np.float32)
    velocity_np = np.zeros_like(pos_np)
    edges_np, rest_np = make_edges_and_rests(W, H, SPACE_DELTA, USE_DIAGONALS)
    pins_np = make_pins(W, H)
    masses_np = np.full(W * H, MASS)
    
    # Convert to Warp arrays
    pos = wp.array(pos_np, dtype=wp.vec2, device=DEVICE)
    velocity = wp.array(velocity_np, dtype=wp.vec2, device=DEVICE)
    edges = wp.array2d(edges_np, dtype=wp.int32, device=DEVICE)
    rest = wp.array(rest_np, dtype=wp.float32, device=DEVICE)
    pins = wp.array(pins_np, dtype=wp.int32, device=DEVICE)
    masses = wp.array(masses_np, dtype=wp.float32, device=DEVICE)
    forces = wp.zeros(W * H, dtype=wp.vec2, device=DEVICE)
    
    return pos, velocity, edges, rest, pins, masses, forces


def run_sim() -> list:
    """
    Run the cloth simulation.
    
    Returns:
        List of position arrays for each simulation step
    """
    print(f"Starting cloth simulation...")
    print(f"Grid size: {W}x{H} ({W*H} particles)")
    print(f"Time step: {DT:.6f}s, Total steps: {STEPS}")
    print(f"Spring constant: {K}, Damping: {C}")
    print(f"Using diagonals: {USE_DIAGONALS}")
    
    # Initialize simulation data
    pos, velocity, edges, rest, pins, masses, forces = setup_data()
    
    frames = []
    
    #append the first frame here, before running the forward sim loop
    frames.append(pos.numpy().copy())

    # Main simulation loop
    for step in range(STEPS):
        # Reset forces to zero
        wp.launch(zero_vec2, dim=W*H, inputs=[forces], device=DEVICE)
        
        # Calculate spring forces
        wp.launch(spring_forces, dim=edges.shape[0], 
                 inputs=[pos, edges, rest, wp.array([float(K)], dtype=wp.float32), forces], device=DEVICE)
        
        # Calculate damping forces
        wp.launch(damping_forces, dim=W*H, 
                 inputs=[velocity, wp.array([float(C)], dtype=wp.float32), forces], device=DEVICE)
        
        # Integrate positions and velocities
        wp.launch(integrate, dim=W*H, 
                 inputs=[pos, velocity, forces, masses, pins, float(G), float(DT)], 
                 device=DEVICE)
        
        # Store current frame
        frames.append(pos.numpy().copy())
        
        # Progress indicator
        if step % 100 == 0:
            print(f"Step {step}/{STEPS} completed")
    
    print("Simulation completed!")
    return frames


def save_sim(sim_frames: list) -> None:
    """
    Save simulation trajectory to file.
    
    Args:
        sim_frames: List of position arrays for each simulation step
    """
    trajectory = np.stack(sim_frames)  # Shape: (frames, particle_count, 2)
    np.save("cloth_target.npy", trajectory)
    print(f"Saved ground-truth to cloth_target.npy with shape {trajectory.shape}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    frames = run_sim()
    save_sim(frames)
"""
Warp kernels for cloth simulation.

All kernels use array parameters (even for scalar values like k, c) to enable
gradient computation via wp.Tape() for the inverse problem.

Note: Kernels must be defined at module level (not inside classes) per Warp requirements.
"""

import warp as wp


@wp.kernel
def zero_vec2(a: wp.array(dtype=wp.vec2)):
    """Zero out a vec2 array.

    Args:
        a: Array to zero out (modified in place).
    """
    i = wp.tid()
    a[i] = wp.vec2(0.0, 0.0)


@wp.kernel
def spring_forces(
    pos: wp.array(dtype=wp.vec2),
    edges: wp.array2d(dtype=wp.int32),
    rest: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),
    f: wp.array(dtype=wp.vec2),
):
    """Compute spring forces between connected particles.

    Uses Hooke's law: F = -k * (L - L_rest) * direction

    Args:
        pos: Particle positions.
        edges: Edge connectivity (N x 2 array of particle indices).
        rest: Rest lengths for each edge.
        k: Spring constant (length-1 array for gradient support).
        f: Force accumulator (modified via atomic_add).
    """
    e = wp.tid()

    # Read spring constant from array (enables gradient computation)
    k_val = k[0]

    # Get connected particle indices
    i = edges[e][0]
    j = edges[e][1]

    # Compute spring force
    dir_vec = pos[i] - pos[j]
    L = wp.length(dir_vec) + 1e-6  # Prevent division by zero
    dir_unit = dir_vec / L
    Fs = -k_val * (L - rest[e]) * dir_unit

    # Apply forces to both particles (Newton's third law)
    wp.atomic_add(f, i, Fs)
    wp.atomic_add(f, j, -Fs)


@wp.kernel
def damping_forces(
    v: wp.array(dtype=wp.vec2),
    c: wp.array(dtype=wp.float32),
    f: wp.array(dtype=wp.vec2),
):
    """Apply viscous damping forces.

    F_damp = -c * v

    Args:
        v: Particle velocities.
        c: Damping coefficient (length-1 array for gradient support).
        f: Force accumulator (modified via atomic_add).
    """
    i = wp.tid()
    c_val = c[0]
    wp.atomic_add(f, i, -c_val * v[i])


@wp.kernel
def integrate(
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    forces: wp.array(dtype=wp.vec2),
    masses: wp.array(dtype=wp.float32),
    pinned: wp.array(dtype=wp.int32),
    g: float,
    dt: float,
):
    """Integrate particle positions using explicit Euler method.

    Args:
        pos: Particle positions (modified in place).
        vel: Particle velocities (modified in place).
        forces: Forces on each particle.
        masses: Mass of each particle.
        pinned: Pin mask (1 = pinned, 0 = free).
        g: Gravitational acceleration.
        dt: Time step.
    """
    i = wp.tid()

    # Pinned particles don't move
    if pinned[i] == 1:
        vel[i] = wp.vec2(0.0, 0.0)
        return

    # Compute acceleration (F/m + gravity)
    acceleration = forces[i] / masses[i] + wp.vec2(0.0, -g)

    # Euler integration
    vel[i] = vel[i] + acceleration * dt
    pos[i] = pos[i] + vel[i] * dt


@wp.kernel
def mse_frame(
    x_sim: wp.array(dtype=wp.vec2),
    x_gt: wp.array(dtype=wp.vec2),
    mask: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    """Compute per-particle MSE between simulation and ground truth.

    Args:
        x_sim: Simulated particle positions.
        x_gt: Ground truth particle positions.
        mask: Mask for free particles (1 = include, 0 = exclude).
        out: Per-particle squared error output.
    """
    i = wp.tid()
    d = x_sim[i] - x_gt[i]
    out[i] = mask[i] * wp.dot(d, d)


@wp.kernel
def accumulate_loss(
    tmp: wp.array(dtype=float),
    mask: wp.array(dtype=float),
    total: wp.array(dtype=float),
):
    """Accumulate per-particle MSE into total loss.

    Normalizes by the number of free (unmasked) particles.
    Should be launched with dim=1 (single thread).

    Args:
        tmp: Per-particle squared errors.
        mask: Mask for free particles.
        total: Accumulated loss (modified in place).
    """
    sum_err = float(0.0)
    sum_mask = float(0.0)
    for i in range(tmp.shape[0]):
        sum_err = sum_err + tmp[i]
        sum_mask = sum_mask + mask[i]
    total[0] = total[0] + sum_err / (sum_mask + 1e-8)


@wp.kernel
def update_param(
    param: wp.array(dtype=float),
    grad: float,
    lr: float,
):
    """Update a parameter using gradient descent.

    Args:
        param: Parameter array (length 1, modified in place).
        grad: Gradient value.
        lr: Learning rate.
    """
    param[0] = param[0] - lr * grad


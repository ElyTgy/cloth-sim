# Cloth Simulation with Inverse Parameter Estimation

A GPU-accelerated cloth simulation framework built with NVIDIA Warp that supports both **forward simulation** and **inverse parameter estimation** through differentiable physics. This project demonstrates how automatic differentiation can be used to solve inverse problems in physics-based animation.

## Overview

This project implements a mass-spring system for cloth dynamics, where a rectangular cloth is modeled as a grid of particles connected by springs. The simulation supports:

- **Forward Simulation**: Given physical parameters (spring constant `k`, damping coefficient `c`), simulate cloth behavior over time
- **Inverse Optimization**: Given a target trajectory, automatically estimate the physical parameters that would produce it using gradient-based optimization

## Technical Architecture

### Physics Model

The cloth is modeled using a **mass-spring system**:

1. **Particle Grid**: Cloth is discretized into a regular grid of `width × height` particles
2. **Spring Connections**: Each particle is connected to its neighbors via springs:
   - Horizontal and vertical connections (4-connectivity)
   - Optional diagonal connections (8-connectivity) for increased stability
3. **Pin Constraints**: Top corners are pinned to simulate a hanging cloth

### Force Computation

The simulation computes three types of forces:

1. **Spring Forces** (Hooke's Law):
   ```
   F_spring = -k × (L - L_rest) × direction
   ```
   where `k` is the spring constant, `L` is current length, and `L_rest` is rest length.

2. **Damping Forces** (Viscous Damping):
   ```
   F_damp = -c × v
   ```
   where `c` is the damping coefficient and `v` is velocity.

3. **Gravity**:
   ```
   F_gravity = -m × g × ŷ
   ```

### Integration Scheme

Positions and velocities are updated using **Explicit Euler integration**:
```
a = F_total / m
v_{t+1} = v_t + a × dt
x_{t+1} = x_t + v_{t+1} × dt
```

### Differentiable Simulation

The entire simulation is implemented using **NVIDIA Warp**, a high-performance computing framework that:

- Runs on GPU (CUDA) for fast parallel computation
- Provides automatic differentiation via `wp.Tape()` for gradient computation
- Enables end-to-end differentiation from loss function back to physical parameters

### Inverse Problem Formulation

The inverse problem estimates parameters `θ = {k, c}` that minimize the discrepancy between simulated and target trajectories:

```
minimize: L(θ) = Σ_t ||x_sim(t, θ) - x_target(t)||²
```

Where:
- `x_sim(t, θ)` is the simulated trajectory at time `t` with parameters `θ`
- `x_target(t)` is the observed/desired trajectory
- The loss is computed at regular intervals (frame stride) for efficiency

**Optimization Strategy**:
- Parameters are optimized in **log-space** (`log_k`, `log_c`) to ensure positivity
- Gradient descent with learning rate scheduling
- Gradient clipping for numerical stability
- Only free (non-pinned) particles contribute to the loss

## Installation

### Prerequisites

- Python 3.7+
- NVIDIA GPU with CUDA support (for GPU acceleration, otherwise it will run on your cpu more slowly)
- CUDA Toolkit 11.0 or later
- ffmpeg

### Dependencies

Install required packages:

```bash
pip install numpy warp-lang matplotlib
```

**Note**: `warp-lang` is NVIDIA's Warp library. If you don't have a GPU, the simulation will fall back to CPU mode.

### Verify Installation

```bash
python -c "import warp as wp; wp.init(); print(wp.get_device())"
```

## Usage

### Forward Simulation

Run a forward simulation to generate a cloth animation:

```bash
python main.py forward --width 20 --height 20 \
    --k 200 --c 0.01 \
    --steps 1200 --dt 1/240 \
    --save trajectory.npy --animate
```

**Parameters**:
- `--width`, `--height`: Grid dimensions (default: 20×20)
- `--k`: Spring constant (stiffness), higher = stiffer cloth (default: 200.0)
- `--c`: Damping coefficient, higher = less oscillation (default: 0.01)
- `--mass`: Particle mass in kg (default: 0.02)
- `--dt`: Time step in seconds (default: 1/240 ≈ 0.00417s)
- `--steps`: Number of simulation steps (default: 1200)
- `--g`: Gravitational acceleration (default: 9.81 m/s²)
- `--diagonals`: Enable diagonal springs for 8-connectivity
- `--save`: Save trajectory to `.npy` file
- `--animate`: Generate MP4 animation
- `--plot`: Generate trajectory plot

**Example Output**:
- Animation saved to `cloth_animation.mp4`
- Trajectory data saved to `trajectory.npy` (shape: `[frames, particles, 2]`)

### Inverse Parameter Estimation

Given a target trajectory, estimate the physical parameters:

```bash
python main.py inverse \
    --target trajectory.npy \
    --k-init 150 --c-init 0.005 \
    --iterations 200 --lr 0.1 \
    --save-params optimized_params.npy \
    --plot-convergence
```

**Parameters**:
- `--target`: Path to target trajectory file (`.npy`)
- `--k-init`, `--c-init`: Initial guess for parameters
- `--iterations`: Number of optimization iterations (default: 200)
- `--lr`: Learning rate for gradient descent (default: 0.1)
- `--frame-stride`: Compute loss every N frames (default: 8, for efficiency)
- `--grad-clip`: Maximum gradient norm (default: 10.0)
- `--save-params`: Save estimated parameters to file
- `--plot-convergence`: Generate convergence plots

**Example Output**:
- Optimized parameters: `k` and `c` values
- Convergence plots showing loss and parameter evolution
- Parameter history saved to `.npy` file

### Advanced Examples

**Stiffer cloth with diagonal springs**:
```bash
python main.py forward --k 500 --c 0.02 --diagonals --animate
```

**Fine-tune inverse optimization**:
```bash
python main.py inverse \
    --target trajectory.npy \
    --k-init 100 --c-init 1e-3 \
    --iterations 500 --lr 0.05 \
    --frame-stride 4 \
    --plot-convergence
```

## Project Structure

```
.
├── main.py                    # CLI entry point
├── cloth_sim/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Simulation configuration
│   ├── geometry.py           # Grid generation, edge connectivity
│   ├── simulation.py         # ClothSimulator class
│   ├── optimization.py       # InverseSolver class
│   ├── kernels.py            # Warp CUDA kernels
│   └── visualization.py      # Plotting and animation utilities
├── trajectory.npy            # Example trajectory data
└── cloth_animation.mp4       # Example animation output
```

## Key Components

### `ClothSimulator`
Forward simulation engine that steps through time, computing forces and integrating positions/velocities.

### `InverseSolver`
Gradient-based optimizer that:
1. Runs forward simulation with current parameter estimates
2. Computes loss against target trajectory
3. Backpropagates gradients through simulation
4. Updates parameters using gradient descent

### Warp Kernels (`kernels.py`)
GPU-accelerated functions:
- `spring_forces`: Parallel spring force computation
- `damping_forces`: Velocity-based damping
- `integrate`: Euler integration step
- `mse_frame`: Loss computation (differentiable)

## Performance Considerations

- **GPU Acceleration**: Significant speedup on CUDA-capable GPUs (10-100× faster than CPU)
- **Frame Stride**: For inverse problems, computing loss every N frames reduces computation while maintaining accuracy
- **Grid Size**: Larger grids (e.g., 50×50) provide smoother cloth but increase computation time
- **Time Step**: Smaller `dt` improves stability but requires more steps


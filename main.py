#!/usr/bin/env python3
"""
Main entry point for cloth simulation.

Usage:
    python main.py forward --save trajectory.npy --animate
    python main.py inverse --target trajectory.npy --k-init 150 --c-init 0.005
"""

import argparse
import sys

import numpy as np

from cloth_sim import (
    SimConfig,
    ClothSimulator,
    InverseSolver,
    animate_particles,
    plot_trajectories,
)
from cloth_sim.optimization import save_trajectory, load_trajectory


def run_forward(args):
    """Run forward simulation."""
    print("=== Forward Simulation ===")

    # Create config
    config = SimConfig(
        width=args.width,
        height=args.height,
        space_delta=args.space_delta,
        mass=args.mass,
        k=args.k,
        c=args.c,
        dt=args.dt,
        steps=args.steps,
        g=args.g,
        use_diagonals=args.diagonals,
    )

    print(f"Config: {config.width}x{config.height} grid, k={config.k}, c={config.c}")
    print(f"Steps: {config.steps}, dt={config.dt:.6f}")

    # Create simulator
    simulator = ClothSimulator(config)
    print(f"Device: {config.device}")

    # Run simulation
    print("Running simulation...")
    trajectory = simulator.run(record=True)
    print(f"Trajectory shape: {trajectory.shape}")

    # Save if requested
    if args.save:
        save_trajectory(trajectory, args.save)

    # Animate if requested
    if args.animate:
        print("Creating animation...")
        anim = animate_particles(trajectory, save_gif=True)
        print(f"Animation saved to cloth_animation.mp4")

    # Plot trajectories if requested
    if args.plot:
        import matplotlib.pyplot as plt

        fig = plot_trajectories(trajectory)
        plt.savefig(args.plot_path)
        print(f"Trajectory plot saved to {args.plot_path}")

    print("Done!")
    return trajectory


def run_inverse(args):
    """Run inverse optimization."""
    print("=== Inverse Problem (Parameter Estimation) ===")

    # Load target trajectory
    target = load_trajectory(args.target)
    steps = len(target)

    # Create config matching target trajectory
    config = SimConfig(
        width=args.width,
        height=args.height,
        space_delta=args.space_delta,
        mass=args.mass,
        k=args.k_init,  # Will be optimized
        c=args.c_init,  # Will be optimized
        dt=args.dt,
        steps=steps,
        g=args.g,
        use_diagonals=args.diagonals,
    )

    print(f"Config: {config.width}x{config.height} grid")
    print(f"Target trajectory: {target.shape}")
    print(f"Initial guess: k={args.k_init}, c={args.c_init}")

    # Create simulator
    simulator = ClothSimulator(config)
    print(f"Device: {config.device}")

    # Create inverse solver
    solver = InverseSolver(
        simulator=simulator,
        target_trajectory=target,
        k_init=args.k_init,
        c_init=args.c_init,
    )

    # Run optimization
    print(f"\nRunning optimization ({args.iterations} iterations, lr={args.lr})...")
    result = solver.optimize(
        iterations=args.iterations,
        lr=args.lr,
        frame_stride=args.frame_stride,
        grad_clip=args.grad_clip,
        verbose=True,
        print_every=args.print_every,
    )

    # Save results if requested
    if args.save_params:
        np.save(
            args.save_params,
            {
                "k": result.k,
                "c": result.c,
                "loss_history": result.loss_history,
                "k_history": result.k_history,
                "c_history": result.c_history,
            },
        )
        print(f"Parameters saved to {args.save_params}")

    # Plot convergence if requested
    if args.plot_convergence:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(result.loss_history)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Convergence")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(result.k_history)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("k")
        axes[1].set_title("Spring Constant Convergence")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(result.c_history)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("c")
        axes[2].set_title("Damping Coefficient Convergence")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.convergence_path)
        print(f"Convergence plot saved to {args.convergence_path}")

    print("\n=== Results ===")
    print(f"Estimated k: {result.k:.4f}")
    print(f"Estimated c: {result.c:.6e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Cloth simulation with forward and inverse modes"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Forward simulation ---
    fwd_parser = subparsers.add_parser("forward", help="Run forward simulation")

    # Grid parameters
    fwd_parser.add_argument("--width", type=int, default=20, help="Grid width")
    fwd_parser.add_argument("--height", type=int, default=20, help="Grid height")
    fwd_parser.add_argument(
        "--space-delta", type=float, default=0.05, help="Particle spacing"
    )
    fwd_parser.add_argument("--mass", type=float, default=0.02, help="Particle mass")
    fwd_parser.add_argument(
        "--diagonals", action="store_true", help="Use diagonal springs"
    )

    # Physics parameters
    fwd_parser.add_argument("--k", type=float, default=200.0, help="Spring constant")
    fwd_parser.add_argument("--c", type=float, default=0.01, help="Damping coefficient")
    fwd_parser.add_argument("--g", type=float, default=9.81, help="Gravity")

    # Simulation parameters
    fwd_parser.add_argument(
        "--dt", type=float, default=1.0 / 240.0, help="Time step"
    )
    fwd_parser.add_argument("--steps", type=int, default=1200, help="Number of steps")

    # Output options
    fwd_parser.add_argument("--save", type=str, help="Save trajectory to file")
    fwd_parser.add_argument(
        "--animate", action="store_true", help="Create animation GIF"
    )
    fwd_parser.add_argument(
        "--gif-path", type=str, default="cloth_animation.gif", help="GIF output path"
    )
    fwd_parser.add_argument(
        "--plot", action="store_true", help="Plot particle trajectories"
    )
    fwd_parser.add_argument(
        "--plot-path", type=str, default="trajectories.png", help="Plot output path"
    )

    # --- Inverse optimization ---
    inv_parser = subparsers.add_parser("inverse", help="Run inverse optimization")

    # Target
    inv_parser.add_argument(
        "--target", type=str, required=True, help="Target trajectory file"
    )

    # Grid parameters (should match target)
    inv_parser.add_argument("--width", type=int, default=20, help="Grid width")
    inv_parser.add_argument("--height", type=int, default=20, help="Grid height")
    inv_parser.add_argument(
        "--space-delta", type=float, default=0.05, help="Particle spacing"
    )
    inv_parser.add_argument("--mass", type=float, default=0.02, help="Particle mass")
    inv_parser.add_argument(
        "--diagonals", action="store_true", help="Use diagonal springs"
    )

    # Initial guess
    inv_parser.add_argument(
        "--k-init", type=float, default=100.0, help="Initial guess for k"
    )
    inv_parser.add_argument(
        "--c-init", type=float, default=1e-3, help="Initial guess for c"
    )

    # Known physics parameters
    inv_parser.add_argument("--g", type=float, default=9.81, help="Gravity")
    inv_parser.add_argument(
        "--dt", type=float, default=1.0 / 240.0, help="Time step"
    )

    # Optimization parameters
    inv_parser.add_argument(
        "--iterations", type=int, default=200, help="Number of optimization iterations"
    )
    inv_parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    inv_parser.add_argument(
        "--frame-stride", type=int, default=8, help="Loss computation stride"
    )
    inv_parser.add_argument(
        "--grad-clip", type=float, default=10.0, help="Gradient clipping threshold"
    )
    inv_parser.add_argument(
        "--print-every", type=int, default=10, help="Print progress every N iterations"
    )

    # Output options
    inv_parser.add_argument(
        "--save-params", type=str, help="Save estimated parameters to file"
    )
    inv_parser.add_argument(
        "--plot-convergence", action="store_true", help="Plot convergence graphs"
    )
    inv_parser.add_argument(
        "--convergence-path",
        type=str,
        default="convergence.png",
        help="Convergence plot path",
    )

    args = parser.parse_args()

    if args.command == "forward":
        run_forward(args)
    elif args.command == "inverse":
        run_inverse(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


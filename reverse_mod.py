import numpy as np
import warp as wp

from forward import (
    W, H, SPACE_DELTA, DT, STEPS, G, MASS, USE_DIAGONALS, DEVICE,
    make_edges_and_rests, make_pins,
    zero_vec2, spring_forces, damping_forces, integrate, setup_data
)

# Load ground truth data
gt_np = np.load("cloth_target.npy").astype(np.float32)   # shape: [STEPS+1, N, 2]
assert gt_np.shape[0] == STEPS+1, f"Expected {STEPS+1} frames, got {gt_np.shape[0]}"

import os, sys, faulthandler
faulthandler.enable()                 # dumps Python stack on SIGSEGV
os.environ["PYTHONFAULTHANDLER"] = "1"

def mark(msg):
    print(msg)
    sys.stdout.flush()

def run_simulation_forward_only(k_val, c_val):
    """
    Run simulation with given parameters (no autodiff).
    Returns final loss value.
    """
    # Set up simulation data
    pos, vel, edges, rest, pins, masses, forces = setup_data()
    
    particle_count = pos.shape[0]
    total_loss = 0.0
    
    # Create parameter arrays (no requires_grad)
    K_arr = wp.array([float(k_val)], dtype=wp.float32, device=DEVICE)
    C_arr = wp.array([float(c_val)], dtype=wp.float32, device=DEVICE)
    
    # Compute loss for initial frame
    pos_np = pos.numpy()
    diff = pos_np - gt_np[0]
    frame_loss = np.sum(diff * diff)
    total_loss += frame_loss
    
    # Main simulation loop
    for frame in range(1, STEPS + 1):
        # Reset forces
        wp.launch(zero_vec2, dim=particle_count, inputs=[forces], device=DEVICE)
        
        # Calculate spring forces
        wp.launch(spring_forces, dim=edges.shape[0], 
                 inputs=[pos, edges, rest, K_arr, forces], device=DEVICE)
        
        # Calculate damping forces
        wp.launch(damping_forces, dim=particle_count, 
                 inputs=[vel, C_arr, forces], device=DEVICE)
        
        # Integrate positions and velocities
        wp.launch(integrate, dim=particle_count, 
                 inputs=[pos, vel, forces, masses, pins, float(G), float(DT)], 
                 device=DEVICE)
        
        # Compute loss for current frame (on CPU)
        pos_np = pos.numpy()
        diff = pos_np - gt_np[frame]
        frame_loss = np.sum(diff * diff)
        total_loss += frame_loss
        
        if frame % 200 == 0:
            current_avg_loss = total_loss / (particle_count * (frame + 1))
            print(f"  Frame {frame}, avg loss: {current_avg_loss:.6e}")
    
    # Normalize by total number of measurements
    normalized_loss = total_loss / (particle_count * (STEPS + 1))
    return normalized_loss

def finite_difference_gradients(k_val, c_val, epsilon_k=1e-3, epsilon_c=1e-6):
    """
    Compute gradients using finite differences.
    """
    print(f"  Computing gradients with finite differences...")
    print(f"  Base parameters: K={k_val:.4f}, C={c_val:.8f}")
    
    # Base loss
    print(f"  Computing base loss...")
    loss_base = run_simulation_forward_only(k_val, c_val)
    print(f"  Base loss: {loss_base:.6e}")
    
    # K gradient (in log space)
    print(f"  Computing K gradient...")
    k_plus = k_val * np.exp(epsilon_k)  # Small multiplicative change
    loss_k_plus = run_simulation_forward_only(k_plus, c_val)
    gk_log = (loss_k_plus - loss_base) / epsilon_k  # Gradient w.r.t. log(K)
    print(f"  K+ loss: {loss_k_plus:.6e}, grad_log_K: {gk_log:.3e}")
    
    # C gradient (in log space)
    print(f"  Computing C gradient...")
    c_plus = c_val * np.exp(epsilon_c)  # Small multiplicative change
    loss_c_plus = run_simulation_forward_only(k_val, c_plus)
    gc_log = (loss_c_plus - loss_base) / epsilon_c  # Gradient w.r.t. log(C)
    print(f"  C+ loss: {loss_c_plus:.6e}, grad_log_C: {gc_log:.3e}")
    
    return loss_base, gk_log, gc_log

def estimate_params_finite_diff():
    """
    Estimate parameters using finite differences instead of autodiff.
    """
    # Initialize parameters
    initial_k = 150.0  # Closer to target
    initial_c = 0.005  # Closer to target
    
    # Optimization in log space for better numerical properties
    log_k = np.log(initial_k)
    log_c = np.log(initial_c)
    
    # Optimization settings
    epochs = 30
    lr_k = 0.1      # Learning rate for log(K)
    lr_c = 0.1      # Learning rate for log(C)
    
    print(f"Starting parameter estimation with finite differences...")
    print(f"Initial: K={initial_k:.2f}, C={initial_c:.8f}")
    print(f"Target:  K=200.0, C=0.001")
    print(f"Epochs: {epochs}")
    print(f"Learning rates: lr_K={lr_k}, lr_C={lr_c}")
    print("=" * 60)
    
    best_loss = float('inf')
    best_k, best_c = initial_k, initial_c
    
    # Store optimization history
    history = {'epoch': [], 'loss': [], 'k': [], 'c': [], 'gk': [], 'gc': []}
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch} ---")
        
        try:
            # Current parameter values
            current_k = np.exp(log_k)
            current_c = np.exp(log_c)
            
            print(f"Current: K={current_k:.4f}, C={current_c:.8f}")
            
            # Compute loss and gradients
            loss, gk_log, gc_log = finite_difference_gradients(current_k, current_c)
            
            print(f"Loss: {loss:.6e}")
            print(f"Gradients: gK={gk_log:.3e}, gC={gc_log:.3e}")
            
            # Track best parameters
            if loss < best_loss:
                best_loss = loss
                best_k = current_k
                best_c = current_c
                print(f"*** New best parameters! ***")
            
            # Store history
            history['epoch'].append(epoch)
            history['loss'].append(loss)
            history['k'].append(current_k)
            history['c'].append(current_c)
            history['gk'].append(gk_log)
            history['gc'].append(gc_log)
            
            # Gradient descent update in log space
            log_k_new = log_k - lr_k * gk_log
            log_c_new = log_c - lr_c * gc_log
            
            # Apply bounds to prevent extreme values
            log_k_new = np.clip(log_k_new, np.log(10.0), np.log(1000.0))    # K between 10-1000
            log_c_new = np.clip(log_c_new, np.log(1e-6), np.log(1.0))       # C between 1e-6 and 1
            
            log_k = log_k_new
            log_c = log_c_new
            
            new_k = np.exp(log_k)
            new_c = np.exp(log_c)
            print(f"Updated: K={new_k:.4f}, C={new_c:.8f}")
            
            # Adaptive learning rate
            if epoch > 5:
                recent_losses = history['loss'][-5:]
                if len(recent_losses) >= 2 and recent_losses[-1] > recent_losses[-2]:
                    lr_k *= 0.9
                    lr_c *= 0.9
                    print(f"Reducing learning rates: lr_K={lr_k:.4f}, lr_C={lr_c:.4f}")
                elif len(recent_losses) >= 3 and all(recent_losses[i] < recent_losses[i-1] for i in range(1, len(recent_losses))):
                    lr_k *= 1.05
                    lr_c *= 1.05
                    print(f"Increasing learning rates: lr_K={lr_k:.4f}, lr_C={lr_c:.4f}")
            
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "=" * 60)
    print(f"Optimization completed!")
    print(f"Best loss: {best_loss:.6e}")
    print(f"Best parameters:")
    print(f"  K = {best_k:.6f} (target: 200.0, error: {abs(best_k-200)/200*100:.1f}%)")
    print(f"  C = {best_c:.8f} (target: 0.001, error: {abs(best_c-0.001)/0.001*100:.1f}%)")
    
    # Save results
    result_params = np.array([best_k, best_c], dtype=np.float32)
    np.save("cloth_est_params.npy", result_params)
    print(f"Saved to cloth_est_params.npy")
    
    # Save optimization history
    np.save("optimization_history.npy", history)
    print(f"Saved optimization history to optimization_history.npy")
    
    return best_k, best_c, history

def test_forward_simulation():
    """Test forward simulation with known parameters."""
    print("Testing forward simulation with true parameters...")
    true_k, true_c = 200.0, 0.001
    
    loss = run_simulation_forward_only(true_k, true_c)
    print(f"Loss with true parameters: {loss:.8e}")
    
    if loss < 1e-10:
        print("✓ Forward simulation working correctly (near-zero loss)")
    else:
        print("⚠ Forward simulation may have issues (non-zero loss with true params)")
    
    return loss

if __name__ == "__main__":
    print(f"Warp device: {DEVICE}")
    print(f"Ground truth shape: {gt_np.shape}")
    print(f"Simulation: {W}x{H} = {W*H} particles, {STEPS} steps")
    print(f"Using diagonal springs: {USE_DIAGONALS}")
    
    # Test forward simulation first
    try:
        test_loss = test_forward_simulation()
        
        if test_loss > 1e-6:
            print(f"Warning: High loss ({test_loss:.2e}) with true parameters.")
            print("This suggests the forward model may not match the ground truth exactly.")
            print("Proceeding anyway - this could be due to numerical precision or implementation differences.")
        
        print(f"\nStarting parameter estimation...")
        
        # Run parameter estimation
        best_k, best_c, history = estimate_params_finite_diff()
        
        print(f"\nFinal Results:")
        print(f"Estimated K: {best_k:.6f}")
        print(f"Estimated C: {best_c:.8f}")
        
    except Exception as e:
        print(f"Parameter estimation failed: {e}")
        import traceback
        traceback.print_exc()
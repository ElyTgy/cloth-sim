import numpy as np
import warp as wp



from forward import (
    W, H, SPACE_DELTA, DT, STEPS, G, MASS, USE_DIAGONALS, DEVICE,
    make_edges_and_rests, make_pins,
    zero_vec2, spring_forces, damping_forces, integrate, setup_data
)

gt_np = np.load("cloth_target.npy").astype(np.float32)   # shape: [STEPS, N, 2]
assert gt_np.shape[0] == STEPS+1, f"Expected {STEPS+1} frames, got {gt_np.shape[0]}"
N = W * H

gt_all = wp.array(gt_np, dtype=wp.vec2, device=DEVICE)
pos0, velocity0, edges, rest, pins, masses, forces = setup_data()

# mask: 1 for free verts, 0 for pinned (as float for loss math)
#free_mask_np = (1 - np.array(pins)).astype(np.float32)
#mask_d = wp.array(free_mask_np, dtype=float, device=DEVICE)

log_k = wp.array([np.log(100.0)], dtype=wp.float32, requires_grad=True, device=DEVICE)
log_c = wp.array([np.log(0.01)], dtype=wp.float32, requires_grad=True, device=DEVICE)

@wp.kernel
def mse_kernel(pos: wp.array(dtype=wp.vec2),
               target: wp.array(dtype=wp.vec2),
               out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    d = pos[i] - target[i]          # both real device arrays
    wp.atomic_add(out, 0, wp.dot(d, d))


def mse(pos: wp.array(dtype=wp.vec2), target: wp.array(dtype=wp.vec2)):
    sum = wp.zeros(1, dtype=wp.float32, device=DEVICE, requires_grad=True)
    wp.launch(mse_kernel, dim=pos.shape[0], inputs=[pos, target, sum], device=DEVICE)
    return sum


def rollout_and_loss(gt_np: np.ndarray, log_k: wp.array, log_c: wp.array):
    #clean up later so everything except pos and vel get set up outside of this func
    pos, vel, edges, rest, pins, masses, forces = setup_data()

    loss = wp.zeros(1, dtype=wp.float32, device=DEVICE)
    K_arr = wp.exp(log_k.numpy()[0])
    C_arr = wp.exp(log_c.numpy()[0])

    for frame in range(gt_np.shape[0]):
        wp.launch(zero_vec2, dim=W*H, inputs=[forces])
        # Calculate spring forces
        wp.launch(spring_forces, dim=edges.shape[0], 
                 inputs=[pos, edges, rest, wp.array([float(K_arr)], dtype=wp.float32), forces], device=DEVICE)
        
        # Calculate damping forces
        wp.launch(damping_forces, dim=W*H, 
                 inputs=[vel, wp.array([float(C_arr)], dtype=wp.float32), forces], device=DEVICE)
        
        # Integrate positions and velocities
        wp.launch(integrate, dim=W*H, 
                 inputs=[pos, vel, forces, masses, pins, float(G), float(DT)], 
                 device=DEVICE)

        target_frame = wp.array(gt_np[frame], dtype=wp.vec2, device=DEVICE)
        loss = mse(pos, target_frame)
        loss = loss/float(pos.shape[0])

    return loss


def estimate_params():
    global log_k, log_c #if not included, python assumes they are local variables and that they havent been assigned yet
    epochs: int = 100
    lr_k: float = 1e-3
    lr_c: float = 1e-3

    for epoch in range(epochs):
        print(epoch)
        with wp.Tape() as tape:
            print("calling rollout and loss")
            loss = rollout_and_loss(gt_np, log_k, log_c)
            print(loss)
        tape.backward(loss)

        # read gradients to host
        gk = tape.gradients[log_k].numpy()[0]
        gc = tape.gradients[log_c].numpy()[0]

        # simple gradient descent step on host
        new_log_k = log_k.numpy()[0] - lr_k * gk
        new_log_c = log_c.numpy()[0] - lr_c * gc

        # re-wrap as fresh autodiff tensors for next epoch
        log_k = wp.array([float(new_log_k)], dtype=wp.float32, device=DEVICE, requires_grad=True)
        log_c = wp.array([float(new_log_c)], dtype=wp.float32, device=DEVICE, requires_grad=True)


        if (epoch % 5) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:03d} | loss={loss.numpy()[0]:.6e} | K={np.exp(new_log_k):.4f} | C={np.exp(new_log_c):.6f} | gK={gk:.3e} gC={gc:.3e}")


    K_hat = float(np.exp(log_k.numpy()[0]))
    C_hat = float(np.exp(log_c.numpy()[0]))
    np.save("cloth_est_params.npy", np.array([K_hat, C_hat], dtype=np.float32))
    print(f"Saved estimated params to cloth_est_params.npy -> K={K_hat:.6f}, C={C_hat:.8f}")
    return K_hat, C_hat


estimate_params()
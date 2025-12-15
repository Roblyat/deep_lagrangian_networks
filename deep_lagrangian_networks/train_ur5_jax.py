import argparse
import time
import functools
import dill as pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk

import matplotlib as mp
try:
    mp.use("Qt5Agg")
    mp.rc('text', usetex=False)
    import matplotlib.pyplot as plt
except:
    plt = None

import deep_lagrangian_networks.jax_DeLaN_model as delan
from deep_lagrangian_networks.replay_memory import ReplayMemory
from deep_lagrangian_networks.utils import init_env, activations, load_npz_trajectory_dataset

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[1], help="Use CUDA (via torch availability check).")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0], help="CUDA id (torch side).")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[4], help="Random seed.")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[0], help="Render plots.")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0], help="Load model.")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1], help="Save model.")

    # UR5 dataset path (NPZ from preprocess)
    parser.add_argument("--npz", type=str, required=False,
                        default="/workspace/shared/data/processed/delan_ur5_dataset.npz",
                        help="Path to delan_ur5_dataset.npz")

    # structured vs black_box (same as jax_example_DeLaN.py)
    parser.add_argument("-t", nargs=1, type=str, required=False, default=['structured'],
                        help="Lagrangian Type: structured|black_box")

    args = parser.parse_args()
    seed, cuda, render, load_model, save_model = init_env(args)
    rng_key = jax.random.PRNGKey(seed)

    model_choice = str(args.t[0])
    if model_choice == "structured":
        lagrangian_type = delan.structured_lagrangian_fn
    elif model_choice == "black_box":
        lagrangian_type = delan.blackbox_lagrangian_fn
    else:
        raise ValueError("Unknown -t. Use structured or black_box.")

    # Hyperparameters (copy from jax_example_DeLaN.py; adjust later)
    hyper = {
        'dataset': 'ur5_npz',
        'n_width': 64,
        'n_depth': 2,
        'n_minibatch': 512,
        'diagonal_epsilon': 0.1,
        'diagonal_shift': 2.0,
        'activation': 'tanh',
        'learning_rate': 1.e-04,
        'weight_decay': 1.e-5,
        'max_epoch': int(2.0 * 1e3),  # start smaller; increase later
        'lagrangian_type': lagrangian_type,
    }

    model_id = "structured" if hyper['lagrangian_type'].__name__ == 'structured_lagrangian_fn' else "black_box"

    # Optional load
    params = None
    if load_model:
        with open(f"data/delan_models/delan_{model_id}_{hyper['dataset']}_seed_{seed}.jax", 'rb') as f:
            saved = pickle.load(f)
        hyper = saved["hyper"]
        params = saved["params"]

    # Load NPZ dataset (EFFORT treated as TAU)
    train_data, test_data, divider, dt = load_npz_trajectory_dataset(args.npz)
    train_labels, train_qp, train_qv, train_qa, train_tau = train_data
    test_labels,  test_qp,  test_qv,  test_qa,  test_tau  = test_data

    n_dof = train_qp.shape[-1]

    print("\n\n################################################")
    print("UR5 Dataset:")
    print(f"  npz = {args.npz}")
    print(f"   dt ≈ {dt}")
    print(f"  dof = {n_dof}")
    print(f"  Train trajectories = {len(train_labels)}")
    print(f"  Test trajectories  = {len(test_labels)}")
    print(f"  Train samples = {train_qp.shape[0]}")
    print(f"  Test samples  = {test_qp.shape[0]}")
    print("################################################\n")

    # Replay memory (same pattern as jax_example_DeLaN.py)
    mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
    mem = ReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim)
    mem.add_samples([train_qp, train_qv, train_qa, train_tau])

    # Build network
    lagrangian_fn = hk.transform(functools.partial(
        hyper['lagrangian_type'],
        n_dof=n_dof,
        shape=(hyper['n_width'],) * hyper['n_depth'],
        activation=activations[hyper['activation']],
        epsilon=hyper['diagonal_epsilon'],
        shift=hyper['diagonal_shift'],
    ))

    q, qd, qdd, tau = [jnp.array(x) for x in next(iter(mem))]
    rng_key, init_key = jax.random.split(rng_key)

    if params is None:
        params = lagrangian_fn.init(init_key, q[0], qd[0])

    lagrangian = lagrangian_fn.apply
    delan_model = jax.jit(functools.partial(delan.dynamics_model, lagrangian=lagrangian, n_dof=n_dof))
    _ = delan_model(params, None, q[:1], qd[:1], qdd[:1], tau[:1])

    # Optimizer + loss (per-joint normalization is already in this pattern)
    optimizer = optax.adamw(learning_rate=hyper['learning_rate'], weight_decay=hyper['weight_decay'])
    opt_state = optimizer.init(params)

    loss_fn = functools.partial(
        delan.inverse_loss_fn,
        lagrangian=lagrangian,
        n_dof=n_dof,
        norm_tau=jnp.var(train_tau, axis=0),
        norm_qdd=jnp.var(train_qa, axis=0),
    )

    def update_fn(params, opt_state, q, qd, qdd, tau):
        (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, qd, qdd, tau)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, logs

    update_fn = jax.jit(update_fn)
    _, _, logs0 = update_fn(params, opt_state, q[:1], qd[:1], qdd[:1], tau[:1])

    print("################################################")
    print("Training DeLaN (UR5):\n")

    t0_start = time.perf_counter()
    epoch_i = 0
    while epoch_i < hyper['max_epoch'] and not load_model:
        n_batches = 0
        logs = jax.tree.map(lambda x: x * 0.0, logs0)

        for data_batch in mem:
            q, qd, qdd, tau = [jnp.array(x) for x in data_batch]
            params, opt_state, batch_logs = update_fn(params, opt_state, q, qd, qdd, tau)
            n_batches += 1
            logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)

        epoch_i += 1
        logs = jax.tree.map(lambda x: x / n_batches, logs)

        if epoch_i == 1 or np.mod(epoch_i, 50) == 0:
            print(f"Epoch {epoch_i:05d}: "
                  f"Time={time.perf_counter()-t0_start:6.1f}s, "
                  f"Loss={float(logs['loss']):.2e}, "
                  f"Inv={float(logs['inverse_mean']):.2e}, "
                  f"For={float(logs['forward_mean']):.2e}, "
                  f"Power={float(logs['energy_mean']):.2e}")

    if save_model:
        os.makedirs("data/delan_models", exist_ok=True)
        with open(f"data/delan_models/delan_{model_id}_{hyper['dataset']}_seed_{seed}.jax", "wb") as f:
            pickle.dump({"epoch": epoch_i, "hyper": hyper, "params": params, "seed": seed}, f)

    print("\n################################################")
    print("Evaluating DeLaN (UR5):")

    q = jnp.array(test_qp)
    qd = jnp.array(test_qv)
    qdd = jnp.array(test_qa)

    t0_eval = time.perf_counter()
    pred_tau = delan_model(params, None, q, qd, qdd, 0.0 * q)[1]
    t_eval = (time.perf_counter() - t0_eval) / float(q.shape[0])

    err_tau = float((1.0 / q.shape[0]) * jnp.sum((pred_tau - jnp.array(test_tau)) ** 2))
    print(f"Torque MSE = {err_tau:.3e}")
    print(f"Comp Time per Sample = {t_eval:.3e}s / {1./t_eval:.1f}Hz")

    if render and plt is not None:
        # Plot only torque, first 6 joints
        pred_tau_np = np.array(pred_tau)
        fig = plt.figure(figsize=(14, 8), dpi=100)
        fig.canvas.manager.set_window_title(f'UR5 DeLaN Torque | Seed={seed}')
        for j in range(min(n_dof, 6)):
            ax = fig.add_subplot(3, 2, j+1)
            ax.set_title(f"Joint {j}")
            ax.plot(test_tau[:, j], label="GT", linewidth=1.0)
            ax.plot(pred_tau_np[:, j], label="DeLaN", linewidth=1.0, alpha=0.85)
            ax.grid(True, alpha=0.2)
            if j == 0:
                ax.legend()
        plt.tight_layout()
        plt.show()

    print("\n################################################\n")
import argparse
import time
import functools
import dill as pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import json
from datetime import datetime

# Matplotlib must be configured AFTER args are parsed (render vs headless).
plt = None

def _init_matplotlib(render: bool):
    """Return pyplot (plt) or None. Uses Qt5Agg only when render=True, else Agg."""
    try:
        import matplotlib as mp
        backend = "Qt5Agg" if render else "Agg"
        mp.use(backend, force=True)
        import matplotlib.pyplot as _plt
        return _plt
    except Exception as e:
        print(f"[warn] Matplotlib disabled: {e}")
        return None

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
                        default="/workspace/shared/data/preprocessed/delan_ur5_dataset.npz",
                        help="Path to delan_ur5_dataset.npz")

    # structured vs black_box (same as jax_example_DeLaN.py)
    parser.add_argument("-t", nargs=1, type=str, required=False, default=['structured'],
                        help="Lagrangian Type: structured|black_box")

    parser.add_argument("--save_path", type=str, default="/workspace/shared/models/delan/delan_ur5.jax")

    # --- NEW: hyperparameter preset + overrides (defaults keep current behavior) ---
    parser.add_argument("--hp_preset", type=str, default="default",
                        choices=["default", "fast_debug", "long_train"],
                        help="Hyperparameter preset (UI dropdown).")

    parser.add_argument("--n_width", type=int, default=None)
    parser.add_argument("--n_depth", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)          # n_minibatch
    parser.add_argument("--lr", type=float, default=None)           # learning_rate
    parser.add_argument("--wd", type=float, default=None)           # weight_decay
    parser.add_argument("--epochs", type=int, default=None)         # max_epoch
    parser.add_argument("--diag_eps", type=float, default=None)     # diagonal_epsilon
    parser.add_argument("--diag_shift", type=float, default=None)   # diagonal_shift
    parser.add_argument("--activation", type=str, default=None,
                        choices=["tanh", "relu", "softplus", "gelu", "swish"])
    
    parser.add_argument("--eval_every", type=int, default=200,
                    help="Evaluate on test split every N epochs (for elbow plot). 0 disables periodic eval.")
    parser.add_argument("--eval_n", type=int, default=0,
                        help="If >0, evaluate only on first eval_n test samples for speed. 0 = full test set.")
    
    args = parser.parse_args()

    render_flag = bool(args.r[0])   # keep your existing render handling
    plt = _init_matplotlib(render_flag)

    # --- NEW: derive per-model output directory from save_path ---
    save_path = args.save_path
    model_stem = os.path.splitext(os.path.basename(save_path))[0]
    base_save_dir = os.path.dirname(save_path)

    run_name = model_stem  # UI-aligned name (derived from --save_path)

    # If save_path is already inside a folder named like the run (model_stem),
    # don't create model_stem/model_stem nesting.
    if os.path.basename(base_save_dir) == model_stem:
        model_dir = base_save_dir
        ckpt_path = save_path
    else:
        model_dir = os.path.join(base_save_dir, model_stem)
        ckpt_path = os.path.join(model_dir, os.path.basename(save_path))

    os.makedirs(model_dir, exist_ok=True)

    seed, cuda, render, load_model, save_model = init_env(args)
    rng_key = jax.random.PRNGKey(seed)

    model_choice = str(args.t[0])
    if model_choice == "structured":
        lagrangian_type = delan.structured_lagrangian_fn
    elif model_choice == "black_box":
        lagrangian_type = delan.blackbox_lagrangian_fn
    else:
        raise ValueError("Unknown -t. Use structured or black_box.")

    # Hyperparameters (defaults = jax_example.py)
    hyper = {
        'dataset': run_name,
        'n_width': 64,
        'n_depth': 2,
        'n_minibatch': 512,
        'diagonal_epsilon': 0.1,
        'diagonal_shift': 2.0,
        'activation': 'tanh',
        'learning_rate': 1.e-04,
        'weight_decay': 1.e-5,
        'max_epoch': int(2.0 * 1e3),
        'lagrangian_type': lagrangian_type,
    }

    # --- NEW: presets from UI ---
    PRESETS = {
        "default": {},
        "fast_debug": {
            "max_epoch": 300,
            "n_minibatch": 256,
            "n_width": 64,
            "n_depth": 2,
            "learning_rate": 3e-4,
        },
        "long_train": {
            "max_epoch": 8000,
            "n_minibatch": 512,
            "n_width": 128,
            "n_depth": 3,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
        },
    }
    hyper.update(PRESETS.get(args.hp_preset, {}))

    if args.activation is not None: hyper["activation"] = args.activation

    # JSON-safe hyper (lagrangian_type is a function -> store its name)
    hyper_json = dict(hyper)
    if "lagrangian_type" in hyper_json and hasattr(hyper_json["lagrangian_type"], "__name__"):
        hyper_json["lagrangian_type"] = hyper_json["lagrangian_type"].__name__
    else:
        hyper_json["lagrangian_type"] = str(hyper_json.get("lagrangian_type"))

    print(f"Final hyper: {hyper}")

    # --- NEW: explicit overrides (if UI sets them) ---
    if args.n_width is not None: hyper["n_width"] = args.n_width
    if args.n_depth is not None: hyper["n_depth"] = args.n_depth
    if args.batch is not None: hyper["n_minibatch"] = args.batch
    if args.lr is not None: hyper["learning_rate"] = args.lr
    if args.wd is not None: hyper["weight_decay"] = args.wd
    if args.epochs is not None: hyper["max_epoch"] = int(args.epochs)
    if args.diag_eps is not None: hyper["diagonal_epsilon"] = args.diag_eps
    if args.diag_shift is not None: hyper["diagonal_shift"] = args.diag_shift
    if args.activation is not None: hyper["activation"] = args.activation

    print(f"Final hyper: {hyper}")

    model_id = "structured" if hyper['lagrangian_type'].__name__ == 'structured_lagrangian_fn' else "black_box"

    # Optional load
    params = None
    if load_model:
        load_path = ckpt_path  # load from the same per-run folder
        with open(load_path, "rb") as f:
            saved = pickle.load(f)
        hyper = saved.get("hyper", hyper)
        params = saved["params"]
        print(f"Loaded DeLaN checkpoint: {load_path}")

    print("\n\n################################################")
    print(f"DeLaN run: {run_name}")
    print(f"  model_dir = {model_dir}")
    print(f"  ckpt_path = {ckpt_path}")
    print(f"  type = {model_choice}")
    print(f"  hp_preset = {args.hp_preset}")
    print("################################################")

    # ---- NEW: metrics JSON scaffold (written at end; updated along the way) ----
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "model_type": model_choice,
        "hp_preset": args.hp_preset,
        "npz": args.npz,
        "model_dir": model_dir,
        "ckpt_path": ckpt_path,
        "seed": int(seed),
        "dt": float(dt) if "dt" in locals() else None,
        "n_dof": None,  # fill after dataset load
        "hyper": hyper_json,
        "args": {
            "render": int(args.r[0]),
            "eval_every": int(args.eval_every),
            "eval_n": int(args.eval_n),
        },
        "dataset": {},
        "train": {},
        "eval_test": {},
        "artifacts": {},
    }

    # Load NPZ dataset (EFFORT treated as TAU)
    train_data, test_data, divider, dt = load_npz_trajectory_dataset(args.npz)
    train_labels, train_qp, train_qv, train_qa, train_tau = train_data
    test_labels,  test_qp,  test_qv,  test_qa,  test_tau  = test_data

    n_dof = train_qp.shape[-1]

    print("\n\n################################################")
    print("Dataset:")
    print(f"  npz = {args.npz}")
    print(f"   dt ≈ {dt}")
    print(f"  dof = {n_dof}")
    print(f"  Train trajectories = {len(train_labels)}")
    print(f"  Test trajectories  = {len(test_labels)}")
    print(f"  Train samples = {train_qp.shape[0]}")
    print(f"  Test samples  = {test_qp.shape[0]}")
    print("################################################\n")

    metrics["n_dof"] = int(n_dof)
    metrics["dt"] = float(dt)
    metrics["dataset"] = {
        "train_trajectories": int(len(train_labels)),
        "test_trajectories": int(len(test_labels)),
        "train_samples": int(train_qp.shape[0]),
        "test_samples": int(test_qp.shape[0]),
    }

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
    print(f"Training DeLaN | run={run_name} | type={model_choice} | dof={n_dof}")
    print("################################################")

    # --- NEW: training history for plotting ---
    hist_epoch = []
    hist_loss = []
    hist_inv = []
    hist_for = []
    hist_energy = []
    hist_time = []  # seconds since training start

    # --- NEW: test elbow history (sampled every eval_every epochs) ---
    hist_test_epoch = []
    hist_test_mse = []

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
            
            # --- NEW: store history points (same cadence as prints) ---
            hist_epoch.append(epoch_i)
            hist_time.append(time.perf_counter() - t0_start)
            hist_loss.append(float(logs['loss']))
            hist_inv.append(float(logs['inverse_mean']))
            hist_for.append(float(logs['forward_mean']))
            hist_energy.append(float(logs['energy_mean']))

        # --- NEW: periodic test evaluation for elbow curve ---
        if args.eval_every > 0 and (epoch_i == 1 or (epoch_i % args.eval_every) == 0):
            # choose subset for speed if requested
            q_eval  = test_qp
            qd_eval = test_qv
            qdd_eval = test_qa
            tau_eval = test_tau

            if args.eval_n and args.eval_n > 0:
                n = min(int(args.eval_n), q_eval.shape[0])
                q_eval  = q_eval[:n]
                qd_eval = qd_eval[:n]
                qdd_eval = qdd_eval[:n]
                tau_eval = tau_eval[:n]

            qj   = jnp.array(q_eval)
            qdj  = jnp.array(qd_eval)
            qddj = jnp.array(qdd_eval)
            tauj = jnp.array(tau_eval)

            pred_tau_eval = delan_model(params, None, qj, qdj, qddj, 0.0 * qj)[1]
            test_mse = float((1.0 / qj.shape[0]) * jnp.sum((pred_tau_eval - tauj) ** 2))

            hist_test_epoch.append(epoch_i)
            hist_test_mse.append(test_mse)

            print(f"  [eval] test_mse={test_mse:.3e}  (n={qj.shape[0]})")

    if save_model:
        with open(ckpt_path, "wb") as f:
            pickle.dump({"epoch": epoch_i, "hyper": hyper, "params": params, "seed": seed}, f)
        print(f"Saved DeLaN checkpoint: {ckpt_path}")

    # --- NEW: save training curves (loss + components) ---
    if plt is not None and len(hist_epoch) > 0:
        # 1) Loss curve
        fig = plt.figure(figsize=(8, 4), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(hist_epoch, hist_loss)
        ax.set_title(f"{run_name} | Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        loss_path = os.path.join(model_dir, f"{run_name}__loss_curve.png")
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        print(f"Saved loss curve: {loss_path}")

        # 2) Components: inverse / forward / energy
        fig = plt.figure(figsize=(8, 4), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(hist_epoch, hist_inv, label="inverse_mean")
        ax.plot(hist_epoch, hist_for, label="forward_mean")
        ax.plot(hist_epoch, hist_energy, label="energy_mean")
        ax.set_title(f"{run_name} | Loss Components")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()
        comp_path = os.path.join(model_dir, f"{run_name}__loss_components.png")
        fig.savefig(comp_path, dpi=150)
        plt.close(fig)
        print(f"Saved loss components: {comp_path}")

        # 3) Optional: CSV dump for later analysis
        csv_path = os.path.join(model_dir, f"{run_name}__train_history.csv")
        with open(csv_path, "w") as f:
            f.write("epoch,time_s,loss,inverse_mean,forward_mean,energy_mean\n")
            for e, ts, lo, inv, fo, en in zip(hist_epoch, hist_time, hist_loss, hist_inv, hist_for, hist_energy):
                f.write(f"{e},{ts},{lo},{inv},{fo},{en}\n")
        print(f"Saved training history: {csv_path}")

        metrics["artifacts"]["train_history_csv"] = csv_path
        metrics["artifacts"]["loss_curve_png"] = loss_path
        metrics["artifacts"]["loss_components_png"] = comp_path

    # --- NEW: elbow plot (train loss + test MSE over epochs) ---
    if plt is not None and len(hist_epoch) > 0 and len(hist_test_epoch) > 0:
        fig = plt.figure(figsize=(8, 4), dpi=120)
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(hist_epoch, hist_loss, label="train_loss", color="C0")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train loss")
        ax1.grid(True, alpha=0.25)

        # second axis for test MSE
        ax2 = ax1.twinx()
        ax2.plot(hist_test_epoch, hist_test_mse, label="test_mse", color="C1")
        ax2.set_ylabel("Test torque MSE")

        # combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        ax1.set_title(f"{run_name} | Elbow (train loss vs test MSE)")

        plt.tight_layout()
        elbow_path = os.path.join(model_dir, f"{run_name}__elbow_train_vs_test.png")
        fig.savefig(elbow_path, dpi=150)
        plt.close(fig)
        print(f"Saved elbow plot: {elbow_path}")

        metrics["artifacts"]["elbow_png"] = elbow_path

    metrics["train"]["epochs_ran"] = int(epoch_i)
    metrics["train"]["history_points"] = int(len(hist_epoch))
    metrics["train"]["elbow_points"] = int(len(hist_test_epoch))


    print("\n################################################")
    print(f"Evaluating DeLaN | run={run_name}")

    q = jnp.array(test_qp)
    qd = jnp.array(test_qv)
    qdd = jnp.array(test_qa)

    t0_eval = time.perf_counter()

    pred_tau = delan_model(params, None, q, qd, qdd, 0.0 * q)[1]
    t_eval = (time.perf_counter() - t0_eval) / float(q.shape[0])

    tau_true = jnp.array(test_tau)
    err = pred_tau - tau_true

    # global MSE over all samples + joints
    err_tau = float(jnp.mean(err ** 2))
    err_tau_rmse = float(np.sqrt(err_tau))

    # per-joint MSE / RMSE
    err_tau_j = np.array(jnp.mean(err ** 2, axis=0))          # shape: (n_dof,)
    err_tau_rmse_j = np.sqrt(err_tau_j)

    print(f"Torque MSE  = {err_tau:.3e}")
    print(f"Torque RMSE = {err_tau_rmse:.3e}")
    print("Per-joint MSE :", " ".join([f"{x:.3e}" for x in err_tau_j]))
    print("Per-joint RMSE:", " ".join([f"{x:.3e}" for x in err_tau_rmse_j]))
    print(f"Comp Time per Sample = {t_eval:.3e}s / {1./t_eval:.1f}Hz")

    # ----------------------------
    # Save eval metrics (ALWAYS)
    # ----------------------------
    metrics["eval_test"] = {
        "torque_mse": float(err_tau),
        "torque_rmse": float(err_tau_rmse),
        "torque_mse_per_joint": [float(x) for x in err_tau_j],
        "torque_rmse_per_joint": [float(x) for x in err_tau_rmse_j],
        "time_per_sample_s": float(t_eval),
        "hz": float(1.0 / t_eval) if t_eval > 0 else None,
    }

    # Always write txt/json even in headless mode
    metrics_path = os.path.join(model_dir, "metrics_test.txt")
    with open(metrics_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"hp_preset={args.hp_preset}\n")
        f.write(f"hyper={hyper}\n")
        f.write(f"npz={args.npz}\n")
        f.write(f"ckpt={ckpt_path}\n")
        f.write(f"seed={seed}\n")
        f.write(f"model_type={model_choice}\n")
        f.write(f"dt={dt}\n")
        f.write(f"n_dof={n_dof}\n")
        f.write(f"torque_mse={err_tau}\n")
        f.write(f"torque_rmse={err_tau_rmse}\n")
        f.write("torque_mse_per_joint=" + " ".join([str(float(x)) for x in err_tau_j]) + "\n")
        f.write("torque_rmse_per_joint=" + " ".join([str(float(x)) for x in err_tau_rmse_j]) + "\n")
        f.write(f"time_per_sample_s={t_eval}\n")
        f.write(f"hz={1.0/t_eval if t_eval > 0 else None}\n")
    print(f"Saved metrics TXT: {metrics_path}")

    metrics_json_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {metrics_json_path}")

    # Register artifacts we already wrote
    metrics.setdefault("artifacts", {})
    metrics["artifacts"]["metrics_txt"] = metrics_path
    metrics["artifacts"]["metrics_json"] = metrics_json_path

    # ----------------------------
    # Plot torque (ONLY if matplotlib available)
    # ----------------------------
    if plt is not None:
        pred_tau_np = np.array(pred_tau)

        fig = plt.figure(figsize=(14, 8), dpi=100)
        title = f"{run_name} | DeLaN Torque | Seed={seed} | {model_choice}"
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        for j in range(min(n_dof, 6)):
            ax = fig.add_subplot(3, 2, j + 1)
            ax.set_title(f"Joint {j}")
            ax.plot(np.asarray(test_tau)[:, j], label="GT", linewidth=1.0)
            ax.plot(pred_tau_np[:, j], label="DeLaN", linewidth=1.0, alpha=0.85)
            ax.grid(True, alpha=0.2)
            if j == 0:
                ax.legend()

        plt.tight_layout()

        plot_path = os.path.join(model_dir, f"{run_name}__DeLaN_Torque.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path}")

        metrics["artifacts"]["torque_plot_png"] = plot_path

        # show only if render=1
        if render:
            plt.show()
        else:
            plt.close(fig)
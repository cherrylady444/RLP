import os
import requests
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RESULTS_DIR = "results_final_experiment"
HISTORY_DIR = os.path.join(RESULTS_DIR, "history")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

DATASETS = {
    "Random": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_random-v2.hdf5",
    "Medium": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
    "Expert": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_expert-v2.hdf5"
}

SIZES = [50, 100, 500, 1000, 10000]
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2]

NOISE_LEVEL = 0.0
MAX_EPOCHS = 2000
BATCH_SIZE = 512

USE_ADAM = True          
LEARNING_RATE = 1e-3   
SGD_MOMENTUM = 0.9

# Threshold line types
# P_EQUALS_N_THRESHOLD - calculated using parameter count vs data points
P_EQUALS_N_THRESHOLD = False   

# Empirical threshold - based on train loss <= eps
EMPIRICAL_THRESHOLD = True 
EMPIRICAL_EPS = 1e-2


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def download_dataset(name, url):
    ensure_dir(RESULTS_DIR)
    fname = url.split('/')[-1]
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return path

def load_data(name, url, size):
    path = download_dataset(name, url)
    try:
        with h5py.File(path, "r") as f:
            obs = f['observations'][:]
            act = f['actions'][:]
            next_obs = f['next_observations'][:]
            rew = f['rewards'][:]
            terminals = f['terminals'][:].astype(bool)
    except:
        return None, None, None, None

    valid = ~terminals
    obs, act, next_obs, rew = obs[valid], act[valid], next_obs[valid], rew[valid]
    deltas = next_obs - obs

    def normalize(d):
        return (d - np.mean(d, axis=0)) / (np.std(d, axis=0) + 1e-6)

    X = np.hstack([normalize(obs), normalize(act)])
    Y = np.hstack([normalize(deltas), normalize(rew).reshape(-1, 1)])

    perm = np.random.RandomState(42).permutation(len(X))
    current_size = min(size, len(X))
    train_idx, val_idx = perm[:current_size], perm[-2000:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    if NOISE_LEVEL > 0.0:
        rng = np.random.RandomState(42)
        noise = rng.normal(scale=NOISE_LEVEL * np.std(Y_train, axis=0), size=Y_train.shape)
        Y_train += noise

    return X_train, Y_train, X_val, Y_val

def calculate_threshold_width(N, input_dim, output_dim):
    """
    Total parameters for 2-hidden-layer MLP:
    P(W) = W^2 + W(input_dim + output_dim + 2) + output_dim
    Solve P(W) = N for W (positive root).
    """
    a = 1.0
    b = input_dim + output_dim + 2.0
    c = output_dim - N

    delta = b**2 - 4*a*c
    if delta < 0:
        return 0.0
    w_threshold = (-b + np.sqrt(delta)) / (2*a)
    return float(w_threshold)

def plot_learning_curves(ds_name, size, width, seed, t_hist, v_hist, save_folder):
    """
    3 plots per (seed, width): train curve, val curve, train+val curve.
    """
    ensure_dir(save_folder)

    plt.figure(figsize=(9, 5))
    plt.plot(t_hist)
    plt.title(f"Train MSE | {ds_name} N={size} W={width} seed={seed}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "1_train_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(v_hist)
    plt.title(f"Validation MSE | {ds_name} N={size} W={width} seed={seed}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "2_val_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(t_hist, label="Train")
    plt.plot(v_hist, label="Val")
    plt.title(f"Train vs Val MSE | {ds_name} N={size} W={width} seed={seed}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "3_train_vs_val.png"), dpi=150)
    plt.close()

def generate_heatmaps(ds_name, size, widths, epochs, input_dim, output_dim,
                      N_train, seed=None, average_over_seeds=False):
    """
    Generates 3 heatmaps (train, val, val+threshold) across widths over epochs.
    """
    if average_over_seeds:
        tag = "avg_seeds"
    else:
        tag = f"seed{seed}"

    save_folder = os.path.join(PLOTS_DIR, f"{ds_name}_{size}", "heatmaps", tag)
    ensure_dir(save_folder)

    train_mat = np.zeros((epochs, len(widths)))
    val_mat = np.zeros((epochs, len(widths)))
    found_any = False

    for i, w in enumerate(widths):
        t_list, v_list = [], []

        seeds_to_use = SEEDS if average_over_seeds else [seed]
        for s in seeds_to_use:
            fname = os.path.join(HISTORY_DIR, f"{ds_name}_{size}_{w}_seed{s}.npz")
            if os.path.exists(fname):
                data = np.load(fname)
                t = np.array(data["train_loss"][:epochs], dtype=np.float32)
                v = np.array(data["val_loss"][:epochs], dtype=np.float32)
                t_list.append(t)
                v_list.append(v)

        if len(t_list) > 0:
            min_len = min(len(x) for x in t_list)
            t_list = [x[:min_len] for x in t_list]
            v_list = [x[:min_len] for x in v_list]
            train_mat[:min_len, i] = np.mean(t_list, axis=0)
            val_mat[:min_len, i] = np.mean(v_list, axis=0)
            found_any = True

    if not found_any:
        return

    combined = np.concatenate([train_mat.flatten(), val_mat.flatten()])
    robust_max = np.percentile(combined, 95)
    norm = mcolors.Normalize(vmin=0, vmax=robust_max)

    thresh_w = calculate_threshold_width(N_train, input_dim, output_dim)
    x_line_pos = np.interp(thresh_w, widths, np.arange(len(widths)))

    empirical_curve_x = None
    if EMPIRICAL_THRESHOLD:
        xs = []
        ys = []
        for t in range(epochs):
            idxs = np.where(train_mat[t, :] <= EMPIRICAL_EPS)[0]
            if len(idxs) > 0:
                xs.append(idxs[0])
                ys.append(t)
        if len(xs) > 5:
            empirical_curve_x = (np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32))

    def plot_heatmap(matrix, title, filename, show_threshold=False):
        plt.figure(figsize=(10, 6))
        plt.imshow(
            matrix,
            aspect="auto",
            cmap="magma",
            norm=norm,
            origin="lower",
            interpolation="nearest",
            extent=[-0.5, len(widths) - 0.5, 0, epochs],
        )

        if show_threshold:
            if P_EQUALS_N_THRESHOLD:
                plt.axvline(
                    x=x_line_pos,
                    color="cyan",
                    linestyle="--",
                    linewidth=2.5,
                    label=f"P ≈ N at W ≈ {int(thresh_w)} (using N_train={N_train})",
                )

            if empirical_curve_x is not None:
                xs, ys = empirical_curve_x
                plt.plot(xs, ys, linestyle="-", linewidth=2.0, label=f"Train <= {EMPIRICAL_EPS:g}")

            plt.legend(loc="upper right", framealpha=0.9, fontsize=9)

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Model Width (Neurons)", fontsize=12)
        plt.ylabel("Epochs", fontsize=12)

        plt.xticks(np.arange(len(widths)), widths)
        plt.yticks(np.linspace(0, epochs, 6))

        plt.colorbar(label="MSE Loss (Linear Scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, filename), dpi=150)
        plt.close()

    plot_heatmap(train_mat, f"Train MSE | {ds_name} N={size} ({tag})", "1_train_mse.png", show_threshold=False)
    plot_heatmap(val_mat, f"Validation MSE | {ds_name} N={size} ({tag})", "2_val_mse.png", show_threshold=False)
    plot_heatmap(val_mat, f"Val MSE + Threshold | {ds_name} N={size} ({tag})", "3_val_mse_threshold.png", show_threshold=True)

class MLP(nn.Module):
    width: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(self.width)(x)
        x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(p):
        preds = state.apply_fn(p, batch_x)
        return jnp.mean((preds - batch_y) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

@jax.jit
def eval_step(state, batch_x, batch_y):
    preds = state.apply_fn(state.params, batch_x)
    return jnp.mean((preds - batch_y) ** 2)

def main():
    ensure_dir(HISTORY_DIR)
    ensure_dir(PLOTS_DIR)

    completed_runs = set(f for f in os.listdir(HISTORY_DIR) if f.endswith(".npz"))
    total = len(DATASETS) * len(SIZES) * len(WIDTHS) * len(SEEDS)
    pbar = tqdm(total=total, desc="Experiment Progress")
    pbar.update(len(completed_runs))

    for ds_name, ds_url in DATASETS.items():
        for size in SIZES:
            X_tr, Y_tr, X_val, Y_val = load_data(ds_name, ds_url, size)
            if X_tr is None:
                continue

            input_dim, output_dim = X_tr.shape[1], Y_tr.shape[1]
            N_train = int(X_tr.shape[0])
            batches = max(1, len(X_tr) // BATCH_SIZE)

            X_tr_j = jax.device_put(jnp.array(X_tr))
            Y_tr_j = jax.device_put(jnp.array(Y_tr))
            X_val_j = jax.device_put(jnp.array(X_val))
            Y_val_j = jax.device_put(jnp.array(Y_val))

            for width in WIDTHS:
                for seed in SEEDS:
                    fname = f"{ds_name}_{size}_{width}_seed{seed}.npz"
                    hist_path = os.path.join(HISTORY_DIR, fname)

                    if fname in completed_runs:
                        data = np.load(hist_path)
                        t_hist = data["train_loss"]
                        v_hist = data["val_loss"]
                    else:
                        model = MLP(width=width, output_dim=output_dim)
                        key = jax.random.PRNGKey(seed)
                        params = model.init(key, jnp.ones((1, input_dim)))

                        if USE_ADAM:
                            tx = optax.adam(LEARNING_RATE)
                        else:
                            tx = optax.sgd(LEARNING_RATE, momentum=SGD_MOMENTUM)

                        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

                        t_hist, v_hist = [], []
                        for epoch in range(MAX_EPOCHS):
                            key, subkey = jax.random.split(key)
                            perms = jax.random.permutation(subkey, len(X_tr))
                            for i in range(batches):
                                idx = perms[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                                state, _ = train_step(state, X_tr_j[idx], Y_tr_j[idx])

                            t_hist.append(float(eval_step(state, X_tr_j, Y_tr_j)))
                            v_hist.append(float(eval_step(state, X_val_j, Y_val_j)))

                        np.savez_compressed(hist_path, train_loss=np.array(t_hist), val_loss=np.array(v_hist))
                        pbar.update(1)

                    run_plot_folder = os.path.join(
                        PLOTS_DIR, f"{ds_name}_{size}", "runs", f"W{width}", f"seed{seed}"
                    )
                    plot_learning_curves(ds_name, size, width, seed, t_hist, v_hist, run_plot_folder)

            for seed in SEEDS:
                generate_heatmaps(
                    ds_name=ds_name,
                    size=size,
                    widths=WIDTHS,
                    epochs=MAX_EPOCHS,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    N_train=N_train,
                    seed=seed,
                    average_over_seeds=False,
                )

            generate_heatmaps(
                ds_name=ds_name,
                size=size,
                widths=WIDTHS,
                epochs=MAX_EPOCHS,
                input_dim=input_dim,
                output_dim=output_dim,
                N_train=N_train,
                seed=None,
                average_over_seeds=True,
            )

    pbar.close()
    print("Experiment Complete.")
    print(f"- Heatmaps: {PLOTS_DIR}/<ds>_<N>/heatmaps/(avg_seeds|seedX)")
    print(f"- Run curves: {PLOTS_DIR}/<ds>_<N>/runs/W<width>/seed<seed>/")

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
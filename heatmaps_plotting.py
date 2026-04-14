import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

INPUT_DIR = "results_final_experiment/history"
OUTPUT_DIR = "results_emc_heatmaps"

# HalfCheetah dimensions
INPUT_DIM = 23
OUTPUT_DIM = 18

# EMC threshold options (will generate plots for each)
EMC_THRESHOLDS = [0.01, 0.05, 0.1, 0.25]

# Default threshold for main plots
DEFAULT_EMC_THRESHOLD = 0.25

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def calculate_interpolation_threshold_width(n_samples):
    """Solve P(W) = N for W."""
    a, b, c = 1.0, INPUT_DIM + OUTPUT_DIM + 2.0, OUTPUT_DIM - n_samples
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 0.0
    return (-b + np.sqrt(discriminant)) / (2*a)


def find_emc_epoch(train_loss, threshold):
    """
    Find the Effective Model Complexity epoch - 
    the first epoch where train loss drops below threshold.
    """
    below = np.where(train_loss < threshold)[0]
    return int(below[0]) if len(below) > 0 else None

def load_all_data():
    """Load all experiment results."""
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: {INPUT_DIR} not found!")
        return []
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]
    data = []
    
    print(f"Loading {len(files)} files...")
    for f in tqdm(files, desc="Loading"):
        try:
            parts = f.replace(".npz", "").split("_")
            ds, size, width = parts[0], int(parts[1]), int(parts[2])
            seed = int(parts[3].replace("seed", ""))
            
            content = np.load(os.path.join(INPUT_DIR, f))
            train_loss = np.array(content['train_loss'][:2000], dtype=np.float64)
            val_loss = np.array(content['val_loss'][:2000], dtype=np.float64)
            
            if len(val_loss) < 100:
                continue
            
            data.append({
                'dataset': ds,
                'size': size,
                'width': width,
                'seed': seed,
                'train_loss': train_loss,
                'val_loss': val_loss,
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    print(f"Loaded {len(data)} runs")
    return data

def plot_emc_heatmap(dataset, size, seed, widths, val_matrix, train_matrix,
                     emc_threshold, save_path):
    """
    Plot validation MSE heatmap with EMC contour line.
    
    The EMC line shows when each model width achieves train loss < threshold,
    indicating the transition to interpolation.
    """
    n_epochs = val_matrix.shape[0]
    theo_w = calculate_interpolation_threshold_width(size)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    robust_max = np.percentile(val_matrix[val_matrix > 0], 95)
    norm = mcolors.Normalize(vmin=0, vmax=robust_max)
    
    im = ax.imshow(val_matrix, aspect='auto', cmap='magma', norm=norm,
                   origin='lower', interpolation='nearest',
                   extent=[-0.5, len(widths)-0.5, 0, n_epochs])
    
    emc_epochs = []
    emc_widths_idx = []
    
    for i, w in enumerate(widths):
        train_col = train_matrix[:, i]
        emc = find_emc_epoch(train_col, emc_threshold)
        if emc is not None:
            emc_epochs.append(emc)
            emc_widths_idx.append(i)
    
    if len(emc_epochs) > 1:
        step_x = []
        step_y = []
        
        for i in range(len(emc_widths_idx)):
            if i == 0:
                step_x.append(emc_widths_idx[i] - 0.5)
            else:
                step_x.append(emc_widths_idx[i-1] + 0.5)
            step_y.append(emc_epochs[i])
            
            step_x.append(emc_widths_idx[i] + 0.5)
            step_y.append(emc_epochs[i])
        
        step_x_ext = [step_x[0]] + step_x + [step_x[-1]]
        step_y_ext = [n_epochs] + step_y + [0]
        
        ax.plot(step_x_ext, step_y_ext, color='cyan', linestyle='--', 
                linewidth=2.5, label=f'Train Loss ≤ {emc_threshold}')
        
        mid_idx = len(step_x) // 2
        ax.text(step_x[mid_idx] + 0.5, step_y[mid_idx] + 100, 
                'Effective\nModel\nComplexity\n(IT)', 
                color='cyan', fontsize=11, fontweight='bold',
                ha='left', va='bottom')
    
    if theo_w >= widths[0] * 0.3 and theo_w <= widths[-1] * 2:
        x_line_pos = np.interp(theo_w, widths, np.arange(len(widths)))
        ax.axvline(x=x_line_pos, color='lime', linestyle=':', linewidth=2,
                   label=f'P ≈ N (W ≈ {theo_w:.1f})')
    
    ax.set_title(f"Val MSE| {dataset} N={size} (seed{seed})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Model Width (Neurons)", fontsize=12)
    ax.set_ylabel("Epochs", fontsize=12)
    ax.set_xticks(np.arange(len(widths)))
    ax.set_xticklabels(widths)
    
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MSE Loss (Linear Scale)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_emc_heatmap_train(dataset, size, seed, widths, train_matrix,
                           emc_threshold, save_path):
    """
    Plot training MSE heatmap with EMC contour line.
    Shows where training loss drops below threshold.
    """
    n_epochs = train_matrix.shape[0]
    theo_w = calculate_interpolation_threshold_width(size)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    robust_max = np.percentile(train_matrix[train_matrix > 0], 95)
    norm = mcolors.Normalize(vmin=0, vmax=robust_max)
    
    im = ax.imshow(train_matrix, aspect='auto', cmap='viridis', norm=norm,
                   origin='lower', interpolation='nearest',
                   extent=[-0.5, len(widths)-0.5, 0, n_epochs])
    
    emc_epochs = []
    emc_widths_idx = []
    
    for i, w in enumerate(widths):
        train_col = train_matrix[:, i]
        emc = find_emc_epoch(train_col, emc_threshold)
        if emc is not None:
            emc_epochs.append(emc)
            emc_widths_idx.append(i)
    
    if len(emc_epochs) > 1:
        step_x = []
        step_y = []
        
        for i in range(len(emc_widths_idx)):
            if i == 0:
                step_x.append(emc_widths_idx[i] - 0.5)
            else:
                step_x.append(emc_widths_idx[i-1] + 0.5)
            step_y.append(emc_epochs[i])
            step_x.append(emc_widths_idx[i] + 0.5)
            step_y.append(emc_epochs[i])
        
        step_x_ext = [step_x[0]] + step_x + [step_x[-1]]
        step_y_ext = [n_epochs] + step_y + [0]
        
        ax.plot(step_x_ext, step_y_ext, color='red', linestyle='--', 
                linewidth=2.5, label=f'Train Loss ≤ {emc_threshold}')
    
    if theo_w >= widths[0] * 0.3 and theo_w <= widths[-1] * 2:
        x_line_pos = np.interp(theo_w, widths, np.arange(len(widths)))
        ax.axvline(x=x_line_pos, color='lime', linestyle=':', linewidth=2,
                   label=f'P ≈ N (W ≈ {theo_w:.1f})')
    
    ax.set_title(f"Train MSE + EMC | {dataset} N={size} (seed{seed})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Model Width (Neurons)", fontsize=12)
    ax.set_ylabel("Epochs", fontsize=12)
    ax.set_xticks(np.arange(len(widths)))
    ax.set_xticklabels(widths)
    
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MSE Loss (Linear Scale)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_emc_combined(dataset, size, seed, widths, val_matrix, train_matrix,
                      emc_threshold, save_path):
    """
    Plot combined validation and training heatmaps side by side with EMC.
    """
    n_epochs = val_matrix.shape[0]
    theo_w = calculate_interpolation_threshold_width(size)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    emc_epochs = []
    emc_widths_idx = []
    
    for i, w in enumerate(widths):
        train_col = train_matrix[:, i]
        emc = find_emc_epoch(train_col, emc_threshold)
        if emc is not None:
            emc_epochs.append(emc)
            emc_widths_idx.append(i)
    
    step_x_ext, step_y_ext = None, None
    if len(emc_epochs) > 1:
        step_x = []
        step_y = []
        
        for i in range(len(emc_widths_idx)):
            if i == 0:
                step_x.append(emc_widths_idx[i] - 0.5)
            else:
                step_x.append(emc_widths_idx[i-1] + 0.5)
            step_y.append(emc_epochs[i])
            step_x.append(emc_widths_idx[i] + 0.5)
            step_y.append(emc_epochs[i])
        
        step_x_ext = [step_x[0]] + step_x + [step_x[-1]]
        step_y_ext = [n_epochs] + step_y + [0]
    
    x_line_pos = None
    show_threshold = theo_w >= widths[0] * 0.3 and theo_w <= widths[-1] * 2
    if show_threshold:
        x_line_pos = np.interp(theo_w, widths, np.arange(len(widths)))
    
    ax = axes[0]
    robust_max_val = np.percentile(val_matrix[val_matrix > 0], 95)
    norm_val = mcolors.Normalize(vmin=0, vmax=robust_max_val)
    
    im = ax.imshow(val_matrix, aspect='auto', cmap='magma', norm=norm_val,
                   origin='lower', interpolation='nearest',
                   extent=[-0.5, len(widths)-0.5, 0, n_epochs])
    
    if step_x_ext is not None:
        ax.plot(step_x_ext, step_y_ext, color='cyan', linestyle='--', 
                linewidth=2.5, label=f'EMC (Train ≤ {emc_threshold})')
        
        mid_idx = len(step_x_ext) // 3
        ax.text(step_x_ext[mid_idx] + 0.3, step_y_ext[mid_idx] + 50, 
                'EMC', color='cyan', fontsize=12, fontweight='bold')
    
    if x_line_pos is not None:
        ax.axvline(x=x_line_pos, color='lime', linestyle=':', linewidth=2,
                   label=f'P ≈ N (W ≈ {theo_w:.1f})')
    
    ax.set_title(f"Validation MSE", fontsize=13, fontweight='bold')
    ax.set_xlabel("Model Width (Neurons)", fontsize=11)
    ax.set_ylabel("Epochs", fontsize=11)
    ax.set_xticks(np.arange(len(widths)))
    ax.set_xticklabels(widths)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    plt.colorbar(im, ax=ax, label='MSE')
    
    ax = axes[1]
    robust_max_train = np.percentile(train_matrix[train_matrix > 0], 95)
    norm_train = mcolors.Normalize(vmin=0, vmax=robust_max_train)
    
    im = ax.imshow(train_matrix, aspect='auto', cmap='viridis', norm=norm_train,
                   origin='lower', interpolation='nearest',
                   extent=[-0.5, len(widths)-0.5, 0, n_epochs])
    
    if step_x_ext is not None:
        ax.plot(step_x_ext, step_y_ext, color='red', linestyle='--', 
                linewidth=2.5, label=f'EMC (Train ≤ {emc_threshold})')
    
    if x_line_pos is not None:
        ax.axvline(x=x_line_pos, color='lime', linestyle=':', linewidth=2,
                   label=f'P ≈ N (W ≈ {theo_w:.1f})')
    
    ax.set_title(f"Training MSE", fontsize=13, fontweight='bold')
    ax.set_xlabel("Model Width (Neurons)", fontsize=11)
    ax.set_ylabel("Epochs", fontsize=11)
    ax.set_xticks(np.arange(len(widths)))
    ax.set_xticklabels(widths)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    plt.colorbar(im, ax=ax, label='MSE')
    
    fig.suptitle(f"EMC Heatmaps | {dataset} N={size} (seed{seed}) | Threshold={emc_threshold}",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    print("="*70)
    print("EMC (Effective Model Complexity) Heatmap Generator")
    print("="*70)
    
    data = load_all_data()
    if not data:
        print("No data found!")
        return
    
    organized = {}
    for run in data:
        key = (run['dataset'], run['size'], run['seed'])
        if key not in organized:
            organized[key] = {}
        organized[key][run['width']] = run
    
    print(f"\nFound {len(organized)} unique (dataset, size, seed) combinations")
    print(f"Generating heatmaps for EMC thresholds: {EMC_THRESHOLDS}")
    
    for (ds, size, seed), width_dict in tqdm(organized.items(), desc="Generating"):
        widths = sorted(width_dict.keys())
        if len(widths) < 4:
            continue
        
        n_epochs = 2000
        
        val_matrix = np.zeros((n_epochs, len(widths)))
        train_matrix = np.zeros((n_epochs, len(widths)))
        
        for i, w in enumerate(widths):
            run = width_dict[w]
            val_matrix[:, i] = run['val_loss'][:n_epochs]
            train_matrix[:, i] = run['train_loss'][:n_epochs]
        
        config_dir = os.path.join(OUTPUT_DIR, f"{ds}_N{size}", f"seed{seed}")
        ensure_dir(config_dir)
        
        plot_emc_heatmap(
            ds, size, seed, widths, val_matrix, train_matrix,
            DEFAULT_EMC_THRESHOLD,
            os.path.join(config_dir, f"val_mse_emc_{DEFAULT_EMC_THRESHOLD}.png")
        )
        
        plot_emc_heatmap_train(
            ds, size, seed, widths, train_matrix,
            DEFAULT_EMC_THRESHOLD,
            os.path.join(config_dir, f"train_mse_emc_{DEFAULT_EMC_THRESHOLD}.png")
        )
        
        plot_emc_combined(
            ds, size, seed, widths, val_matrix, train_matrix,
            DEFAULT_EMC_THRESHOLD,
            os.path.join(config_dir, f"combined_emc_{DEFAULT_EMC_THRESHOLD}.png")
        )
        
        thresholds_dir = os.path.join(config_dir, "threshold_comparison")
        ensure_dir(thresholds_dir)
        
        for thresh in EMC_THRESHOLDS:
            plot_emc_heatmap(
                ds, size, seed, widths, val_matrix, train_matrix,
                thresh,
                os.path.join(thresholds_dir, f"val_mse_thresh_{thresh}.png")
            )
    
    print("\nGenerating seed-averaged plots...")
    
    by_config = {}
    for run in data:
        key = (run['dataset'], run['size'])
        if key not in by_config:
            by_config[key] = {}
        if run['width'] not in by_config[key]:
            by_config[key][run['width']] = []
        by_config[key][run['width']].append(run)
    
    for (ds, size), width_dict in tqdm(by_config.items(), desc="Averaging"):
        widths = sorted(width_dict.keys())
        if len(widths) < 4:
            continue
        
        n_epochs = 2000
        
        val_matrix = np.zeros((n_epochs, len(widths)))
        train_matrix = np.zeros((n_epochs, len(widths)))
        
        for i, w in enumerate(widths):
            runs = width_dict[w]
            val_matrix[:, i] = np.mean([r['val_loss'][:n_epochs] for r in runs], axis=0)
            train_matrix[:, i] = np.mean([r['train_loss'][:n_epochs] for r in runs], axis=0)
        
        avg_dir = os.path.join(OUTPUT_DIR, f"{ds}_N{size}", "averaged")
        ensure_dir(avg_dir)
        
        plot_emc_heatmap(
            ds, size, "avg", widths, val_matrix, train_matrix,
            DEFAULT_EMC_THRESHOLD,
            os.path.join(avg_dir, f"val_mse_emc_{DEFAULT_EMC_THRESHOLD}.png")
        )
        
        plot_emc_combined(
            ds, size, "avg", widths, val_matrix, train_matrix,
            DEFAULT_EMC_THRESHOLD,
            os.path.join(avg_dir, f"combined_emc_{DEFAULT_EMC_THRESHOLD}.png")
        )
    
    print("COMPLETE")

if __name__ == "__main__":
    main()
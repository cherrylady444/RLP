import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import json


INPUT_DIR = "results_final_experiment/history"
OUTPUT_DIR = "results_dd_analysis"

# HalfCheetah dimensions
INPUT_DIM = 23
OUTPUT_DIM = 18

# Detection parameters
EPOCH_WISE_CONFIG = {
    'prominence_min': 0.002,
    'peak_min_epoch': 30,
    'peak_max_epoch': 1950,
    'min_drop_after_peak': 0.01,
}

MODEL_WISE_CONFIG = {
    'prominence_min': 0.005,
    'min_drop_ratio': 0.95,
    'scan_epochs': list(range(0, 2001, 10)), 
}

TRAIN_ZERO_EPSILON = 0.02

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_python_type(data), f, indent=2)

def calculate_num_parameters(width):
    """Calculate total parameters for the specific architecture."""
    return width**2 + width * (INPUT_DIM + OUTPUT_DIM + 2) + OUTPUT_DIM

def calculate_interpolation_threshold_width(n_samples):
    """
    Solve P(W) = N for W.
    P(W) = W^2 + W(Input + Output + 2) + Output
    """
    a, b, c = 1.0, INPUT_DIM + OUTPUT_DIM + 2.0, OUTPUT_DIM - n_samples
    discriminant = b**2 - 4*a*c
    if discriminant < 0: return 0.0
    return (-b + np.sqrt(discriminant)) / (2*a)

def find_interpolation_epoch(train_loss):
    """Find the first epoch where training loss drops near zero."""
    below = np.where(train_loss < TRAIN_ZERO_EPSILON)[0]
    return int(below[0]) if len(below) > 0 else None

def analyze_regime(widths, n_samples):
    """Determine if the range of widths covers the interpolation threshold."""
    widths = np.array(widths)
    theo_width = calculate_interpolation_threshold_width(n_samples)
    min_w, max_w = widths.min(), widths.max()
    threshold_visible = (theo_width >= min_w * 0.5) and (theo_width <= max_w * 2.0)
    
    min_params = calculate_num_parameters(min_w)
    max_params = calculate_num_parameters(max_w)
    
    if min_params < n_samples < max_params:
        status = 'mixed'
    elif min_params > n_samples:
        status = 'all_over'
    else:
        status = 'all_under'

    return {
        'theo_width': theo_width,
        'regime_status': status,
        'threshold_visible': threshold_visible,
    }

def load_all_data():
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
            
            if len(val_loss) < 100: continue
            
            data.append({
                'dataset': ds, 'size': size, 'width': width, 'seed': seed,
                'train_loss': train_loss, 'val_loss': val_loss,
                'n_params': calculate_num_parameters(width),
                'theo_width': calculate_interpolation_threshold_width(size),
                'interpolation_epoch': find_interpolation_epoch(train_loss),
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return data

def detect_epoch_wise_dd(run):
    """
    Detects if validation loss increases then decreases significantly over time.
    """
    cfg = EPOCH_WISE_CONFIG
    val_loss = run['val_loss']
    
    peaks, props = find_peaks(val_loss, prominence=cfg['prominence_min'], width=5)
    
    if len(peaks) == 0: return None
    
    detections = []
    for i, peak_idx in enumerate(peaks):
        if peak_idx < cfg['peak_min_epoch'] or peak_idx > cfg['peak_max_epoch']: continue
        
        peak_val = val_loss[peak_idx]
        post_peak_min = np.min(val_loss[peak_idx:])
        drop_after = peak_val - post_peak_min
        
        if drop_after < cfg['min_drop_after_peak']: continue
        
        pre_peak_min = np.min(val_loss[:peak_idx]) if peak_idx > 0 else val_loss[0]
        dd_strength = drop_after / (peak_val + 1e-8)
        
        detections.append({
            'peak_epoch': int(peak_idx),
            'peak_value': float(peak_val),
            'prominence': float(props['prominences'][i]),
            'dd_strength': float(dd_strength),
            'beats_classical': bool(post_peak_min < pre_peak_min),
            'interpolation_epoch': run['interpolation_epoch'],
        })
    
    if not detections: return None
    return max(detections, key=lambda x: x['dd_strength'])

def detect_model_wise_dd(widths, val_errors, train_errors, n_samples):
    """
    Detects if validation loss peaks at a specific model width.
    """
    cfg = MODEL_WISE_CONFIG
    widths = np.array(widths)
    val_errors = np.array(val_errors)
    
    regime_info = analyze_regime(widths, n_samples)
    
    peaks, props = find_peaks(val_errors, prominence=cfg['prominence_min'])
    
    if len(peaks) == 0: return None
    
    detections = []
    for i, peak_idx in enumerate(peaks):
        if peak_idx == 0 or peak_idx == len(val_errors) - 1: continue
        
        peak_width = widths[peak_idx]
        peak_val = val_errors[peak_idx]
        left_min = np.min(val_errors[:peak_idx])
        right_min = np.min(val_errors[peak_idx:])
        
        if right_min > peak_val * cfg['min_drop_ratio']: continue
        
        dd_strength = min(peak_val - left_min, peak_val - right_min) / (peak_val + 1e-8)
        
        peak_params = calculate_num_parameters(peak_width)
        if peak_params < n_samples * 0.85: peak_regime = 'underparameterized'
        elif peak_params > n_samples * 1.15: peak_regime = 'overparameterized'
        else: peak_regime = 'critical'
        
        detections.append({
            'peak_idx': int(peak_idx),
            'peak_width': int(peak_width),
            'peak_value': float(peak_val),
            'prominence': float(props['prominences'][i]),
            'peak_regime': peak_regime,
            'dd_strength': float(dd_strength),
            'regime_info': regime_info,
        })
    
    if not detections: return None
    return max(detections, key=lambda x: x['dd_strength'])

def plot_epoch_wise_case(run, detection, save_path):
    """
    Plots Loss vs Epochs (Time) with the DD peak marked.
    Refined Style: Moves info text outside the plot to prevent overlap.
    """
    epochs = np.arange(len(run['val_loss']))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    plt.subplots_adjust(right=0.75)
    
    ax.plot(epochs, run['train_loss'], color='green', alpha=0.3, label='Train Loss', linewidth=1.5)
    ax.plot(epochs, run['val_loss'], color='navy', alpha=0.9, label='Val Loss', linewidth=2)
    
    peak_idx = detection['peak_epoch']
    ax.plot(peak_idx, run['val_loss'][peak_idx], 'ro', markersize=8, label=f"Peak (Ep {peak_idx})")
    
    if detection['interpolation_epoch']:
        ax.axvline(detection['interpolation_epoch'], color='orange', linestyle='--', 
                   alpha=0.8, label='Interp Epoch (Train~0)')
        
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    
    info_text = (
        f"CONFIGURATION\n"
        f"-----------------\n"
        f"Dataset: {run['dataset']}\n"
        f"Samples (N): {run['size']}\n"
        f"Width (W): {run['width']}\n"
        f"Seed: {run['seed']}\n\n"
        f"STATISTICS\n"
        f"-----------------\n"
        f"Params: {run['n_params']:,}\n"
        f"Peak Ep: {detection['peak_epoch']}\n"
        f"Strength: {detection['dd_strength']:.4f}\n"
        f"Prominence: {detection['prominence']:.4f}"
    )
    
    fig.text(0.77, 0.85, info_text, fontsize=11, fontfamily='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'))

    ax.set_title(f"Epoch-wise DD: {run['dataset']} (N={run['size']}, W={run['width']})", fontweight='bold', fontsize=14)
    
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_model_wise_curve_n(dataset, size, epoch, widths, val_errors, train_errors,
                            detection, save_path):
    """
    Plots Loss vs Model Width (log scale) at a specific epoch.
    """
    widths = np.array(widths)
    x_coords = np.log2(widths)
    
    regime_info = detection['regime_info'] if detection else analyze_regime(widths, size)
    theo_w = regime_info['theo_width']
    theo_n = np.log2(theo_w) if theo_w > 0 else 0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    
    ax.plot(x_coords, val_errors, 'o-', color='crimson', linewidth=2, label='Test Error')
    ax.plot(x_coords, train_errors, 'o--', color='darkgreen', alpha=0.6, label='Train Error')
    
    show_threshold = regime_info['threshold_visible'] and theo_w >= widths.min()
    if show_threshold:
        ax.axvline(theo_n, color='black', linestyle='-.', label=f'Threshold (W={theo_w:.1f})')
        ax.axvspan(theo_n - 0.5, theo_n + 0.5, color='grey', alpha=0.1)

    if detection:
        peak_w = detection['peak_width']
        peak_n = np.log2(peak_w)
        ax.scatter([peak_n], [detection['peak_value']], color='gold', s=200, marker='*', 
                   edgecolors='darkred', zorder=5, label=f'Peak (W={peak_w})')
    
    ax.set_xlabel(r"Model Width Exponent $n$ (Width = $2^n$)", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_xticks(x_coords)
    ax.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in x_coords])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True)
    
    header_text = f"Model-wise DD: {dataset} (N={size}) at Epoch {epoch}"
    fig.suptitle(header_text, fontsize=16, fontweight='bold', y=0.98)
    
    lines = [f"Regime: {regime_info['regime_status'].upper()}"]
    if detection:
        lines.append(f"Peak Strength: {detection['dd_strength']:.3f}")
        lines.append(f"Prominence: {detection['prominence']:.4f}")
    
    fig.text(0.5, 0.94, " | ".join(lines), ha='center', fontsize=11, color='#333')
    
    plt.savefig(save_path)
    plt.close()

def scan_and_visualize(all_data):
    print("\n--- Scanning Epoch-wise ---")
    epoch_save_dir = os.path.join(OUTPUT_DIR, "1_Epoch_Wise")
    ensure_dir(epoch_save_dir)
    
    epoch_detections = []
    for run in tqdm(all_data, desc="Scanning Epoch-wise"):
        det = detect_epoch_wise_dd(run)
        if det:
            sub_dir = os.path.join(epoch_save_dir, f"{run['dataset']}_N{run['size']}")
            ensure_dir(sub_dir)
            fname = os.path.join(sub_dir, f"W{run['width']}_seed{run['seed']}_ep{det['peak_epoch']}.png")
            plot_epoch_wise_case(run, det, fname)
            
            record = run.copy()
            del record['train_loss']
            del record['val_loss']
            record.update(det)
            epoch_detections.append(record)
            
    save_json(epoch_detections, os.path.join(epoch_save_dir, "epoch_detections.json"))

    print("\n--- Scanning Model-wise ---")
    model_save_dir = os.path.join(OUTPUT_DIR, "2_Model_Wise")
    ensure_dir(model_save_dir)
    
    grouped = {}
    for run in all_data:
        key = (run['dataset'], run['size'])
        if key not in grouped: grouped[key] = {}
        if run['width'] not in grouped[key]: grouped[key][run['width']] = []
        grouped[key][run['width']].append(run)
    
    model_detections = []
    
    for (ds, size), width_dict in tqdm(grouped.items(), desc="Processing Groups"):
        widths = sorted(width_dict.keys())
        if len(widths) < 4: continue
        
        n_epochs = 2000
        val_matrix = np.zeros((n_epochs, len(widths)))
        train_matrix = np.zeros((n_epochs, len(widths)))
        
        for i, w in enumerate(widths):
            runs = width_dict[w]
            val_matrix[:, i] = np.mean([r['val_loss'][:n_epochs] for r in runs], axis=0)
            train_matrix[:, i] = np.mean([r['train_loss'][:n_epochs] for r in runs], axis=0)
        
        config_dir = os.path.join(model_save_dir, f"{ds}_N{size}")
        ensure_dir(config_dir)
        
        for epoch in MODEL_WISE_CONFIG['scan_epochs']:
            if epoch >= n_epochs: continue
            
            val_curve = val_matrix[epoch, :]
            train_curve = train_matrix[epoch, :]
            
            det = detect_model_wise_dd(widths, val_curve, train_curve, size)
            if det:
                fname = os.path.join(config_dir, f"Epoch_{epoch:04d}_DD.png")
                plot_model_wise_curve_n(ds, size, epoch, widths, val_curve, train_curve, det, fname)
                
                rec = {'dataset': ds, 'size': size, 'epoch': epoch}
                rec.update(det)
                del rec['regime_info']
                model_detections.append(rec)
                                    
    save_json(model_detections, os.path.join(model_save_dir, "model_detections.json"))
    return epoch_detections, model_detections

def main():
    ensure_dir(OUTPUT_DIR)
    print("="*60)
    print("DOUBLE DESCENT ANALYSIS")
    print("Epoch-wise and Model-wise")
    print("="*60)
    
    data = load_all_data()
    if not data: return
    
    scan_and_visualize(data)
    
    print("\nProcessing Complete.")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
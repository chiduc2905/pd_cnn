"""
CWT Scalogram Generator for Partial Discharge Classification
=============================================================
ADAPTED FOR pulse_minh folder structure:

- 4 classes: corona, hf_nopd, surface, void
- Input: pulse_minh/{corona,hf_nopd,surface,void}/*.mat
- MAT file format: Trace_3_VOLT, Time_s
- 80 MHz sampling frequency
- Random sample 2500 files per class

PIPELINE:
    HFCT pulse → CWT → |coefficients| → log1p(200·|coefficients|) 
    → per-image normalization → Blackbody colormap → resize 224×224

Author: PD Analysis Team
Date: December 2024
"""

import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pywt
import os
import random
from typing import Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional cv2 import with PIL fallback
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image


# =============================================================================
# CONFIGURATION - ADAPTED FOR MINH DATA (PULSE_MINH)
# =============================================================================

# Paths for Linux server
INPUT_ROOT = "/mnt/disk2/nhatnc/res/dataset/pulse_minh"
OUTPUT_ROOT = "/mnt/disk2/nhatnc/res/dataset/scalogram_minh"

# Sampling configuration
SAMPLES_PER_CLASS = 2500
NUM_WORKERS = max(1, cpu_count() - 2)

CONFIG = {
    'freq_min': 50e3,         # 50 kHz
    'freq_max': 16e6,         # 16 MHz
    'n_scales': 200,          # Number of scales (log-spaced)
    'wavelet': 'cmor1.5-1.0', # Complex Morlet
    'log_gain': 200,          # log1p(gain × |CWT|)
    'output_size': (224, 224),
    'colormap': 'inferno',    # Blackbody colormap (Planck spectrum)
}

# 4 Classes for Minh data (from pulse_minh folder)
CLASSES = ['corona', 'hf_nopd', 'surface', 'void']


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def generate_scales_for_frequency_range(freq_min, freq_max, fs, wavelet='cmor1.5-1.0', n_scales=200):
    """Generate log-spaced scales for frequency range."""
    dt = 1.0 / fs
    nyquist = fs / 2
    if freq_max > nyquist:
        freq_max = 0.95 * nyquist
    
    center_freq = pywt.central_frequency(wavelet)
    scale_max = center_freq / (freq_min * dt)
    scale_min = center_freq / (freq_max * dt)
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
    frequencies = pywt.scale2frequency(wavelet, scales) / dt
    
    return scales, frequencies


def compute_cwt(signal, fs):
    """Compute CWT magnitude."""
    scales, frequencies = generate_scales_for_frequency_range(
        CONFIG['freq_min'], CONFIG['freq_max'], fs, 
        CONFIG['wavelet'], CONFIG['n_scales']
    )
    coefficients, _ = pywt.cwt(signal, scales, CONFIG['wavelet'], sampling_period=1.0/fs)
    scalogram_mag = np.abs(coefficients)
    return scalogram_mag, frequencies


def apply_log_compression(scalogram_mag):
    """Apply log compression: log1p(gain × |CWT|)"""
    return np.log1p(CONFIG['log_gain'] * scalogram_mag)


def normalize_per_image(scalogram_log):
    """Normalize to [0, 1] range."""
    vmin = scalogram_log.min()
    vmax = scalogram_log.max()
    if vmax - vmin > 0:
        return (scalogram_log - vmin) / (vmax - vmin)
    return np.zeros_like(scalogram_log)


def apply_colormap(scalogram_norm):
    """Apply Blackbody colormap (hot)."""
    cmap = cm.get_cmap(CONFIG['colormap'])
    scalogram_rgba = cmap(scalogram_norm)
    scalogram_rgb = (scalogram_rgba[:, :, :3] * 255).astype(np.uint8)
    return scalogram_rgb


def resize_image(img, size):
    """Resize image to target size."""
    if HAS_CV2:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    else:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size, Image.LANCZOS)
        return np.array(pil_img)


# =============================================================================
# MAIN SCALOGRAM GENERATION
# =============================================================================

def generate_scalogram(voltage, time, output_path=None):
    """
    Generate 224x224 scalogram with Blackbody colormap.
    Optimized for batch processing (no labeled version).
    """
    fs = 1.0 / np.mean(np.diff(time))
    
    # CWT -> magnitude
    scalogram_mag, frequencies = compute_cwt(voltage, fs)
    
    # Log compression
    scalogram_log = apply_log_compression(scalogram_mag)
    
    # Per-image normalization
    scalogram_norm = normalize_per_image(scalogram_log)
    
    time_us = (time - time[0]) * 1e6
    T, F = np.meshgrid(time_us, frequencies / 1e6)
    
    # Create 224x224 scalogram (no axes, pure scalogram content)
    fig_224, ax_224 = plt.subplots(figsize=(3, 3), dpi=100)
    fig_224.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax_224.axis('off')
    
    ax_224.pcolormesh(T, F, scalogram_norm, shading='gouraud', 
                      cmap=CONFIG['colormap'], vmin=0, vmax=1)
    ax_224.set_xlim(time_us.min(), time_us.max())
    ax_224.set_ylim(frequencies.min() / 1e6, frequencies.max() / 1e6)
    
    # Extract image from plot using ARGB buffer
    fig_224.canvas.draw()
    plot_array_argb = np.frombuffer(fig_224.canvas.tostring_argb(), dtype=np.uint8)
    plot_array_argb = plot_array_argb.reshape(fig_224.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig_224)
    
    # Convert ARGB to BGR (ARGB: [:,:,0]=A, [:,:,1]=R, [:,:,2]=G, [:,:,3]=B)
    r, g, b = plot_array_argb[:, :, 1], plot_array_argb[:, :, 2], plot_array_argb[:, :, 3]
    plot_array_bgr = np.stack([b, g, r], axis=2)
    
    # Resize to 224x224
    if HAS_CV2:
        scalogram_224 = cv2.resize(plot_array_bgr, CONFIG['output_size'], interpolation=cv2.INTER_AREA)
    else:
        # Convert BGR to RGB for PIL
        plot_array_rgb = np.stack([r, g, b], axis=2)
        img = Image.fromarray(plot_array_rgb)
        img = img.resize(CONFIG['output_size'], Image.LANCZOS)
        scalogram_224 = np.array(img)
        # Convert back to BGR
        scalogram_224 = scalogram_224[:, :, ::-1]
    
    # Save 224x224 image
    if output_path:
        if HAS_CV2:
            cv2.imwrite(output_path, scalogram_224)
        else:
            # Convert BGR to RGB for PIL
            Image.fromarray(scalogram_224[:, :, ::-1]).save(output_path)
    
    return scalogram_224


def load_mat_minh_format(file_path):
    """
    Load .mat file in Minh format (Trace_3_VOLT, Time_s).
    Falls back to standard format (Voltage, Time) if not found.
    """
    mat_data = scipy.io.loadmat(file_path)
    
    # Try Minh format first
    if 'Trace_3_VOLT' in mat_data and 'Time_s' in mat_data:
        voltage = mat_data['Trace_3_VOLT'].flatten()
        time = mat_data['Time_s'].flatten()
    # Fallback to standard format
    elif 'Voltage' in mat_data and 'Time' in mat_data:
        voltage = mat_data['Voltage'].flatten()
        time = mat_data['Time'].flatten()
    else:
        raise KeyError(f"Cannot find voltage/time data in {file_path}. "
                      f"Available keys: {list(mat_data.keys())}")
    
    return voltage, time


# =============================================================================
# DATASET-AWARE PROCESSING FOR PULSE_MINH (4 CLASSES)
# =============================================================================

def _process_single_file(args):
    """Worker function for multiprocessing."""
    file_path, output_path = args
    try:
        voltage, time = load_mat_minh_format(file_path)
        generate_scalogram(voltage, time, output_path)
        return (file_path, True, None)
    except Exception as e:
        return (file_path, False, str(e))


def process_pulse_minh_to_scalograms(input_root=INPUT_ROOT, output_root=OUTPUT_ROOT, 
                                      samples_per_class=SAMPLES_PER_CLASS):
    """
    Process pulse_minh dataset and create scalogram dataset.
    Random sample 2500 files from each class.
    
    Input structure:
        pulse_minh/
        ├── corona/*.mat
        ├── hf_nopd/*.mat
        ├── surface/*.mat
        └── void/*.mat
    
    Output structure:
        scalogram_minh/
        ├── corona/*.png  (2500 images)
        ├── hf_nopd/*.png (2500 images)
        ├── surface/*.png (2500 images)
        └── void/*.png    (2500 images)
    
    Parameters:
    ----------
    input_root : str
        Path to pulse_minh dataset root
    output_root : str
        Path to scalogram output root
    samples_per_class : int
        Number of samples per class (default: 2500)
    
    Returns:
    -------
    dict: Summary statistics
    """
    print("="*80)
    print("SCALOGRAM DATASET GENERATION - PULSE_MINH (4 CLASSES)")
    print("="*80)
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Classes: {CLASSES}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Pipeline: CWT -> log1p(200*|CWT|) -> normalize -> {CONFIG['colormap']} -> 224x224")
    print()
    
    # Create output root directory
    os.makedirs(output_root, exist_ok=True)
    
    # Statistics
    stats = {cls: {'total': 0, 'sampled': 0, 'processed': 0, 'failed': 0} for cls in CLASSES}
    
    for cls in CLASSES:
        print(f"\n{'─'*60}")
        print(f"Processing class: {cls.upper()}")
        print(f"{'─'*60}")
        
        input_folder = os.path.join(input_root, cls)
        output_folder = os.path.join(output_root, cls)
        
        if not os.path.exists(input_folder):
            print(f"  ⚠️ {cls}: folder not found, skipping")
            continue
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all .mat files
        all_mat_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.mat')])
        stats[cls]['total'] = len(all_mat_files)
        
        if not all_mat_files:
            print(f"  ⚠️ {cls}: no .mat files found")
            continue
        
        print(f"  📁 {cls}: {len(all_mat_files)} total files")
        
        # Random sample if we have more files than needed
        if len(all_mat_files) > samples_per_class:
            random.seed(42)  # For reproducibility
            sampled_files = random.sample(all_mat_files, samples_per_class)
            print(f"  🎲 Randomly sampled {samples_per_class} files")
        else:
            sampled_files = all_mat_files
            print(f"  📋 Using all {len(sampled_files)} files (less than {samples_per_class})")
        
        stats[cls]['sampled'] = len(sampled_files)
        
        # Prepare arguments for multiprocessing
        file_args = []
        for mat_file in sampled_files:
            file_path = os.path.join(input_folder, mat_file)
            output_name = mat_file.replace('.mat', '.png')
            output_path = os.path.join(output_folder, output_name)
            file_args.append((file_path, output_path))
        
        # Process with multiprocessing
        print(f"  🚀 Processing with {NUM_WORKERS} workers...")
        
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap_unordered(_process_single_file, file_args),
                total=len(file_args),
                desc=f"  [{cls}]"
            ))
        
        # Count results
        for file_path, success, error_msg in results:
            if success:
                stats[cls]['processed'] += 1
            else:
                stats[cls]['failed'] += 1
                if stats[cls]['failed'] <= 3:  # Only show first 3 errors
                    print(f"      ✗ Failed {os.path.basename(file_path)}: {error_msg}")
        
        print(f"  ✓ Processed: {stats[cls]['processed']}, Failed: {stats[cls]['failed']}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_processed = 0
    total_failed = 0
    
    for cls in CLASSES:
        cls_total = stats[cls]['total']
        cls_sampled = stats[cls]['sampled']
        cls_processed = stats[cls]['processed']
        cls_failed = stats[cls]['failed']
        total_processed += cls_processed
        total_failed += cls_failed
        
        print(f"\n{cls.upper()}:")
        print(f"  Total files: {cls_total}")
        print(f"  Sampled: {cls_sampled}")
        print(f"  Processed: {cls_processed}")
        print(f"  Failed: {cls_failed}")
    
    print(f"\n" + "─"*60)
    print(f"TOTAL: {total_processed} scalograms generated ({total_failed} failed)")
    print(f"Output location: {output_root}/")
    print(f"{'='*80}")
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Usage:
        python scalogram_minh.py  # Use default paths
        python scalogram_minh.py --input_root /path/to/pulse_minh --output_root /path/to/scalogram_minh
        python scalogram_minh.py --samples_per_class 1000
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='CWT Scalogram Generator for Pulse Minh Data (4 classes)')
    
    parser.add_argument('--input_root', type=str, default=INPUT_ROOT, 
                        help=f'Input dataset root (default: {INPUT_ROOT})')
    parser.add_argument('--output_root', type=str, default=OUTPUT_ROOT,
                        help=f'Output scalogram root (default: {OUTPUT_ROOT})')
    parser.add_argument('--samples_per_class', type=int, default=SAMPLES_PER_CLASS, 
                        help=f'Number of random samples per class (default: {SAMPLES_PER_CLASS})')
    
    args = parser.parse_args()
    
    # Run processing
    process_pulse_minh_to_scalograms(
        input_root=args.input_root,
        output_root=args.output_root,
        samples_per_class=args.samples_per_class
    )


if __name__ == "__main__":
    main()

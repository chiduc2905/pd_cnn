"""1D Signal Feature Extraction for ML Classification.

Extracts statistical features from 1D PD signals (.mat files) for Pipeline A.

FEATURE EXTRACTION (STRICT from paper):
- Mean
- Variance
- Standard deviation
- Skewness
- Kurtosis

Input: 1D PD signals from .mat files
Output: Feature vectors for ML classifiers (Pipeline A)

Dataset path: /mnt/disk2/nhatnc/res/dataset/pulse_dataset_augmented

Author: PD Analysis Team
"""
import numpy as np
import scipy.io
import scipy.stats
import os
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Fix random seed
SEED = 42
np.random.seed(SEED)


def extract_statistical_features(signal):
    """Extract exactly 5 statistical features as per paper.
    
    Features (STRICT):
    - Mean
    - Variance
    - Standard deviation
    - Skewness
    - Kurtosis
    
    Args:
        signal: 1D numpy array of voltage/amplitude values
        
    Returns:
        features: numpy array of shape (5,)
    """
    mean = np.mean(signal)
    variance = np.var(signal)
    std = np.std(signal)
    skewness = scipy.stats.skew(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    
    return np.array([mean, variance, std, skewness, kurtosis])


def load_mat_file(file_path):
    """Load 1D signal from .mat file.
    
    Supports multiple formats:
    - Trace_3_VOLT, Time_s (Minh format)
    - Voltage, Time (standard format)
    - voltage, time (lowercase)
    
    Args:
        file_path: Path to .mat file
        
    Returns:
        voltage: 1D signal array
        time: Time array (or None if not found)
    """
    mat_data = scipy.io.loadmat(file_path)
    
    # Try different key formats
    voltage_keys = ['Trace_3_VOLT', 'Voltage', 'voltage', 'signal', 'Signal']
    time_keys = ['Time_s', 'Time', 'time']
    
    voltage = None
    time = None
    
    for key in voltage_keys:
        if key in mat_data:
            voltage = mat_data[key].flatten()
            break
    
    for key in time_keys:
        if key in mat_data:
            time = mat_data[key].flatten()
            break
    
    if voltage is None:
        # Try to find any array that looks like signal
        for key, value in mat_data.items():
            if not key.startswith('_') and isinstance(value, np.ndarray):
                if value.size > 100:  # Likely signal data
                    voltage = value.flatten()
                    break
    
    if voltage is None:
        raise KeyError(f"Cannot find signal data in {file_path}. "
                      f"Available keys: {[k for k in mat_data.keys() if not k.startswith('_')]}")
    
    return voltage, time


def process_dataset(data_root, classes=None, samples_per_class=None):
    """Process entire dataset and extract features.
    
    Supports two dataset structures:
    
    1. Flat structure:
        data_root/
        ├── class1/*.mat
        ├── class2/*.mat
        └── class3/*.mat
    
    2. Split structure (train/val/test):
        data_root/
        ├── train/
        │   ├── class1/*.mat
        │   └── class2/*.mat
        ├── val/
        └── test/
    
    Args:
        data_root: Root directory of dataset
        classes: List of class names (subfolders). If None, auto-detect.
        samples_per_class: Max samples per class. If None, use all.
        
    Returns:
        X: Feature matrix (N, 5) or dict with train/val/test splits
        y: Labels (N,) or dict with train/val/test splits
        class_names: List of class names
    """
    # Check if dataset has train/val/test structure
    splits = ['train', 'val', 'test']
    has_splits = all(os.path.isdir(os.path.join(data_root, s)) for s in splits)
    
    if has_splits:
        print(f"Dataset root: {data_root}")
        print(f"Detected train/val/test structure")
        return process_split_dataset(data_root, classes, samples_per_class)
    else:
        return process_flat_dataset(data_root, classes, samples_per_class)


def normalize_signal(signal):
    """Normalize signal to [0, 1] range to ensure fairness.
    
    This forces the model to learn the SHAPE (skewness, kurtosis)
    rather than just the AMPLITUDE (mean, var) which typically varies 
    with distance/sensor gain and causes unfair high accuracy.
    """
    if signal.max() == signal.min():
        return np.zeros_like(signal)
    return (signal - signal.min()) / (signal.max() - signal.min())


def check_data_leakage(train_files, test_files):
    """Check if augmented versions of training files exist in test set.
    
    Assumes filename format: pulse_XXXX[_augY].mat
    """
    def get_base_id(fname):
        # Extract "pulse_XXXX" from "pulse_XXXX_augY.mat"
        base = os.path.basename(fname)
        parts = base.split('_')
        if len(parts) >= 2 and parts[0] == 'pulse':
            return f"{parts[0]}_{parts[1].split('.')[0]}" # pulse_0002
        return base

    train_ids = set(get_base_id(f) for f in train_files)
    test_ids = set(get_base_id(f) for f in test_files)
    
    intersection = train_ids.intersection(test_ids)
    
    if intersection:
        print(f"\n{'!'*60}")
        print(f"WARNING: DATA LEAKAGE DETECTED!")
        print(f"Found {len(intersection)} pulses present in both TRAIN and TEST sets.")
        print(f"Example leakage: {list(intersection)[:3]}")
        print("High accuracy might be due to memorizing augmented samples.")
        print(f"{'!'*60}\n")
        return True
    return False


def process_split_dataset(data_root, classes=None, samples_per_class=None, normalize=True):
    """Process dataset with train/val/test splits already defined."""
    splits = ['train', 'val', 'test']
    
    # Auto-detect classes from train folder
    if classes is None:
        train_dir = os.path.join(data_root, 'train')
        classes = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"Classes: {classes}")
    if normalize:
        print("Note: Signal normalization enabled (fairness check)")
    
    # Collect all files first to check leakage
    split_files = {}
    for split in splits:
        split_files[split] = []
        for class_name in classes:
            class_dir = os.path.join(data_root, split, class_name)
            if os.path.exists(class_dir):
                files = sorted(glob(os.path.join(class_dir, '*.mat')))
                split_files[split].extend(files)
    
    # Check leakage
    check_data_leakage(split_files['train'], split_files['test'])
    
    # Debug: show first .mat file keys (same as before)
    # ... (debug print omitted for brevity)
    
    X_splits = {}
    y_splits = {}
    
    for split in splits:
        print(f"\n{'='*40}")
        print(f"Processing {split.upper()} set")
        print(f"{'='*40}")
        
        all_features = []
        all_labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(data_root, split, class_name)
            
            if not os.path.exists(class_dir):
                # Warning already handled in detection loop logic or acceptable
                continue
            
            # Re-glob needed? We already have them but per-class structure logic is cleaner here
            mat_files = sorted(glob(os.path.join(class_dir, '*.mat')))
            
            if samples_per_class is not None and len(mat_files) > samples_per_class:
                np.random.shuffle(mat_files)
                mat_files = mat_files[:samples_per_class]
            
            print(f"  {class_name}: {len(mat_files)} files")
            
            success_count = 0
            for mat_file in tqdm(mat_files, desc=f"    {class_name}", leave=False):
                try:
                    voltage, _ = load_mat_file(mat_file)
                    
                    if normalize:
                        voltage = normalize_signal(voltage)
                        
                    features = extract_statistical_features(voltage)
                    
                    all_features.append(features)
                    all_labels.append(class_idx)
                    success_count += 1
                    
                except Exception as e:
                    if success_count == 0:
                        print(f"      Error: {os.path.basename(mat_file)}: {e}")
                    continue
            
            print(f"    Processed: {success_count}/{len(mat_files)}")
        
        if len(all_features) > 0:
            X_splits[split] = np.vstack(all_features)
            y_splits[split] = np.array(all_labels)
            print(f"\n  {split} total: {len(y_splits[split])} samples, shape: {X_splits[split].shape}")
        else:
            X_splits[split] = np.array([]).reshape(0, 5)
            y_splits[split] = np.array([])
            print(f"\n  {split}: No samples processed!")
    
    return X_splits, y_splits, classes


def process_flat_dataset(data_root, classes=None, samples_per_class=None):
    """Process dataset with flat structure (classes directly under root)."""
    # Auto-detect classes if not provided
    if classes is None:
        classes = sorted([d for d in os.listdir(data_root) 
                         if os.path.isdir(os.path.join(data_root, d))])
    
    print(f"Dataset root: {data_root}")
    print(f"Classes: {classes}")
    
    # Debug: show first .mat file keys
    for class_name in classes:
        class_dir = os.path.join(data_root, class_name)
        if os.path.exists(class_dir):
            mat_files = glob(os.path.join(class_dir, '*.mat'))
            if mat_files:
                first_mat = scipy.io.loadmat(mat_files[0])
                keys = [k for k in first_mat.keys() if not k.startswith('_')]
                print(f"\n  DEBUG - {class_name} sample keys: {keys}")
                for k in keys:
                    if isinstance(first_mat[k], np.ndarray):
                        print(f"    {k}: shape={first_mat[k].shape}, dtype={first_mat[k].dtype}")
                break
    
    all_features = []
    all_labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_root, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping")
            continue
        
        mat_files = sorted(glob(os.path.join(class_dir, '*.mat')))
        
        if samples_per_class is not None and len(mat_files) > samples_per_class:
            np.random.shuffle(mat_files)
            mat_files = mat_files[:samples_per_class]
        
        print(f"\n  {class_name}: {len(mat_files)} files")
        
        success_count = 0
        for mat_file in tqdm(mat_files, desc=f"  Processing {class_name}", leave=False):
            try:
                voltage, _ = load_mat_file(mat_file)
                features = extract_statistical_features(voltage)
                
                all_features.append(features)
                all_labels.append(class_idx)
                success_count += 1
                
            except Exception as e:
                if success_count == 0:
                    print(f"    Error loading {os.path.basename(mat_file)}: {e}")
                continue
        
        print(f"    Successfully processed: {success_count}/{len(mat_files)}")
    
    if len(all_features) == 0:
        print("\n*** ERROR: No files were successfully processed! ***")
        return np.array([]).reshape(0, 5), np.array([]), classes
    
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"\nTotal samples: {len(y)}")
    print(f"Feature shape: {X.shape}")
    
    return X, y, classes


def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test.
    
    Args:
        X: Feature matrix
        y: Labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
    
    n = len(y)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def print_feature_stats(X, feature_names=None):
    """Print statistics of extracted features."""
    if feature_names is None:
        feature_names = ['Mean', 'Variance', 'Std', 'Skewness', 'Kurtosis']
    
    # Check if X is valid 2D array
    if X.size == 0 or X.ndim != 2:
        print("\nFeature Statistics: No data to display (array is empty or 1D)")
        return
    
    print("\nFeature Statistics:")
    print("-" * 50)
    for i, name in enumerate(feature_names):
        if i < X.shape[1]:
            print(f"  {name}:")
            print(f"    Min: {X[:, i].min():.6f}")
            print(f"    Max: {X[:, i].max():.6f}")
            print(f"    Mean: {X[:, i].mean():.6f}")
            print(f"    Std: {X[:, i].std():.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from 1D PD signals')
    parser.add_argument('--data_root', type=str, 
                        default='/mnt/disk2/nhatnc/res/dataset/pulse_dataset_augmented',
                        help='Root directory of 1D signal dataset')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Max samples per class (default: all)')
    parser.add_argument('--output', type=str, default='features.npz',
                        help='Output file for features')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable signal normalization (use raw amplitude)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("1D SIGNAL FEATURE EXTRACTION")
    print("="*60)
    print(f"Features: Mean, Variance, Std, Skewness, Kurtosis")
    
    if not args.no_normalize:
        print("Normalization: ENABLED (signals scaled to [0,1])")
        print("  -> Improves fairness by focusing on shape rather than amplitude")
    else:
        print("Normalization: DISABLED (using raw voltage)")
        print("  -> Warning: Model may bias towards high-amplitude signals")

    # Process dataset
    result = process_dataset(
        args.data_root,
        samples_per_class=args.samples_per_class,
        normalize=not args.no_normalize
    )
    
    # Check if result is split dataset (dict) or flat dataset (arrays)
    if isinstance(result[0], dict):
        # Split dataset structure (train/val/test already defined)
        X_splits, y_splits, class_names = result
        
        X_train, y_train = X_splits['train'], y_splits['train']
        X_val, y_val = X_splits['val'], y_splits['val']
        X_test, y_test = X_splits['test'], y_splits['test']
        
        # Print stats for training set
        print_feature_stats(X_train)
        
    else:
        # Flat dataset structure - need to split
        X, y, class_names = result
        
        # Print stats
        print_feature_stats(X)
        
        # Split dataset
        X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(y_train)}")
    print(f"  Val: {len(y_val)}")
    print(f"  Test: {len(y_test)}")
    
    # Save features
    np.savez(args.output,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             class_names=class_names)
    
    print(f"\nFeatures saved to: {args.output}")


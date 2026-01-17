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
    
    Dataset structure expected:
        data_root/
        ├── class1/*.mat
        ├── class2/*.mat
        └── class3/*.mat
    
    Args:
        data_root: Root directory of dataset
        classes: List of class names (subfolders). If None, auto-detect.
        samples_per_class: Max samples per class. If None, use all.
        
    Returns:
        X: Feature matrix (N, 5)
        y: Labels (N,)
        class_names: List of class names
    """
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
                if success_count == 0:  # Only show first error
                    print(f"    Error loading {os.path.basename(mat_file)}: {e}")
                continue
        
        print(f"    Successfully processed: {success_count}/{len(mat_files)}")
    
    if len(all_features) == 0:
        print("\n*** ERROR: No files were successfully processed! ***")
        print("Please check the .mat file format and keys above.")
        return np.array([]).reshape(0, 5), np.array([]), classes
    
    X = np.vstack(all_features)  # Ensure 2D array
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
    
    args = parser.parse_args()
    
    print("="*60)
    print("1D SIGNAL FEATURE EXTRACTION")
    print("="*60)
    print(f"Features: Mean, Variance, Std, Skewness, Kurtosis")
    
    # Process dataset
    X, y, class_names = process_dataset(
        args.data_root,
        samples_per_class=args.samples_per_class
    )
    
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

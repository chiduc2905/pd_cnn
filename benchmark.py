"""Unified Benchmark Script for PD Classification.

SYNCED WITH pd_fewshot for fair comparison:
- Same seed (42) and seed_func
- Same normalization (computed from train set only)
- Same visualization (UMAP, t-SNE, Confusion Matrix)
- Same data loading pipeline

Experiments:
- 30 training samples (10 per class) vs ALL training samples
- Test set: All test images (or val+test combined)
- Metrics: Accuracy, Precision, Recall, F1

Author: PD Analysis Team
"""
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS (same as pd_fewshot)
# =============================================================================

SEED = 42  # Same as pd_fewshot default
CLASS_MAP = {'corona': 0, 'NotPD': 1, 'surface': 2}  # Same as pd_fewshot

# Default paths
DEFAULT_SCALOGRAM_PATH = '/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_cnn/scalogram_official'
DEFAULT_1D_PATH = '/mnt/disk2/nhatnc/res/dataset/pulse_dataset_augmented'


# =============================================================================
# SEED FUNCTION (EXACT same as pd_fewshot)
# =============================================================================

def seed_func(seed=SEED):
    """Set random seeds for reproducibility (same as pd_fewshot)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASET LOADER (synced with pd_fewshot/dataset.py)
# =============================================================================

class PDScalogramBenchmark:
    """Dataset loader synced with pd_fewshot's PDScalogramPreSplit.
    
    Key features:
    - Normalization computed from train set only
    - Same shuffle seeds as pd_fewshot
    - Supports sampling N images per class
    """
    
    def __init__(self, data_path, image_size=224, samples_per_class=None):
        """
        Args:
            data_path: Path to dataset with train/val/test structure
            image_size: Input image size (default: 224 for standard pretrained models)
            samples_per_class: If set, sample this many images per class for training
        """
        self.data_path = os.path.abspath(data_path)
        self.image_size = image_size
        self.samples_per_class = samples_per_class
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # File lists
        self.train_files = []
        self.test_files = []
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        print(f'Image size: {image_size}x{image_size}')
        if samples_per_class:
            print(f'Sampling: {samples_per_class} per class')
        
        # 1. Scan folders
        self._scan_folders()
        
        # 2. Compute stats (ONLY on training data - same as pd_fewshot)
        self._compute_stats()
        
        # 3. Load images
        self._load_images()
        
        # 4. Shuffle with fixed seeds (same as pd_fewshot)
        self._shuffle_all()
    
    def _scan_folders(self):
        """Scan train/val/test folders and collect file paths."""
        # Train files
        train_path = os.path.join(self.data_path, 'train')
        for class_name in CLASS_MAP:
            class_path = os.path.join(train_path, class_name)
            if not os.path.exists(class_path):
                continue
            files = sorted([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                           and 'labeled' not in f.lower()])
            
            # Sample if needed (with fixed seed for reproducibility)
            if self.samples_per_class and len(files) > self.samples_per_class:
                rng = random.Random(SEED)
                files = rng.sample(files, self.samples_per_class)
            
            label = CLASS_MAP[class_name]
            self.train_files.extend([(os.path.join(class_path, f), label) for f in files])
        
        # Test files = val + test (combine for benchmark)
        for split in ['val', 'test']:
            split_path = os.path.join(self.data_path, split)
            if not os.path.exists(split_path):
                continue
            for class_name in CLASS_MAP:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                files = sorted([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                               and 'labeled' not in f.lower()])
                label = CLASS_MAP[class_name]
                self.test_files.extend([(os.path.join(class_path, f), label) for f in files])
        
        print(f'Found: Train={len(self.train_files)}, Test={len(self.test_files)}')
    
    def _compute_stats(self):
        """Compute per-channel mean and std using ONLY training data (same as pd_fewshot)."""
        print('Computing mean/std on training set...')
        pixels = []
        
        for fpath, _ in self.train_files:
            img = Image.open(fpath).convert('RGB')
            pixels.append(self._base_transform(img).numpy())
        
        if not pixels:
            print("Warning: No training data. Using default mean/std.")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            all_imgs = np.stack(pixels)  # (N, 3, H, W)
            self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
            self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_images(self):
        """Load images using computed normalization."""
        for fpath, label in tqdm(self.train_files, desc='Loading train'):
            img = Image.open(fpath).convert('RGB')
            self.X_train.append(self.transform(img).numpy())
            self.y_train.append(label)
        
        for fpath, label in tqdm(self.test_files, desc='Loading test'):
            img = Image.open(fpath).convert('RGB')
            self.X_test.append(self.transform(img).numpy())
            self.y_test.append(label)
        
        self.X_train = np.array(self.X_train) if self.X_train else np.array([])
        self.y_train = np.array(self.y_train) if self.y_train else np.array([])
        self.X_test = np.array(self.X_test) if self.X_test else np.array([])
        self.y_test = np.array(self.y_test) if self.y_test else np.array([])
        
        print(f'Loaded: Train={len(self.X_train)}, Test={len(self.X_test)}')
    
    def _shuffle_all(self):
        """Shuffle with fixed seeds (same as pd_fewshot)."""
        if len(self.X_train) > 0:
            idx = np.arange(len(self.X_train))
            np.random.default_rng(0).shuffle(idx)  # Same seed as pd_fewshot
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
        
        if len(self.X_test) > 0:
            idx = np.arange(len(self.X_test))
            np.random.default_rng(2).shuffle(idx)  # Same seed as pd_fewshot
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true, y_pred):
    """Compute all required metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def print_metrics(name, metrics):
    """Print metrics in formatted way."""
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1:        {metrics['f1']*100:.2f}%")


# =============================================================================
# VISUALIZATION (synced with pd_fewshot/function/function.py)
# =============================================================================

def plot_confusion_matrix(targets, preds, save_path=None, class_names=None):
    """Plot confusion matrix (IEEE format) - synced with pd_fewshot."""
    if class_names is None:
        class_names = ['Corona', 'NotPD', 'Surface']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 14
    })
    
    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens',
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 14}, vmin=0, square=True,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    
    plt.tight_layout()
    if save_path:
        base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        plt.savefig(f"{base}_cm.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{base}_cm.png", format='png', dpi=300, bbox_inches='tight')
        print(f'Saved: {base}_cm.pdf/png')
    plt.close()


def plot_umap(features, labels, save_path=None, class_names=None):
    """UMAP visualization - synced with pd_fewshot."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Run: pip install umap-learn")
        return
    
    if class_names is None:
        class_names = ['Corona', 'NotPD', 'Surface']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
    })
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric='cosine', random_state=42)
    embedded = reducer.fit_transform(features_scaled)
    
    # Rescale to [-55, 55]
    max_val = np.abs(embedded).max()
    if max_val > 0:
        embedded = embedded / max_val * 55
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    
    colors = ['#E64B35', '#3C5488', '#00A087']  # NPG colors
    unique_labels = sorted(set(labels))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[colors[i % len(colors)]], s=50, alpha=0.8,
                   marker='o', edgecolors='white', linewidths=0.6,
                   label=class_names[i] if i < len(class_names) else str(label))
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        plt.savefig(f"{base}_umap.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{base}_umap.png", format='png', dpi=300, bbox_inches='tight')
        print(f'Saved: {base}_umap.pdf/png')
    plt.close()


# =============================================================================
# CNN PIPELINE (Proposed CNN)
# =============================================================================

def run_cnn_pipeline(X_train, y_train, X_test, y_test, num_classes=3,
                     epochs=50, batch_size=32, lr=0.0005, device='cuda'):
    """Run Proposed CNN from paper."""
    from cnn_proposed import ProposedCNN
    
    print("\n" + "="*60)
    print("PIPELINE B: Proposed CNN")
    print("="*60)
    
    seed_func(SEED)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = ProposedCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    
    # Evaluate
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    metrics = compute_metrics(y_test, y_pred)
    print_metrics("Proposed CNN", metrics)
    
    # Extract features for visualization
    features = model.get_features(X_test_t).cpu().numpy()
    
    return metrics, y_pred, features, model


# =============================================================================
# TRANSFER LEARNING PIPELINE
# =============================================================================

def run_transfer_learning_pipeline(X_train, y_train, X_test, y_test,
                                    model_names=['vgg19', 'resnet50', 'densenet201'],
                                    device='cuda'):
    """Run Transfer Learning with frozen backbones + ML classifiers (SVM, RF, kNN)."""
    from transfer_learning import FeatureExtractor, train_svm_on_features, train_rf_on_features, train_knn_on_features
    
    print("\n" + "="*60)
    print("PIPELINE C: Transfer Learning + ML (SVM, RF, kNN)")
    print("="*60)
    
    results = {}
    
    for model_name in model_names:
        print(f"\n--- {model_name.upper()} ---")
        try:
            seed_func(SEED)
            extractor = FeatureExtractor(model_name, device=device)
            
            X_train_t = torch.FloatTensor(X_train).to(device)
            X_test_t = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                train_features = extractor(X_train_t).cpu().numpy()
                test_features = extractor(X_test_t).cpu().numpy()
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_features)
            test_scaled = scaler.transform(test_features)
            
            # SVM
            svm_model = train_svm_on_features(train_scaled, y_train, cv=3)
            svm_pred = svm_model.predict(test_scaled)
            svm_metrics = compute_metrics(y_test, svm_pred)
            
            # RF
            rf_model = train_rf_on_features(train_scaled, y_train, cv=3)
            rf_pred = rf_model.predict(test_scaled)
            rf_metrics = compute_metrics(y_test, rf_pred)
            
            # kNN
            knn_model = train_knn_on_features(train_scaled, y_train, cv=3)
            knn_pred = knn_model.predict(test_scaled)
            knn_metrics = compute_metrics(y_test, knn_pred)
            
            results[model_name] = {
                'SVM': svm_metrics, 'RF': rf_metrics, 'kNN': knn_metrics,
                'svm_pred': svm_pred, 'rf_pred': rf_pred, 'knn_pred': knn_pred,
                'features': test_features
            }
            
            print(f"  SVM: Acc={svm_metrics['accuracy']*100:.2f}%")
            print(f"  RF:  Acc={rf_metrics['accuracy']*100:.2f}%")
            print(f"  kNN: Acc={knn_metrics['accuracy']*100:.2f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


# =============================================================================
# ML PIPELINE (for 1D signals - 5 runs)
# =============================================================================

def run_ml_pipeline(X_train, y_train, X_test, y_test, n_runs=5, device='cuda'):
    """Run ML classifiers on 1D signal features (5 runs, mean±std).
    
    Models: ANN, SVM, RF, kNN (as per paper)
    
    Returns:
        agg: Aggregated metrics (mean±std)
        best_preds: Predictions from best run (for confusion matrix)
        features: Scaled test features
    """
    from ml_statistical import train_svm, train_random_forest, train_knn, train_ann
    
    print("\n" + "="*60)
    print("PIPELINE A: Traditional ML Classifiers (ANN, SVM, RF, kNN)")
    print(f"Running {n_runs} times...")
    print("="*60)
    
    # Standard scaling - fit on train, transform on test (FAIR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    num_classes = len(np.unique(y_train))
    results = {'ANN': [], 'SVM': [], 'RF': [], 'kNN': []}
    all_preds = {'ANN': [], 'SVM': [], 'RF': [], 'kNN': []}
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}")
        
        # ANN
        ann_model = train_ann(X_train_scaled, y_train, X_train_scaled, y_train,  # val=train for simplicity
                              num_classes=num_classes, device=device)
        ann_model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_scaled).to(device)
            ann_pred = ann_model(X_test_t).argmax(dim=1).cpu().numpy()
        results['ANN'].append(compute_metrics(y_test, ann_pred))
        all_preds['ANN'].append(ann_pred)
        
        # SVM
        svm = train_svm(X_train_scaled, y_train, cv=3)
        svm_pred = svm.predict(X_test_scaled)
        results['SVM'].append(compute_metrics(y_test, svm_pred))
        all_preds['SVM'].append(svm_pred)
        
        # RF
        rf = train_random_forest(X_train_scaled, y_train, cv=3)
        rf_pred = rf.predict(X_test_scaled)
        results['RF'].append(compute_metrics(y_test, rf_pred))
        all_preds['RF'].append(rf_pred)
        
        # kNN
        knn = train_knn(X_train_scaled, y_train, cv=3)
        knn_pred = knn.predict(X_test_scaled)
        results['kNN'].append(compute_metrics(y_test, knn_pred))
        all_preds['kNN'].append(knn_pred)
    
    # Aggregate
    agg = {}
    best_preds = {}
    for model, runs in results.items():
        agg[model] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            vals = [r[metric] for r in runs]
            agg[model][metric] = {'mean': np.mean(vals), 'std': np.std(vals)}
        
        # Get best run's predictions
        accs = [r['accuracy'] for r in runs]
        best_idx = np.argmax(accs)
        best_preds[model] = all_preds[model][best_idx]
    
    return agg, best_preds, X_test_scaled


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_full_benchmark(scalogram_path, samples_per_class, output_dir, device='cuda',
                       run_cnn=True, run_tl=True):
    """Run complete benchmark for given number of training samples."""
    
    label = f"{samples_per_class}samples" if samples_per_class else "all"
    print("\n" + "#"*70)
    print(f"# BENCHMARK: {label}")
    print("#"*70)
    
    seed_func(SEED)
    
    # Load dataset
    dataset = PDScalogramBenchmark(
        scalogram_path, 
        image_size=224,  # Standard pretrained model input size
        samples_per_class=samples_per_class
    )
    
    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test
    
    results = {
        'n_train': len(y_train),
        'n_test': len(y_test),
        'samples_per_class': samples_per_class
    }
    
    # CNN Pipeline
    if run_cnn:
        cnn_metrics, cnn_pred, cnn_features, _ = run_cnn_pipeline(
            X_train, y_train, X_test, y_test, device=device
        )
        results['cnn'] = cnn_metrics
        
        # Visualizations
        save_prefix = os.path.join(output_dir, f"cnn_{label}")
        plot_confusion_matrix(y_test, cnn_pred, save_prefix)
        plot_umap(cnn_features, y_test, save_prefix)
    
    # Transfer Learning Pipeline
    if run_tl:
        tl_results = run_transfer_learning_pipeline(
            X_train, y_train, X_test, y_test, device=device
        )
        results['tl'] = tl_results
        
        # Visualization for best TL model
        best_model = max(tl_results.keys(), 
                        key=lambda m: tl_results[m]['SVM']['accuracy'])
        save_prefix = os.path.join(output_dir, f"tl_{best_model}_{label}")
        plot_confusion_matrix(y_test, tl_results[best_model]['svm_pred'], save_prefix)
        plot_umap(tl_results[best_model]['features'], y_test, save_prefix)
# =============================================================================
# 1D SIGNAL DATASET LOADER (for ML Pipeline)
# =============================================================================

class Signal1DDataset:
    """Load 1D signals from .mat files with same fairness as pd_fewshot."""
    
    def __init__(self, data_path, samples_per_class=None):
        from feature_extraction_1d import load_mat_file, extract_statistical_features, normalize_signal
        
        self.data_path = os.path.abspath(data_path)
        self.samples_per_class = samples_per_class
        
        self.X_train, self.y_train = [], []
        self.X_test, self.y_test = [], []
        
        print(f'1D Dataset: {self.data_path}')
        if samples_per_class:
            print(f'Sampling: {samples_per_class} per class')
        
        # Load train
        train_path = os.path.join(self.data_path, 'train')
        for class_name in CLASS_MAP:
            class_path = os.path.join(train_path, class_name)
            if not os.path.exists(class_path):
                continue
            files = sorted(glob(os.path.join(class_path, '*.mat')))
            
            if self.samples_per_class and len(files) > self.samples_per_class:
                rng = random.Random(SEED)
                files = rng.sample(files, self.samples_per_class)
            
            for f in files:
                try:
                    voltage, _ = load_mat_file(f)
                    voltage = normalize_signal(voltage)
                    features = extract_statistical_features(voltage)
                    self.X_train.append(features)
                    self.y_train.append(CLASS_MAP[class_name])
                except:
                    continue
        
        # Load test (val + test combined)
        for split in ['val', 'test']:
            split_path = os.path.join(self.data_path, split)
            if not os.path.exists(split_path):
                continue
            for class_name in CLASS_MAP:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                files = sorted(glob(os.path.join(class_path, '*.mat')))
                for f in files:
                    try:
                        voltage, _ = load_mat_file(f)
                        voltage = normalize_signal(voltage)
                        features = extract_statistical_features(voltage)
                        self.X_test.append(features)
                        self.y_test.append(CLASS_MAP[class_name])
                    except:
                        continue
        
        self.X_train = np.array(self.X_train) if self.X_train else np.array([])
        self.y_train = np.array(self.y_train) if self.y_train else np.array([])
        self.X_test = np.array(self.X_test) if self.X_test else np.array([])
        self.y_test = np.array(self.y_test) if self.y_test else np.array([])
        
        # Shuffle with fixed seeds (same as pd_fewshot)
        if len(self.X_train) > 0:
            idx = np.arange(len(self.X_train))
            np.random.default_rng(0).shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
        
        if len(self.X_test) > 0:
            idx = np.arange(len(self.X_test))
            np.random.default_rng(2).shuffle(idx)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]
        
        print(f'1D Loaded: Train={len(self.X_train)}, Test={len(self.X_test)}')


# =============================================================================
# UNIFIED BENCHMARK RUNNER
# =============================================================================

def run_unified_benchmark(scalogram_path, signal1d_path, samples_per_class, output_dir, 
                          device='cuda', run_ml=True, run_cnn=True, run_tl=True):
    """Run ALL pipelines in one command with professional folder structure."""
    
    label = f"{samples_per_class}samples" if samples_per_class else "all"
    
    # Create experiment directory: output_dir/30samples/
    exp_dir = os.path.join(output_dir, label)
    os.makedirs(exp_dir, exist_ok=True)
    
    print("\n" + "#"*70)
    print(f"# UNIFIED BENCHMARK: {label}")
    print(f"# saving results to: {exp_dir}")
    print("#"*70)
    
    seed_func(SEED)
    results = {'label': label, 'samples_per_class': samples_per_class}
    
    # =========================================================================
    # PIPELINE A: ML on 1D Signals
    # =========================================================================
    if run_ml and signal1d_path:
        print("\n[PIPELINE A: ML on 1D Signals]")
        dataset_1d = Signal1DDataset(signal1d_path, samples_per_class=samples_per_class)
        
        if len(dataset_1d.X_train) > 0:
            ml_agg, ml_preds, _ = run_ml_pipeline(
                dataset_1d.X_train, dataset_1d.y_train,
                dataset_1d.X_test, dataset_1d.y_test, device=device
            )
            results['ml'] = ml_agg
            results['ml_n_train'] = len(dataset_1d.X_train)
            results['ml_n_test'] = len(dataset_1d.X_test)
            
            # Save visualizations in: output_dir/30samples/ML/{model}/
            ml_dir = os.path.join(exp_dir, 'ML')
            for model_name in ['ANN', 'SVM', 'RF', 'kNN']:
                if model_name in ml_preds:
                    model_dir = os.path.join(ml_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    save_path = os.path.join(model_dir, 'cm')
                    plot_confusion_matrix(dataset_1d.y_test, ml_preds[model_name], save_path)
    
    # =========================================================================
    # PIPELINE B & C: CNN and TL on Scalograms
    # =========================================================================
    if (run_cnn or run_tl) and scalogram_path:
        print("\n[PIPELINE B/C: CNN/TL on Scalograms]")
        dataset_img = PDScalogramBenchmark(
            scalogram_path, image_size=224, samples_per_class=samples_per_class
        )
        
        results['img_n_train'] = len(dataset_img.X_train)
        results['img_n_test'] = len(dataset_img.X_test)
        
        # CNN Pipeline
        if run_cnn:
            cnn_metrics, cnn_pred, cnn_features, _ = run_cnn_pipeline(
                dataset_img.X_train, dataset_img.y_train,
                dataset_img.X_test, dataset_img.y_test, device=device
            )
            results['cnn'] = cnn_metrics
            
            # Save in: output_dir/30samples/CNN/ProposedCNN/
            cnn_dir = os.path.join(exp_dir, 'CNN', 'ProposedCNN')
            os.makedirs(cnn_dir, exist_ok=True)
            
            plot_confusion_matrix(dataset_img.y_test, cnn_pred, os.path.join(cnn_dir, 'cm'))
            plot_umap(cnn_features, dataset_img.y_test, os.path.join(cnn_dir, 'umap'))
        
        # Transfer Learning Pipeline
        if run_tl:
            tl_results = run_transfer_learning_pipeline(
                dataset_img.X_train, dataset_img.y_train,
                dataset_img.X_test, dataset_img.y_test, device=device
            )
            results['tl'] = tl_results
            
            # Save in: output_dir/30samples/TL/{backbone}/
            tl_dir = os.path.join(exp_dir, 'TL')
            for model_name, model_results in tl_results.items():
                model_dir = os.path.join(tl_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                
                # Save plots for all classifiers (SVM, RF, kNN)
                plot_confusion_matrix(dataset_img.y_test, model_results['svm_pred'], 
                                     os.path.join(model_dir, 'svm_cm'))
                plot_confusion_matrix(dataset_img.y_test, model_results['rf_pred'], 
                                     os.path.join(model_dir, 'rf_cm'))
                plot_confusion_matrix(dataset_img.y_test, model_results['knn_pred'], 
                                     os.path.join(model_dir, 'knn_cm'))
                                     
                # UMAP is usually based on features, so it's common for all classifiers
                plot_umap(model_results['features'], dataset_img.y_test, os.path.join(model_dir, 'umap'))
    
    return results


def print_summary(results_30, results_all):
    """Print final summary table."""
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for label, results in [("30 Samples (10/class)", results_30), ("ALL Samples", results_all)]:
        print(f"\n{'='*50}")
        print(f"{label}")
        print(f"{'='*50}")
        
        if 'ml' in results:
            print(f"\n[ML] Train={results.get('ml_n_train', '?')}, Test={results.get('ml_n_test', '?')}")
            for model, metrics in results['ml'].items():
                acc = f"{metrics['accuracy']['mean']*100:.2f}±{metrics['accuracy']['std']*100:.2f}"
                f1 = f"{metrics['f1']['mean']*100:.2f}±{metrics['f1']['std']*100:.2f}"
                print(f"  {model}: Acc={acc}%, F1={f1}%")
        
        if 'cnn' in results:
            print(f"\n[CNN] Train={results.get('img_n_train', '?')}, Test={results.get('img_n_test', '?')}")
            print(f"  Acc={results['cnn']['accuracy']*100:.2f}%, F1={results['cnn']['f1']*100:.2f}%")
        
        if 'tl' in results:
            print(f"\n[Transfer Learning]")
            for model_name, model_res in results['tl'].items():
                svm_acc = model_res['SVM']['accuracy']*100
                rf_acc = model_res['RF']['accuracy']*100
                knn_acc = model_res['kNN']['accuracy']*100
                print(f"  {model_name}: SVM={svm_acc:.2f}%, RF={rf_acc:.2f}%, kNN={knn_acc:.2f}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Benchmark - All Pipelines in One Command')
    parser.add_argument('--scalogram_path', type=str, default=DEFAULT_SCALOGRAM_PATH,
                        help='Path to scalogram dataset (for CNN/TL)')
    parser.add_argument('--signal1d_path', type=str, default=DEFAULT_1D_PATH,
                        help='Path to 1D signal dataset (for ML)')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_ml', action='store_true', help='Skip ML pipeline')
    parser.add_argument('--skip_cnn', action='store_true', help='Skip CNN pipeline')
    parser.add_argument('--skip_tl', action='store_true', help='Skip Transfer Learning')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("UNIFIED PD CLASSIFICATION BENCHMARK")
    print("="*70)
    print(f"Scalogram: {args.scalogram_path}")
    print(f"1D Signals: {args.signal1d_path}")
    print(f"Seed: {SEED} (same as pd_fewshot)")
    print(f"Output: {args.output_dir}")
    print(f"Pipelines: ML={'✓' if not args.skip_ml else '✗'}, "
          f"CNN={'✓' if not args.skip_cnn else '✗'}, "
          f"TL={'✓' if not args.skip_tl else '✗'}")
    
    # Experiment 1: 10 samples per class (30 total)
    results_30 = run_unified_benchmark(
        args.scalogram_path, args.signal1d_path, samples_per_class=10,
        output_dir=args.output_dir, device=args.device,
        run_ml=not args.skip_ml, run_cnn=not args.skip_cnn, run_tl=not args.skip_tl
    )
    
    # Experiment 2: ALL training samples
    results_all = run_unified_benchmark(
        args.scalogram_path, args.signal1d_path, samples_per_class=None,
        output_dir=args.output_dir, device=args.device,
        run_ml=not args.skip_ml, run_cnn=not args.skip_cnn, run_tl=not args.skip_tl
    )
    
    # Final Summary
    print_summary(results_30, results_all)


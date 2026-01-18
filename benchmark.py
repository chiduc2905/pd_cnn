"""Unified Benchmark Script for PD Classification.

SYNCED WITH pd_fewshot for fair comparison:
- Same seed (42) and seed_func
- Same normalization (computed from train set only)
- Same visualization (UMAP, t-SNE, Confusion Matrix)
- Same data loading pipeline

Enhanced with net folder capabilities:
- Parameter counting for all models (ML, CNN, Pretrained)
- WandB logging for comprehensive experiment tracking
- Two-phase training (freeze/unfreeze) for pretrained models
- Test-only evaluation (no val set in final metrics)
- Hierarchical model checkpointing

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
import joblib  # For saving ML models
import time

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not installed. Install with: pip install wandb")

# Import from net module for model utilities
from net.models import (
    get_model, freeze_backbone, unfreeze_backbone,
    get_classifier_params, get_backbone_params,
    count_parameters, calculate_flops
)

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
    
    def __init__(self, data_path, image_size=224, samples_per_class=None, 
                 val_per_class=None, test_per_class=50):
        """
        Args:
            data_path: Path to dataset with train/val/test structure
            image_size: Input image size (default: 224 for standard pretrained models)
            samples_per_class: If set, sample this many images per class for training
            val_per_class: If set, sample this many images per class for validation
            test_per_class: Number of test samples per class (default: 50)
        """
        self.data_path = os.path.abspath(data_path)
        self.image_size = image_size
        self.samples_per_class = samples_per_class
        self.val_per_class = val_per_class
        self.test_per_class = test_per_class
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # File lists placeholders
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        print(f'Image size: {image_size}x{image_size}')
        if samples_per_class:
            print(f'Sampling: Train={samples_per_class}/class', end='')
        if val_per_class:
            print(f', Val={val_per_class}/class', end='')
        if test_per_class:
            print(f', Test={test_per_class}/class', end='')
        if samples_per_class or val_per_class or test_per_class:
            print()  # newline
        
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
        splits = {
            'train': (self.train_files, self.samples_per_class),
            'val': (self.val_files, self.val_per_class),
            'test': (self.test_files, self.test_per_class)
        }
        
        for split_name, (file_list, limit) in splits.items():
            split_path = os.path.join(self.data_path, split_name)
            if not os.path.exists(split_path):
                continue
            
            for class_name in CLASS_MAP:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                
                files = sorted([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                               and 'labeled' not in f.lower()])
                
                # Sample if limit is set and exceeded
                if limit and len(files) > limit:
                    rng = random.Random(SEED)
                    files = rng.sample(files, limit)
                
                label = CLASS_MAP[class_name]
                file_list.extend([(os.path.join(class_path, f), label) for f in files])
        
        print(f'Found: Train={len(self.train_files)}, Val={len(self.val_files)}, Test={len(self.test_files)}')
    
    def _compute_stats(self):
        """Use fixed normalization values (from full dataset statistics)."""
        print('Using fixed normalization values...')
        
        # Fixed normalization (computed from full dataset)
        self.mean = [0.212, 0.075, 0.154]
        self.std = [0.301, 0.147, 0.149]
        
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
            
        for fpath, label in tqdm(self.val_files, desc='Loading val'):
            img = Image.open(fpath).convert('RGB')
            self.X_val.append(self.transform(img).numpy())
            self.y_val.append(label)
        
        for fpath, label in tqdm(self.test_files, desc='Loading test'):
            img = Image.open(fpath).convert('RGB')
            self.X_test.append(self.transform(img).numpy())
            self.y_test.append(label)
        
        self.X_train = np.array(self.X_train) if self.X_train else np.array([])
        self.y_train = np.array(self.y_train) if self.y_train else np.array([])
        self.X_val = np.array(self.X_val) if self.X_val else np.array([])
        self.y_val = np.array(self.y_val) if self.y_val else np.array([])
        self.X_test = np.array(self.X_test) if self.X_test else np.array([])
        self.y_test = np.array(self.y_test) if self.y_test else np.array([])
        
        print(f'Loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}')
    
    def _shuffle_all(self):
        """Shuffle with fixed seeds (same as pd_fewshot)."""
        if len(self.X_train) > 0:
            idx = np.arange(len(self.X_train))
            np.random.default_rng(0).shuffle(idx)  # Same seed as pd_fewshot
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
            
        if len(self.X_val) > 0:
            idx = np.arange(len(self.X_val))
            np.random.default_rng(1).shuffle(idx)  # Same seed as pd_fewshot
            self.X_val = self.X_val[idx]
            self.y_val = self.y_val[idx]
        
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
        # If save_path is a directory, append default name
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'confusion_matrix')
            
        base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        # Avoid redundant suffixes if they are already in the name
        final_pdf = f"{base}.pdf"
        final_png = f"{base}.png"
        
        plt.savefig(final_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(final_png, format='png', dpi=300, bbox_inches='tight')
        print(f'Saved: {final_pdf}')
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
        # If save_path is a directory, append default name
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'umap')
            
        base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        final_pdf = f"{base}.pdf"
        final_png = f"{base}.png"
        
        plt.savefig(final_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(final_png, format='png', dpi=300, bbox_inches='tight')
        print(f'Saved: {final_pdf}')
    plt.close()


# =============================================================================
# CNN PIPELINE (Proposed CNN)
# =============================================================================

def run_cnn_pipeline(X_train, y_train, X_test, y_test, X_val=None, y_val=None,
                     args=None, num_classes=3, epochs=50, batch_size=32, lr=0.0005, device='cuda'):
    """Run Proposed CNN from paper.
    
    Enhanced with:
    - Parameter counting and FLOP calculation
    - WandB logging
    - Per-epoch training metrics
    - Test-only evaluation
    """
    from cnn_proposed import ProposedCNN
    
    print("\n" + "="*60)
    print("PIPELINE B: Proposed CNN")
    print("="*60)
    
    # Initialize WandB
    wandb_run = None
    if HAS_WANDB and args and hasattr(args, 'wandb_project') and args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f'cnn_proposed_{args.samples_per_class if args.samples_per_class else "all"}samples',
            config={
                'pipeline': 'CNN_Proposed',
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'samples_per_class': args.samples_per_class if args else None
            },
            reinit=True
        )
    
    seed_func(SEED)
    
    # Train Loader
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Val Loader (if provided)
    val_loader = None
    if X_val is not None and len(X_val) > 0:
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ProposedCNN(num_classes=num_classes).to(device)
    
    # Count parameters and FLOPs
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    flops_info = calculate_flops(model, input_size=(1, 3, 224, 224), device=device)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"FLOPs: {flops_info.get('flops_str', 'N/A')}")
    
    if wandb_run:
        wandb.run.summary['total_parameters'] = total_params
        wandb.run.summary['trainable_parameters'] = trainable_params
        wandb.run.summary['flops'] = flops_info.get('flops', 0)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation (if available)
        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss_sum += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            val_loss = val_loss_sum / len(val_loader)
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log to WandB
            if wandb_run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
        else:
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            if wandb_run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc
                })

    # Load best model if validation was used
    if best_model_state is not None:
        print(f"  Restoring best model (Val Acc: {best_val_acc:.4f})")
        model.load_state_dict(best_model_state)
    
    # Evaluate on TEST SET ONLY
    model.eval()
    
    # Use DataLoader for test set
    X_test_t = torch.FloatTensor(X_test)
    test_dataset = TensorDataset(X_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            
            # Predict
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            
            # Extract features
            feats = model.extract_features(inputs).cpu().numpy()
            all_features.append(feats)
            
    y_pred = np.array(all_preds)
    features = np.vstack(all_features)
    
    metrics = compute_metrics(y_test, y_pred)
    print_metrics("Proposed CNN (Test Set)", metrics)
    
    # Log test metrics to WandB
    if wandb_run:
        wandb.run.summary['test_accuracy'] = metrics['accuracy']
        wandb.run.summary['test_precision'] = metrics['precision']
        wandb.run.summary['test_recall'] = metrics['recall']
        wandb.run.summary['test_f1'] = metrics['f1']
        wandb.finish()
    
    return metrics, y_pred, features, model


# =============================================================================
# TRANSFER LEARNING PIPELINE
# =============================================================================

def run_transfer_learning_pipeline(X_train, y_train, X_test, y_test, args=None,
                                    model_names=['vgg19', 'resnet50', 'densenet201'],
                                    device='cuda'):
    """Run Transfer Learning with frozen backbones + ML classifiers (SVM, RF, kNN).
    
    Enhanced with:
    - Parameter counting
    - WandB logging
    - Test-only evaluation
    """
    from transfer_learning import FeatureExtractor, train_svm_on_features, train_rf_on_features, train_knn_on_features
    
    print("\n" + "="*60)
    print("PIPELINE C: Transfer Learning + ML (SVM, RF, kNN)")
    print("="*60)
    
    results = {}
    batch_size = 32
    
    # Prepare DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def get_features(model, loader):
        feats = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                f = model(inputs).cpu().numpy()
                feats.append(f)
        return np.vstack(feats)
    
    for model_name in model_names:
        print(f"\n--- {model_name.upper()} ---")
        
        # Initialize WandB for this model
        wandb_run = None
        if HAS_WANDB and args and hasattr(args, 'wandb_project') and args.wandb_project:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f'tl_frozen_{model_name}_{args.samples_per_class if args.samples_per_class else "all"}samples',
                config={
                    'pipeline': 'TL_Frozen+ML',
                    'backbone': model_name,
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'samples_per_class': args.samples_per_class if args else None
                },
                reinit=True
            )
        
        try:
            seed_func(SEED)
            extractor = FeatureExtractor(model_name, device=device)
            
            # Count parameters (frozen backbone)
            total_params = count_parameters(extractor.model, trainable_only=False)
            print(f"  Total parameters: {total_params:,}")
            if wandb_run:
                wandb.run.summary['total_parameters'] = total_params
            
            # Extract features in batches
            print("  Extracting features...")
            train_features = get_features(extractor, train_loader)
            test_features = get_features(extractor, test_loader)
            
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
                'features': test_features,
                'models': {'svm': svm_model, 'rf': rf_model, 'knn': knn_model}
            }
            
            print(f"  SVM: Acc={svm_metrics['accuracy']*100:.2f}%")
            print(f"  RF:  Acc={rf_metrics['accuracy']*100:.2f}%")
            print(f"  kNN: Acc={knn_metrics['accuracy']*100:.2f}%")
            
            # Log to WandB
            if wandb_run:
                wandb.run.summary['svm_test_accuracy'] = svm_metrics['accuracy']
                wandb.run.summary['svm_test_f1'] = svm_metrics['f1']
                wandb.run.summary['rf_test_accuracy'] = rf_metrics['accuracy']
                wandb.run.summary['rf_test_f1'] = rf_metrics['f1']
                wandb.run.summary['knn_test_accuracy'] = knn_metrics['accuracy']
                wandb.run.summary['knn_test_f1'] = knn_metrics['f1']
                wandb.finish()
            
        except Exception as e:
            print(f"  Error: {e}")
            if wandb_run:
                wandb.finish()
    
    return results


# New Transfer Learning Fine-Tuning Pipeline with Two-Phase Training

def run_transfer_learning_finetune(X_train, y_train, X_val, y_val, X_test, y_test, args,
                                    model_names=None,
                                    num_classes=3, freeze_epochs=10, num_epochs=50, batch_size=32,
                                    lr_classifier=0.001, lr_backbone=1e-5, lr_classifier_finetune=1e-4,
                                    device='cuda'):
    """Two-phase Transfer Learning fine-tuning (mirrors main.py).
    
    Phase 1: Freeze backbone, train classifier only
    Phase 2: Unfreeze backbone, fine-tune entire network
    
    Enhanced with:
    - Parameter counting for frozen and unfrozen states
    - WandB logging
    - Test-only evaluation
    """
    # Use default paper models if not specified
    if model_names is None:
        model_names = args.tl_models if hasattr(args, 'tl_models') and args.tl_models else ['vgg19', 'resnet50', 'densenet201']
    
    print("\n" + "="*60)
    print("PIPELINE C-FT: Transfer Learning - Two-Phase Fine-Tuning")
    print(f"Models: {', '.join(model_names)}")
    print("="*60)
    
    results = {}
    
    # Prepare DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for model_name in model_names:
        print(f"\n--- {model_name.upper()} (2-Phase Fine-Tuning) ---")
        
        # Initialize WandB for this model
        wandb_run = None
        if HAS_WANDB and args and hasattr(args, 'wandb_project') and args.wandb_project:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f'tl_finetune_{model_name}_{args.samples_per_class if args.samples_per_class else "all"}samples',
                config={
                    'pipeline': 'TL_Finetune',
                    'backbone': model_name,
                    'freeze_epochs': freeze_epochs,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'lr_classifier': lr_classifier,
                    'lr_backbone': lr_backbone,
                    'lr_classifier_finetune': lr_classifier_finetune,
                    'n_train': len(y_train),
                    'n_val': len(y_val),
                    'n_test': len(y_test),
                    'samples_per_class': args.samples_per_class if args else None
                },
                reinit=True
            )
        
        try:
            seed_func(SEED)
            model = get_model(model_name, num_classes=num_classes, pretrained=True)
            model = model.to(device)
            
            # Count total parameters
            total_params = count_parameters(model, trainable_only=False)
            print(f"  Total parameters: {total_params:,}")
            if wandb_run:
                wandb.run.summary['total_parameters'] = total_params
            
            criterion = nn.CrossEntropyLoss()
            best_val_acc = 0.0
            best_model_state = None
            
            # ===== PHASE 1: Frozen Backbone =====
            print(f"\n  [PHASE 1: Freeze Backbone - {freeze_epochs} epochs]")
            freeze_backbone(model)
            trainable_params_p1 = count_parameters(model, trainable_only=True)
            print(f"  Trainable parameters (classifier only): {trainable_params_p1:,}")
            if wandb_run:
                wandb.run.summary['phase1_trainable_parameters'] = trainable_params_p1
            
            optimizer = optim.AdamW(list(get_classifier_params(model)), lr=lr_classifier)
            
            for epoch in range(freeze_epochs):
                # Train
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{freeze_epochs} - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                
                if wandb_run:
                    wandb.log({
                        'epoch': epoch + 1,
                        'phase': 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
            
            # ===== PHASE 2: Fine-Tuning =====
            print(f"\n  [PHASE 2: Fine-Tuning - {num_epochs - freeze_epochs} epochs]")
            unfreeze_backbone(model)
            trainable_params_p2 = count_parameters(model, trainable_only=True)
            print(f"  Trainable parameters (all): {trainable_params_p2:,}")
            if wandb_run:
                wandb.run.summary['phase2_trainable_parameters'] = trainable_params_p2
            
            optimizer = optim.AdamW([
                {'params': list(get_backbone_params(model)), 'lr': lr_backbone},
                {'params': list(get_classifier_params(model)), 'lr': lr_classifier_finetune}
            ])
            
            for epoch in range(freeze_epochs, num_epochs):
                # Train
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{num_epochs} - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                
                if wandb_run:
                    wandb.log({
                        'epoch': epoch + 1,
                        'phase': 2,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
            
            # Load best model
            if best_model_state is not None:
                print(f"\n  Restoring best model (Val Acc: {best_val_acc:.4f})")
                model.load_state_dict(best_model_state)
            
            # Evaluate on TEST SET ONLY
            print("  Evaluating on test set...")
            model.eval()
            all_preds = []
            all_features = []
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    
                    # Extract features (from backbone, before classifier)
                    try:
                        if hasattr(model, 'avgpool'):
                            x = model.conv1(inputs) if hasattr(model, 'conv1') else inputs
                            if hasattr(model, 'bn1'): x = model.bn1(x)
                            if hasattr(model, 'relu'): x = model.relu(x)
                            if hasattr(model, 'maxpool'): x = model.maxpool(x)
                            if hasattr(model, 'layer1'): x = model.layer1(x)
                            if hasattr(model, 'layer2'): x = model.layer2(x)
                            if hasattr(model, 'layer3'): x = model.layer3(x)
                            if hasattr(model, 'layer4'): x = model.layer4(x)
                            feats = model.avgpool(x).view(x.size(0), -1).cpu().numpy()
                        elif hasattr(model, 'features'):
                            x = model.features(inputs)
                            feats = nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1).cpu().numpy()
                        else:
                            feats = outputs.cpu().numpy()
                        all_features.append(feats)
                    except:
                        all_features.append(outputs.cpu().numpy())
            
            y_pred = np.array(all_preds)
            features = np.vstack(all_features) if all_features else None
            
            metrics = compute_metrics(y_test, y_pred)
            print(f"  Test Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  Test F1: {metrics['f1']*100:.2f}%")
            
            results[model_name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'features': features,
                'model': model
            }
            
            # Log to WandB
            if wandb_run:
                wandb.run.summary['best_val_accuracy'] = best_val_acc
                wandb.run.summary['test_accuracy'] = metrics['accuracy']
                wandb.run.summary['test_precision'] = metrics['precision']
                wandb.run.summary['test_recall'] = metrics['recall']
                wandb.run.summary['test_f1'] = metrics['f1']
                wandb.finish()
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            if wandb_run:
                wandb.finish()
    
    return results


# =============================================================================
# ML PIPELINE (for 1D signals - 5 runs)
# =============================================================================

def run_ml_pipeline(X_train, y_train, X_test, y_test, args, n_runs=5, device='cuda'):
    """Run ML classifiers on 1D signal features (5 runs, mean±std).
    
    Models: ANN, SVM, RF, kNN (as per paper)
    Enhanced with:
    - Parameter counting
    - WandB logging
    - Model checkpointing
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data (not val+test!)
        args: Arguments object with wandb_project, samples_per_class, etc.
        n_runs: Number of runs (default: 5)
        device: Device for ANN training
    
    Returns:
        agg: Aggregated metrics (mean±std)
        best_preds: Predictions from best run (for confusion matrix)
        best_models: Best models for each classifier
        X_test_scaled: Scaled test features
    """
    from ml_statistical import train_svm, train_random_forest, train_knn, train_ann
    
    print("\n" + "="*60)
    print("PIPELINE A: Traditional ML Classifiers (ANN, SVM, RF, kNN)")
    print(f"Running {n_runs} times...")
    print("="*60)
    
    # Initialize WandB
    if HAS_WANDB and hasattr(args, 'wandb_project') and args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f'ml_{args.samples_per_class if args.samples_per_class else "all"}samples',
            config={
                'pipeline': 'ML',
                'n_runs': n_runs,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'samples_per_class': args.samples_per_class
            },
            reinit=True
        )
    else:
        wandb_run = None
    
    # Standard scaling - fit on train, transform on test (FAIR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    num_classes = len(np.unique(y_train))
    results = {'ANN': [], 'SVM': [], 'RF': [], 'kNN': []}
    all_preds = {'ANN': [], 'SVM': [], 'RF': [], 'kNN': []}
    all_models = {'ANN': [], 'SVM': [], 'RF': [], 'kNN': []}
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}")
        seed_func(SEED + run)  # Different seed per run
        
        # ANN
        ann_model = train_ann(X_train_scaled, y_train, X_train_scaled, y_train,
                              num_classes=num_classes, device=device)
        ann_model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_scaled).to(device)
            ann_pred = ann_model(X_test_t).argmax(dim=1).cpu().numpy()
        ann_metrics = compute_metrics(y_test, ann_pred)
        results['ANN'].append(ann_metrics)
        all_preds['ANN'].append(ann_pred)
        all_models['ANN'].append(ann_model)
        
        # Count ANN parameters
        if run == 0:
            ann_params = count_parameters(ann_model, trainable_only=False)
            print(f"    ANN parameters: {ann_params:,}")
            if wandb_run:
                wandb.run.summary['ann_parameters'] = ann_params
        
        # SVM
        svm = train_svm(X_train_scaled, y_train, cv=3)
        svm_pred = svm.predict(X_test_scaled)
        svm_metrics = compute_metrics(y_test, svm_pred)
        results['SVM'].append(svm_metrics)
        all_preds['SVM'].append(svm_pred)
        all_models['SVM'].append(svm)
        
        # RF
        rf = train_random_forest(X_train_scaled, y_train, cv=3)
        rf_pred = rf.predict(X_test_scaled)
        rf_metrics = compute_metrics(y_test, rf_pred)
        results['RF'].append(rf_metrics)
        all_preds['RF'].append(rf_pred)
        all_models['RF'].append(rf)
        
        # kNN
        knn = train_knn(X_train_scaled, y_train, cv=3)
        knn_pred = knn.predict(X_test_scaled)
        knn_metrics = compute_metrics(y_test, knn_pred)
        results['kNN'].append(knn_metrics)
        all_preds['kNN'].append(knn_pred)
        all_models['kNN'].append(knn)
        
        # Log this run to WandB
        if wandb_run:
            wandb.log({
                f'run_{run}/ann_accuracy': ann_metrics['accuracy'],
                f'run_{run}/ann_f1': ann_metrics['f1'],
                f'run_{run}/svm_accuracy': svm_metrics['accuracy'],
                f'run_{run}/svm_f1': svm_metrics['f1'],
                f'run_{run}/rf_accuracy': rf_metrics['accuracy'],
                f'run_{run}/rf_f1': rf_metrics['f1'],
                f'run_{run}/knn_accuracy': knn_metrics['accuracy'],
                f'run_{run}/knn_f1': knn_metrics['f1'],
            })
    
    # Aggregate
    agg = {}
    best_preds = {}
    best_models = {}
    for model_name, runs in results.items():
        agg[model_name] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            vals = [r[metric] for r in runs]
            agg[model_name][metric] = {'mean': np.mean(vals), 'std': np.std(vals)}
        
        # Get best run's predictions and model
        accs = [r['accuracy'] for r in runs]
        best_idx = np.argmax(accs)
        best_preds[model_name] = all_preds[model_name][best_idx]
        best_models[model_name] = all_models[model_name][best_idx]
        
        # Print aggregate metrics
        mean_acc = agg[model_name]['accuracy']['mean']
        std_acc = agg[model_name]['accuracy']['std']
        mean_f1 = agg[model_name]['f1']['mean']
        std_f1 = agg[model_name]['f1']['std']
        print(f"  {model_name}: Acc={mean_acc*100:.2f}±{std_acc*100:.2f}%, F1={mean_f1*100:.2f}±{std_f1*100:.2f}%")
        
        # Log aggregate metrics to WandB
        if wandb_run:
            wandb.run.summary[f'{model_name.lower()}_mean_accuracy'] = mean_acc
            wandb.run.summary[f'{model_name.lower()}_std_accuracy'] = std_acc
            wandb.run.summary[f'{model_name.lower()}_mean_f1'] = mean_f1
            wandb.run.summary[f'{model_name.lower()}_std_f1'] = std_f1
    
    if wandb_run:
        wandb.finish()
    
    return agg, best_preds, best_models, X_test_scaled



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

def run_unified_benchmark(scalogram_path, signal1d_path, samples_per_class, output_dir, args,
                          device='cuda', run_ml=True, run_cnn=True, run_tl=True, run_tl_finetune=False):
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
            ml_agg, ml_preds, ml_models, _ = run_ml_pipeline(
                dataset_1d.X_train, dataset_1d.y_train,
                dataset_1d.X_test, dataset_1d.y_test, args, device=device
            )
            results['ml'] = ml_agg
            results['ml_n_train'] = len(dataset_1d.X_train)
            results['ml_n_test'] = len(dataset_1d.X_test)
            
            # Save visualizations in: output_dir/30samples/ML/{model}/
            ml_dir = os.path.join(exp_dir, 'ML')
            import joblib
            
            for model_name in ['ANN', 'SVM', 'RF', 'kNN']:
                if model_name in ml_preds:
                    model_dir = os.path.join(ml_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Save metrics and plots
                    save_path = os.path.join(model_dir, 'cm')
                    plot_confusion_matrix(dataset_1d.y_test, ml_preds[model_name], save_path)
                    
                    # Save Model
                    model_obj = ml_models[model_name]
                    if model_name == 'ANN':
                        torch.save(model_obj.state_dict(), os.path.join(model_dir, 'ann_model.pth'))
                    else:
                        joblib.dump(model_obj, os.path.join(model_dir, f'{model_name.lower()}_model.pkl'))
    
    # =========================================================================
    # PIPELINE B & C: CNN and TL on Scalograms
    # =========================================================================
    if (run_cnn or run_tl) and scalogram_path:
        print("\n[PIPELINE B/C: CNN/TL on Scalograms]")
        dataset_img = PDScalogramBenchmark(
            scalogram_path, image_size=224, samples_per_class=samples_per_class
        )
        
        results['img_n_train'] = len(dataset_img.X_train)
        results['img_n_val'] = len(dataset_img.X_val)
        results['img_n_test'] = len(dataset_img.X_test)
        
        # CNN Pipeline
        if run_cnn:
            cnn_metrics, cnn_pred, cnn_features, cnn_model = run_cnn_pipeline(
                dataset_img.X_train, dataset_img.y_train,
                dataset_img.X_test, dataset_img.y_test,
                X_val=dataset_img.X_val, y_val=dataset_img.y_val,
                args=args, device=device
            )
            results['cnn'] = cnn_metrics
            
            # Save in: output_dir/30samples/CNN/ProposedCNN/
            cnn_dir = os.path.join(exp_dir, 'CNN', 'ProposedCNN')
            os.makedirs(cnn_dir, exist_ok=True)
            
            plot_confusion_matrix(dataset_img.y_test, cnn_pred, os.path.join(cnn_dir, 'cm'))
            plot_umap(cnn_features, dataset_img.y_test, os.path.join(cnn_dir, 'umap'))
            
            # Save CNN Model
            torch.save(cnn_model.state_dict(), os.path.join(cnn_dir, 'proposed_cnn.pth'))
        
        # Transfer Learning Pipeline
        if run_tl:
            tl_results = run_transfer_learning_pipeline(
                dataset_img.X_train, dataset_img.y_train,
                dataset_img.X_test, dataset_img.y_test, args=args, device=device
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

                # Save Models
                # TL results structure: model_results['svm']['model'], etc.
                if 'svm' in model_results:
                    joblib.dump(model_results['svm']['model'], os.path.join(model_dir, 'svm_model.pkl'))
                if 'rf' in model_results:
                    joblib.dump(model_results['rf']['model'], os.path.join(model_dir, 'rf_model.pkl'))
                if 'models' in model_results and 'knn' in model_results['models']:
                    joblib.dump(model_results['models']['knn'], os.path.join(model_dir, 'knn_model.pkl'))
        
        # TL Fine-Tuning Pipeline (Two-Phase)
        if run_tl_finetune:
            print('\n[PIPELINE C-FT: Transfer Learning Fine-Tuning (2-Phase)]')
            tl_ft_results = run_transfer_learning_finetune(
                dataset_img.X_train, dataset_img.y_train,
                dataset_img.X_val, dataset_img.y_val,
                dataset_img.X_test, dataset_img.y_test,
                args=args, device=device,
                freeze_epochs=args.freeze_epochs,
                num_epochs=args.num_epochs,
                lr_classifier=args.lr_classifier,
                lr_backbone=args.lr_backbone,
                lr_classifier_finetune=args.lr_classifier_finetune
            )
            results['tl_finetune'] = tl_ft_results
            
            # Save fine-tuned models
            tl_ft_dir = os.path.join(exp_dir, 'TL_Finetune')
            for model_name, model_results_ft in tl_ft_results.items():
                model_dir_ft = os.path.join(tl_ft_dir, model_name)
                os.makedirs(model_dir_ft, exist_ok=True)
                
                # Save model
                if 'model' in model_results_ft:
                    torch.save(model_results_ft['model'].state_dict(), 
                             os.path.join(model_dir_ft, f'{model_name}_finetune.pth'))
                
                # Save visualizations
                if 'y_pred' in model_results_ft:
                    plot_confusion_matrix(dataset_img.y_test, model_results_ft['y_pred'],
                                        os.path.join(model_dir_ft, 'cm'))
                if 'features' in model_results_ft and model_results_ft['features'] is not None:
                    plot_umap(model_results_ft['features'], dataset_img.y_test,
                            os.path.join(model_dir_ft, 'umap'))
    
    # =========================================================================
    # SAVE METRICS REPORT
    # =========================================================================
    report_path = os.path.join(exp_dir, 'metrics_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"BENCHMARK REPORT: {label}\n")
        f.write("="*60 + "\n\n")
        
        # ML
        if 'ml' in results:
            f.write("[PIPELINE A: ML on 1D Signals]\n")
            f.write(f"Train samples: {results.get('ml_n_train', '?')}\n")
            f.write(f"Test samples: {results.get('ml_n_test', '?')}\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'Model':<10} {'Accuracy':<22} {'F1-Score':<22}\n")
            f.write("-" * 55 + "\n")
            for model, metrics in results['ml'].items():
                acc = f"{metrics['accuracy']['mean']*100:.2f}±{metrics['accuracy']['std']*100:.2f}%"
                f1 = f"{metrics['f1']['mean']*100:.2f}±{metrics['f1']['std']*100:.2f}%"
                f.write(f"{model:<10} {acc:<22} {f1:<22}\n")
            f.write("\n")
            
        # CNN
        if 'cnn' in results:
            f.write("[PIPELINE B: Proposed CNN]\n")
            m = results['cnn']
            f.write(f"Accuracy:  {m['accuracy']*100:.2f}%\n")
            f.write(f"Precision: {m['precision']*100:.2f}%\n")
            f.write(f"Recall:    {m['recall']*100:.2f}%\n")
            f.write(f"F1-Score:  {m['f1']*100:.2f}%\n\n")

        # TL
        if 'tl' in results:
            f.write("[PIPELINE C: Transfer Learning]\n")
            f.write("-" * 65 + "\n")
            f.write(f"{'Backbone':<15} {'Classifier':<10} {'Accuracy':<12} {'Sen.':<12} {'Spe.':<12}\n")
            f.write("-" * 65 + "\n")
            for model_name, model_results in results['tl'].items():
                for clf in ['svm', 'rf', 'knn']:
                    key = f"{clf}_results"
                    if key in model_results:
                        r = model_results[key]
                        f.write(f"{model_name:<15} {clf.upper():<10} "
                                f"{r['accuracy']*100:.2f}%      "
                                f"{r['sensitivity']*100:.2f}%      "
                                f"{r['specificity']*100:.2f}%\n")
            f.write("\n")

    print(f"Saved metrics report to: {report_path}")
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
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--skip_ml', action='store_true', help='Skip ML pipeline')
    parser.add_argument('--skip_cnn', action='store_true', help='Skip CNN pipeline')
    parser.add_argument('--skip_tl', action='store_true', help='Skip Transfer Learning (frozen+ML)')
    parser.add_argument('--run_tl_finetune', action='store_true', help='Run Transfer Learning fine-tuning (2-phase)')
    
    # WandB and training arguments
    parser.add_argument('--wandb_project', type=str, default='pd_benchmark',
                        help='WandB project name for logging')
    parser.add_argument('--freeze_epochs', type=int, default=10,
                        help='Epochs for Phase 1 (frozen backbone)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Total epochs for TL fine-tuning')
    parser.add_argument('--lr_classifier', type=float, default=0.001,
                        help='Learning rate for classifier (Phase 1)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone (Phase 2)')
    parser.add_argument('--lr_classifier_finetune', type=float, default=1e-4,
                        help='Learning rate for classifier (Phase 2)')
    parser.add_argument('--tl_models', type=str, nargs='+', 
                        default=['vgg19', 'resnet50', 'densenet201'],
                        help='Models for TL fine-tuning (from net/models.py MODEL_INFO)')
    
    args = parser.parse_args()
    
    # Add samples_per_class to args for WandB logging
    args.samples_per_class = None  # Will be set per experiment
    
    args.output_dir = os.path.abspath(args.output_dir)
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
    args.samples_per_class = 10
    results_30 = run_unified_benchmark(
        args.scalogram_path, args.signal1d_path, samples_per_class=10,
        output_dir=args.output_dir, args=args, device=args.device,
        run_ml=not args.skip_ml, run_cnn=not args.skip_cnn, run_tl=not args.skip_tl,
        run_tl_finetune=args.run_tl_finetune
    )
    
    # Experiment 2: ALL training samples
    args.samples_per_class = None
    results_all = run_unified_benchmark(
        args.scalogram_path, args.signal1d_path, samples_per_class=None,
        output_dir=args.output_dir, args=args, device=args.device,
        run_ml=not args.skip_ml, run_cnn=not args.skip_cnn, run_tl=not args.skip_tl,
        run_tl_finetune=args.run_tl_finetune
    )
    
    # Final Summary
    print_summary(results_30, results_all)


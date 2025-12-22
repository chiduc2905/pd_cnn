"""PD Scalogram Classification - Standard CNN Training.

Training pipeline for CNN-based PD classification using pretrained or scratch models.
Implements proper transfer learning with 2-phase training:
  - Phase 1: Feature extraction (frozen backbone, train classifier only)
  - Phase 2: Fine-tuning (unfreeze backbone, discriminative learning rates)

Includes WandB logging, confusion matrix, t-SNE visualization.
"""
import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import norm
import wandb

from dataset import load_dataset
from net.models import (
    get_model, get_available_models, count_parameters, calculate_flops,
    freeze_backbone, unfreeze_backbone, freeze_partial_backbone,
    get_classifier_params, get_backbone_params
)
from function.function import plot_confusion_matrix, plot_tsne
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PD Scalogram CNN Classification')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='./scalogram/')
    parser.add_argument('--path_weights', type=str, default='checkpoints/')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--weights', type=str, default=None, help='Checkpoint for testing')
    
    # Model
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=get_available_models(),
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16, suitable for few-shot scenarios)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    
    # Transfer learning settings
    parser.add_argument('--freeze_epochs', type=int, default=20,
                        help='Epochs to train with frozen backbone (Phase 1, default: 20)')
    parser.add_argument('--lr_classifier', type=float, default=1e-3,
                        help='Learning rate for classifier during Phase 1 (default: 1e-3)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone during Phase 2 (default: 1e-5)')
    parser.add_argument('--lr_classifier_finetune', type=float, default=1e-4,
                        help='Learning rate for classifier during Phase 2 (default: 1e-4)')
    
    # Regularization
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'plateau'],
                        help='LR scheduler type')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR')
    
    # Data
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (default: 224 for ImageNet models)')
    parser.add_argument('--training_samples', type=int, default=None,
                        help='Limit training samples per class (18=6/class, 60=20/class)')
    parser.add_argument('--val_per_class', type=int, default=60,
                        help='Validation samples per class (default: 60, total: 180)')
    parser.add_argument('--test_per_class', type=int, default=60,
                        help='Test samples per class (default: 60, total: 180)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (default: 0)')
    parser.add_argument('--project', type=str, default='pd_cnn',
                        help='WandB project name')
    
    # Episode-based evaluation (for fair benchmark with few-shot)
    parser.add_argument('--eval_mode', type=str, default='episode',
                        choices=['standard', 'episode'],
                        help='Evaluation mode: standard (batch) or episode (few-shot style, default)')
    parser.add_argument('--episode_num_val', type=int, default=200,
                        help='Number of validation episodes (default: 200)')
    parser.add_argument('--episode_num_test', type=int, default=300,
                        help='Number of test episodes (default: 300)')
    parser.add_argument('--query_per_class', type=int, default=1,
                        help='Query samples per class per episode (default: 1)')
    parser.add_argument('--shot_list', type=str, default='1,5',
                        help='Comma-separated list of shot settings to evaluate (default: 1,5)')
    
    # Experiment tracking
    parser.add_argument('--experiment_id', type=int, default=None,
                        help='Experiment ID for checkpoint naming (set by run_experiments.py)')
    
    return parser.parse_args()


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Episode-Based Evaluation (for fair benchmark with few-shot)
# =============================================================================

class EpisodeGenerator:
    """Generate episodes from a dataset for episode-based evaluation.
    
    Each episode samples query_per_class images per class randomly.
    This allows fair comparison with few-shot models that also evaluate per-episode.
    """
    
    def __init__(self, X, y, num_classes, query_per_class=1, seed=42):
        """
        Args:
            X: Images array (N, C, H, W)
            y: Labels array (N,)
            num_classes: Number of classes
            query_per_class: Number of query samples per class per episode
            seed: Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.query_per_class = query_per_class
        self.rng = np.random.default_rng(seed)
        
        # Group indices by class
        self.class_indices = {}
        for c in range(num_classes):
            self.class_indices[c] = np.where(y == c)[0]
            
    def sample_episode(self):
        """Sample one episode with query_per_class samples per class.
        
        Returns:
            query_X: Images for this episode (num_classes * query_per_class, C, H, W)
            query_y: Labels for this episode (num_classes * query_per_class,)
        """
        query_X, query_y = [], []
        for c in range(self.num_classes):
            # Randomly sample query_per_class indices from this class
            indices = self.rng.choice(
                self.class_indices[c], 
                self.query_per_class, 
                replace=False
            )
            query_X.append(self.X[indices])
            query_y.extend([c] * self.query_per_class)
        return np.vstack(query_X), np.array(query_y)


def evaluate_by_episode(model, X, y, num_episodes, args, device):
    """Evaluate model using episode-based evaluation (few-shot style).
    
    This allows fair comparison with few-shot models by:
    - Sampling random episodes from test/val set
    - Computing accuracy per episode
    - Reporting mean ± std with 95% CI
    
    Args:
        model: PyTorch model
        X: Images array (N, C, H, W)
        y: Labels array (N,)
        num_episodes: Number of episodes to evaluate
        args: Arguments containing num_classes, query_per_class
        device: torch device
        
    Returns:
        mean_acc: Mean accuracy over episodes
        std_acc: Standard deviation of accuracy
        ci95: 95% confidence interval
        all_preds: All predictions (for confusion matrix)
        all_labels: All ground truth labels
        all_features: All features (for t-SNE)
    """
    model.eval()
    generator = EpisodeGenerator(X, y, args.num_classes, args.query_per_class, args.seed)
    
    episode_accs = []
    all_preds, all_labels = [], []
    all_features = []
    
    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc='Episode Evaluation', leave=False):
            query_X, query_y = generator.sample_episode()
            query_X_tensor = torch.from_numpy(query_X.astype(np.float32)).to(device)
            query_y_tensor = torch.from_numpy(query_y).long().to(device)
            
            outputs = model(query_X_tensor)
            _, predicted = outputs.max(1)
            
            # Per-episode accuracy
            correct = predicted.eq(query_y_tensor).sum().item()
            episode_accs.append(correct / len(query_y))
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(query_y)
            
            # Extract features for t-SNE
            features = extract_features(model, query_X_tensor)
            all_features.append(features.cpu().numpy())
    
    mean_acc = np.mean(episode_accs)
    std_acc = np.std(episode_accs)
    ci95 = 1.96 * std_acc / np.sqrt(num_episodes)
    
    all_features = np.vstack(all_features) if all_features else None
    
    return mean_acc, std_acc, ci95, np.array(all_preds), np.array(all_labels), all_features


# =============================================================================
# K-Shot Episodic Evaluation (inference only, no weight updates)
# =============================================================================

class EpisodeFewShotGenerator:
    """Generate K-shot episodes with support + query sets for evaluation.
    
    For each episode:
    - Support set: shot_num samples per class (used to compute prototype/reference)
    - Query set: query_per_class samples per class (for evaluation)
    
    NOTE: This is for inference-only evaluation. No weight updates occur.
    """
    
    def __init__(self, X, y, num_classes, shot_num=5, query_per_class=15, seed=42):
        """
        Args:
            X: Images array (N, C, H, W)
            y: Labels array (N,)
            num_classes: Number of classes
            shot_num: Support samples per class (K-shot)
            query_per_class: Query samples per class
            seed: Random seed
        """
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.shot_num = shot_num
        self.query_per_class = query_per_class
        self.rng = np.random.default_rng(seed)
        
        # Group indices by class
        self.class_indices = {}
        for c in range(num_classes):
            self.class_indices[c] = np.where(y == c)[0]
            
    def sample_episode(self):
        """Sample one episode with support and query sets.
        
        Returns:
            support_X: Support images (num_classes * shot_num, C, H, W)
            support_y: Support labels (num_classes * shot_num,)
            query_X: Query images (num_classes * query_per_class, C, H, W)
            query_y: Query labels (num_classes * query_per_class,)
        """
        support_X, support_y = [], []
        query_X, query_y = [], []
        
        for c in range(self.num_classes):
            # Sample shot_num + query_per_class indices
            total_needed = self.shot_num + self.query_per_class
            indices = self.rng.choice(
                self.class_indices[c], 
                total_needed, 
                replace=False
            )
            
            # Split into support and query
            support_idx = indices[:self.shot_num]
            query_idx = indices[self.shot_num:]
            
            support_X.append(self.X[support_idx])
            support_y.extend([c] * self.shot_num)
            query_X.append(self.X[query_idx])
            query_y.extend([c] * self.query_per_class)
            
        return (np.vstack(support_X), np.array(support_y),
                np.vstack(query_X), np.array(query_y))


def evaluate_episodic_kshot(model, X, y, num_episodes, shot_num, query_per_class, 
                             num_classes, seed, device):
    """K-shot episodic evaluation (inference only, NO weight updates).
    
    For each episode:
      1. Sample K support samples per class + Q query samples per class
      2. Classify queries using the trained model (no adaptation)
      3. Compute per-episode accuracy
      4. Report mean ± std with 95% CI
    
    This is pure inference - the model weights are never updated during evaluation.
    
    Args:
        model: PyTorch model (already trained)
        X: Images array (N, C, H, W)
        y: Labels array (N,)
        num_episodes: Number of episodes to evaluate
        shot_num: Support samples per class (K-shot)
        query_per_class: Query samples per class per episode
        num_classes: Number of classes
        seed: Random seed
        device: torch device
        
    Returns:
        mean_acc: Mean accuracy over all episodes
        std_acc: Standard deviation
        ci95: 95% confidence interval (1.96 * σ / √N)
        all_preds: All predictions (for confusion matrix)
        all_labels: All ground truth labels
    """
    model.eval()
    
    generator = EpisodeFewShotGenerator(
        X, y, num_classes, 
        shot_num=shot_num, 
        query_per_class=query_per_class, 
        seed=seed
    )
    
    episode_accs = []
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc=f'{shot_num}-shot Evaluation', leave=False):
            # Sample episode
            support_X, support_y, query_X, query_y = generator.sample_episode()
            
            query_X_tensor = torch.from_numpy(query_X.astype(np.float32)).to(device)
            query_y_tensor = torch.from_numpy(query_y).long().to(device)
            
            # === Inference only: classify queries using the trained model ===
            outputs = model(query_X_tensor)
            _, predicted = outputs.max(1)
            
            # Per-episode accuracy
            correct = predicted.eq(query_y_tensor).sum().item()
            episode_accs.append(correct / len(query_y))
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(query_y)
    
    mean_acc = np.mean(episode_accs)
    std_acc = np.std(episode_accs)
    ci95 = 1.96 * std_acc / np.sqrt(num_episodes)
    
    return mean_acc, std_acc, ci95, np.array(all_preds), np.array(all_labels)


def evaluate_multishot(model, X, y, args, device, num_episodes, phase='test'):
    """Evaluate model with multiple shot settings (e.g., 1-shot and 5-shot).
    
    Args:
        model: PyTorch model
        X: Images array
        y: Labels array
        args: Arguments with shot_list, query_per_class, num_classes, seed
        device: torch device
        num_episodes: Number of episodes
        phase: 'val' or 'test' (for logging)
        
    Returns:
        results: Dict mapping shot_num -> (mean_acc, std_acc, ci95, preds, labels)
    """
    shot_list = [int(s) for s in args.shot_list.split(',')]
    results = {}
    
    print(f'\n{"="*60}')
    print(f'{phase.upper()} Episodic Evaluation ({num_episodes} episodes)')
    print(f'Shot settings: {shot_list}')
    print(f'Query per class: {args.query_per_class}')
    print(f'{"="*60}')
    
    for shot_num in shot_list:
        mean_acc, std_acc, ci95, preds, labels = evaluate_episodic_kshot(
            model, X, y, num_episodes, 
            shot_num=shot_num,
            query_per_class=args.query_per_class,
            num_classes=args.num_classes,
            seed=args.seed,
            device=device
        )
        
        results[shot_num] = (mean_acc, std_acc, ci95, preds, labels)
        
        print(f'  {shot_num}-shot: {mean_acc*100:.2f} ± {std_acc*100:.2f}% (95% CI: ±{ci95*100:.2f}%)')
    
    return results


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch.
    
    Handles InceptionV3 special case: returns (output, aux_output) tuple during training.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Check if this is InceptionV3 with aux_logits enabled
    is_inception = hasattr(model, 'AuxLogits') and hasattr(model, 'aux_logits') and model.aux_logits
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle InceptionV3 tuple output (main, aux)
        if is_inception and isinstance(outputs, tuple):
            main_output, aux_output = outputs
            loss1 = criterion(main_output, labels)
            loss2 = criterion(aux_output, labels)
            # Standard weighting: 1.0 * main + 0.4 * aux (as per original InceptionV3 paper)
            loss = loss1 + 0.4 * loss2
            outputs = main_output  # Use main output for accuracy
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*correct/total:.1f}%')
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Extract features (before classifier) for t-SNE
            # Use global average pooling on the output before classifier
            features = extract_features(model, images)
            all_features.append(features.cpu().numpy())
    
    all_features = np.vstack(all_features) if all_features else None
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels), all_features


def extract_features(model, images):
    """Extract features before the classifier layer."""
    model.eval()
    
    # Get the base model (remove classifier)
    if hasattr(model, 'features'):  # SqueezeNet, DenseNet, EfficientNet, MobileNetV3
        features = model.features(images)
        features = nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
    elif hasattr(model, 'avgpool'):  # ResNet, ShuffleNet
        # Forward through all layers except fc
        x = model.conv1(images) if hasattr(model, 'conv1') else images
        if hasattr(model, 'bn1'):
            x = model.bn1(x)
        if hasattr(model, 'relu'):
            x = model.relu(x)
        if hasattr(model, 'maxpool'):
            x = model.maxpool(x)
        if hasattr(model, 'layer1'):
            x = model.layer1(x)
        if hasattr(model, 'layer2'):
            x = model.layer2(x)
        if hasattr(model, 'layer3'):
            x = model.layer3(x)
        if hasattr(model, 'layer4'):
            x = model.layer4(x)
        features = model.avgpool(x)
        features = features.view(features.size(0), -1)
    else:
        # Fallback: just use the outputs
        features = model(images)
    
    return features


def create_optimizer_phase1(model, args):
    """Create optimizer for Phase 1 (frozen backbone, train classifier only)."""
    classifier_params = list(get_classifier_params(model))
    return optim.AdamW(classifier_params, lr=args.lr_classifier, weight_decay=args.weight_decay)


def create_optimizer_phase2(model, args):
    """Create optimizer for Phase 2 (fine-tune with discriminative LR)."""
    # Discriminative learning rates: lower LR for backbone, higher for classifier
    param_groups = [
        {'params': list(get_backbone_params(model)), 'lr': args.lr_backbone},
        {'params': list(get_classifier_params(model)), 'lr': args.lr_classifier_finetune}
    ]
    return optim.AdamW(param_groups, weight_decay=args.weight_decay)


def create_scheduler(optimizer, scheduler_type, args, num_epochs):
    """Create learning rate scheduler."""
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:  # plateau
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)


def plot_training_curves(history, save_dir, model_name, pretrained, training_samples):
    """
    Plot training/validation accuracy and loss curves.
    
    Args:
        history: dict with keys 'train_acc', 'val_acc', 'train_loss', 'val_loss'
        save_dir: directory to save plots
        model_name: model architecture name
        pretrained: whether pretrained weights were used
        training_samples: number of training samples (or None for all)
    """
    pretrained_str = 'pretrained' if pretrained else 'scratch'
    samples_str = f'_{training_samples}samples' if training_samples else '_allsamples'
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === Accuracy Plot ===
    ax1 = axes[0]
    ax1.plot(epochs, [acc * 100 for acc in history['train_acc']], 
             'b-', linewidth=2, label='Train Accuracy', marker='o', markersize=3)
    ax1.plot(epochs, [acc * 100 for acc in history['val_acc']], 
             'r-', linewidth=2, label='Val Accuracy', marker='s', markersize=3)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'{model_name} ({pretrained_str}) - Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim([0, 105])
    
    # Mark best val accuracy
    best_val_idx = history['val_acc'].index(max(history['val_acc']))
    best_val_acc = history['val_acc'][best_val_idx] * 100
    ax1.axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: {best_val_acc:.1f}%', 
                 xy=(best_val_idx + 1, best_val_acc),
                 xytext=(best_val_idx + 1, best_val_acc + 3),
                 fontsize=9, color='green')
    
    # === Loss Plot ===
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss'], 
             'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
    ax2.plot(epochs, history['val_loss'], 
             'r-', linewidth=2, label='Val Loss', marker='s', markersize=3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'{model_name} ({pretrained_str}) - Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Mark best val loss
    best_loss_idx = history['val_loss'].index(min(history['val_loss']))
    best_val_loss = history['val_loss'][best_loss_idx]
    ax2.axhline(y=best_val_loss, color='g', linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: {best_val_loss:.4f}', 
                 xy=(best_loss_idx + 1, best_val_loss),
                 xytext=(best_loss_idx + 1, best_val_loss + 0.05),
                 fontsize=9, color='green')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'training_curves_{model_name}_{pretrained_str}{samples_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'Training curves saved to {save_path}')
    return save_path


def train(model, train_loader, val_loader, args, device):
    """
    Full training loop with 2-phase transfer learning.
    
    Phase 1: Feature Extraction (frozen backbone)
        - Train only classifier head
        - Higher learning rate (1e-3)
        
    Phase 2: Fine-tuning (unfrozen backbone)  
        - Train entire network
        - Discriminative LR: backbone (1e-5), classifier (1e-4)
    """
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    patience_counter = 0
    current_phase = 1
    
    # History tracking for plotting
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # =========================================================================
    # Phase 1: Feature Extraction (Frozen Backbone)
    # =========================================================================
    if args.pretrained and args.freeze_epochs > 0:
        print(f'\n{"="*60}')
        print(f'PHASE 1: Feature Extraction (Frozen Backbone)')
        print(f'Epochs: 1-{args.freeze_epochs} | LR: {args.lr_classifier}')
        print(f'{"="*60}\n')
        
        freeze_backbone(model)
        trainable_params = count_parameters(model, trainable_only=True)
        print(f'Trainable parameters (classifier only): {trainable_params:,}')
        
        optimizer = create_optimizer_phase1(model, args)
        scheduler = create_scheduler(optimizer, args.scheduler, args, args.freeze_epochs)
        
        for epoch in range(1, args.freeze_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            
            if args.scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_val_gap = train_acc - val_acc
            
            # Track history
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f'[P1] Epoch {epoch:3d}/{args.freeze_epochs} | '
                  f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} (gap={train_val_gap:+.4f}) | '
                  f'LR: {current_lr:.2e}')
            
            wandb.log({
                'epoch': epoch,
                'phase': 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_val_gap': train_val_gap,
                'lr': current_lr
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                _save_checkpoint(model, optimizer, epoch, val_acc, args, phase=1)
                print(f'  → Best model saved ({val_acc:.4f})')
                wandb.run.summary['best_val_acc'] = best_val_acc
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at Phase 1, epoch {epoch}')
                break
    
    # =========================================================================
    # Phase 2: Fine-tuning (Unfrozen Backbone)
    # =========================================================================
    phase2_epochs = args.num_epochs - args.freeze_epochs if args.pretrained else args.num_epochs
    start_epoch = args.freeze_epochs + 1 if args.pretrained else 1
    
    if phase2_epochs > 0:
        current_phase = 2
        print(f'\n{"="*60}')
        if args.pretrained:
            print(f'PHASE 2: Fine-tuning (Unfrozen Backbone)')
            print(f'Epochs: {start_epoch}-{args.num_epochs} | LR: backbone={args.lr_backbone}, classifier={args.lr_classifier_finetune}')
        else:
            print(f'Training from Scratch')
            print(f'Epochs: 1-{args.num_epochs}')
        print(f'{"="*60}\n')
        
        unfreeze_backbone(model)
        trainable_params = count_parameters(model, trainable_only=True)
        print(f'Trainable parameters (all): {trainable_params:,}')
        
        if args.pretrained:
            optimizer = create_optimizer_phase2(model, args)
        else:
            # From scratch: use single LR for all params
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay)
        
        scheduler = create_scheduler(optimizer, args.scheduler, args, phase2_epochs)
        
        # Reset patience for phase 2 if we had early stopping in phase 1
        if patience_counter >= args.patience:
            patience_counter = 0
        
        for epoch in range(start_epoch, args.num_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            
            if args.scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            # Get LR (for discriminative LR, show backbone LR)
            current_lr = optimizer.param_groups[0]['lr']
            train_val_gap = train_acc - val_acc
            
            # Track history
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            phase_str = 'P2' if args.pretrained else 'TR'
            print(f'[{phase_str}] Epoch {epoch:3d}/{args.num_epochs} | '
                  f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} (gap={train_val_gap:+.4f}) | '
                  f'LR: {current_lr:.2e}')
            
            wandb.log({
                'epoch': epoch,
                'phase': 2 if args.pretrained else 0,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_val_gap': train_val_gap,
                'lr': current_lr
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                _save_checkpoint(model, optimizer, epoch, val_acc, args, phase=2)
                print(f'  → Best model saved ({val_acc:.4f})')
                wandb.run.summary['best_val_acc'] = best_val_acc
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at Phase 2, epoch {epoch} (patience={args.patience})')
                break
    
    # Plot training curves and save locally
    if len(history['train_acc']) > 0:
        curves_path = plot_training_curves(
            history, args.path_results, args.model, args.pretrained, args.training_samples
        )
        # Also log to wandb
        wandb.log({'training_curves': wandb.Image(curves_path)})
    
    return best_val_acc, history


def _save_checkpoint(model, optimizer, epoch, val_acc, args, phase=1):
    """Save model checkpoint with experiment ID if available."""
    pretrained_str = 'pretrained' if args.pretrained else 'scratch'
    samples_str = f'_{args.training_samples}samples' if args.training_samples else '_allsamples'
    
    # Include experiment ID in filename if provided
    if args.experiment_id:
        filename = f'exp{args.experiment_id:03d}_{args.model}_{pretrained_str}{samples_str}_best.pth'
    else:
        filename = f'{args.model}_{pretrained_str}{samples_str}_best.pth'
    
    save_path = os.path.join(args.path_weights, filename)
    torch.save({
        'epoch': epoch,
        'phase': phase,
        'experiment_id': args.experiment_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': vars(args)
    }, save_path)
    print(f'  → Checkpoint saved: {filename}')


# =============================================================================
# Testing
# =============================================================================

def calculate_p_value(acc, baseline, n):
    """Z-test for proportion significance."""
    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def test(model, test_loader, args, device, test_X=None, test_y=None):
    """Final evaluation on test set with visualizations.
    
    Supports two modes:
    - standard: Run through entire test set once
    - episode: Multi-shot episodic evaluation (1-shot and 5-shot, 300 episodes each)
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for standard mode
        args: Arguments
        device: torch device
        test_X: Test images array (required for episode mode)
        test_y: Test labels array (required for episode mode)
    """
    criterion = nn.CrossEntropyLoss()
    
    pretrained_str = 'pretrained' if args.pretrained else 'scratch'
    samples_str = f'_{args.training_samples}samples' if args.training_samples else '_allsamples'
    
    if args.eval_mode == 'episode' and test_X is not None:
        # Multi-shot episodic evaluation (1-shot and 5-shot)
        results = evaluate_multishot(
            model, test_X, test_y, args, device, 
            num_episodes=args.episode_num_test, 
            phase='test'
        )
        
        shot_list = [int(s) for s in args.shot_list.split(',')]
        
        # Log results for each shot setting
        for shot_num in shot_list:
            mean_acc, std_acc, ci95, all_preds, all_labels = results[shot_num]
            
            # Metrics
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, 
                labels=list(range(args.num_classes)),
                average='macro', 
                zero_division=0
            )
            p_val = calculate_p_value(mean_acc, 1.0/args.num_classes, len(all_labels))
            
            # Log to WandB
            wandb.log({
                f'test_{shot_num}shot_accuracy': mean_acc,
                f'test_{shot_num}shot_accuracy_std': std_acc,
                f'test_{shot_num}shot_accuracy_ci95': ci95,
                f'test_{shot_num}shot_precision': prec,
                f'test_{shot_num}shot_recall': rec,
                f'test_{shot_num}shot_f1': f1,
            })
            
            # Update summary
            wandb.run.summary[f'{shot_num}shot_accuracy'] = mean_acc
            wandb.run.summary[f'{shot_num}shot_accuracy_std'] = std_acc
            wandb.run.summary[f'{shot_num}shot_accuracy_ci95'] = ci95
            
            # Save results to text file
            txt_path = os.path.join(args.path_results,
                                    f'results_{args.model}_{pretrained_str}{samples_str}_{shot_num}shot.txt')
            with open(txt_path, 'w') as f:
                f.write(f'Model: {args.model}\n')
                f.write(f'Pretrained: {args.pretrained}\n')
                f.write(f'Training Samples: {args.training_samples if args.training_samples else "All"}\n')
                f.write(f'Eval Mode: {shot_num}-shot episodic\n')
                f.write(f'Episodes: {args.episode_num_test}\n')
                f.write(f'Query per class: {args.query_per_class}\n')
                f.write('-' * 30 + '\n')
                f.write(f'Accuracy : {mean_acc*100:.2f} ± {std_acc*100:.2f}%\n')
                f.write(f'95% CI   : ±{ci95*100:.2f}%\n')
                f.write(f'Precision: {prec:.4f}\n')
                f.write(f'Recall   : {rec:.4f}\n')
                f.write(f'F1-Score : {f1:.4f}\n')
                f.write(f'p-value  : {p_val:.2e}\n')
            
            # Confusion Matrix
            cm_base = os.path.join(args.path_results, 
                                   f'confusion_matrix_{args.model}_{pretrained_str}{samples_str}_{shot_num}shot')
            plot_confusion_matrix(all_labels, all_preds, args.num_classes, cm_base + '.png')
            # plot_confusion_matrix saves with _1col.png suffix
            cm_png = cm_base + '_1col.png'
            if os.path.exists(cm_png):
                wandb.log({f'{shot_num}shot_confusion_matrix': wandb.Image(cm_png)})
        
        # Use 5-shot as primary metric if available, else use first shot
        primary_shot = 5 if 5 in shot_list else shot_list[0]
        test_acc = results[primary_shot][0]
        wandb.run.summary['accuracy'] = test_acc
        wandb.run.summary['eval_mode'] = 'episode'
        wandb.run.summary['shot_settings'] = args.shot_list
        
    else:
        # Standard batch evaluation
        test_loss, test_acc, all_preds, all_labels, all_features = evaluate(
            model, test_loader, criterion, device
        )
        
        # Metrics
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, 
            labels=list(range(args.num_classes)),
            average='macro', 
            zero_division=0
        )
        p_val = calculate_p_value(test_acc, 1.0/args.num_classes, len(all_labels))
        
        print(f'\n{"="*50}')
        print(f'Test Results: {args.model} | {pretrained_str}{samples_str}')
        print(f'{"="*50}')
        print(f'Accuracy : {test_acc:.4f}')
        print(f'Precision: {prec:.4f}')
        print(f'Recall   : {rec:.4f}')
        print(f'F1-Score : {f1:.4f}')
        print(f'p-value  : {p_val:.2e}')
        
        # Log final results to WandB
        wandb.log({
            'test_accuracy': test_acc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1': f1,
            'test_p_value': p_val,
            'test_loss': test_loss,
            'eval_mode': 'standard'
        })
        
        # Update summary
        wandb.run.summary['accuracy'] = test_acc
        wandb.run.summary['precision'] = prec
        wandb.run.summary['recall'] = rec
        wandb.run.summary['f1_score'] = f1
        wandb.run.summary['p_value'] = p_val
        wandb.run.summary['eval_mode'] = 'standard'
        
        # Save results to text file
        txt_path = os.path.join(args.path_results,
                                f'results_{args.model}_{pretrained_str}{samples_str}.txt')
        with open(txt_path, 'w') as f:
            f.write(f'Model: {args.model}\n')
            f.write(f'Pretrained: {args.pretrained}\n')
            f.write(f'Training Samples: {args.training_samples if args.training_samples else "All"}\n')
            f.write(f'Freeze Epochs: {args.freeze_epochs}\n')
            f.write('-' * 30 + '\n')
            f.write(f'Accuracy : {test_acc:.4f}\n')
            f.write(f'Precision: {prec:.4f}\n')
            f.write(f'Recall   : {rec:.4f}\n')
            f.write(f'F1-Score : {f1:.4f}\n')
            f.write(f'p-value  : {p_val:.2e}\n')
        
        # Confusion Matrix
        cm_base = os.path.join(args.path_results, 
                               f'confusion_matrix_{args.model}_{pretrained_str}{samples_str}')
        plot_confusion_matrix(all_labels, all_preds, args.num_classes, cm_base + '.png')
        # plot_confusion_matrix saves with _1col.png suffix
        cm_png = cm_base + '_1col.png'
        if os.path.exists(cm_png):
            wandb.log({'confusion_matrix': wandb.Image(cm_png)})
        
        # t-SNE
        if all_features is not None and len(all_features) > 0:
            tsne_base = os.path.join(args.path_results,
                                     f'tsne_{args.model}_{pretrained_str}{samples_str}')
            plot_tsne(all_features, all_labels, args.num_classes, tsne_base + '.png')
            # plot_tsne saves with _1col.png suffix
            tsne_png = tsne_base + '_1col.png'
            if os.path.exists(tsne_png):
                wandb.log({'tsne_plot': wandb.Image(tsne_png)})
    
    print(f'\nResults saved to {args.path_results}')
    
    # Generate model comparison bar chart
    log_model_comparison_bar(args)
    
    return test_acc


def log_model_comparison_bar(args):
    """Read all results files and generate model comparison bar chart."""
    import matplotlib.pyplot as plt
    
    samples_str = f'{args.training_samples}samples' if args.training_samples else 'allsamples'
    results_dir = args.path_results
    
    # Model name mapping for display
    model_display_names = {
        'squeezenet1_1': 'SqueezeNet',
        'shufflenetv2_x0_5': 'ShuffleNetV2',
        'mobilenetv3_small': 'MobileNetV3',
        'efficientnet_b0': 'EfficientNet',
        'densenet121': 'DenseNet',
        'resnet18': 'ResNet-18'
    }
    
    # Collect results
    model_results = {}
    models = get_available_models()
    
    for model in models:
        display_name = model_display_names.get(model, model)
        model_results[display_name] = {'pretrained': None, 'scratch': None}
        
        for pretrained in ['pretrained', 'scratch']:
            result_file = os.path.join(results_dir, 
                f'results_{model}_{pretrained}_{samples_str}.txt')
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    content = f.read()
                    match = re.search(r'Accuracy\s*:\s*([\d.]+)', content)
                    if match:
                        acc = float(match.group(1))
                        model_results[display_name][pretrained] = acc
    
    # Remove models with missing data
    model_results = {k: v for k, v in model_results.items() 
                     if v['pretrained'] is not None or v['scratch'] is not None}
    
    if len(model_results) < 2:
        print("Not enough results for model comparison chart.")
        return
    
    # Generate chart
    fig, ax = plt.subplots(figsize=(12, len(model_results) * 0.8 + 2))
    
    models_list = list(model_results.keys())
    acc_pretrained = [model_results[m]['pretrained'] * 100 if model_results[m]['pretrained'] else 0 for m in models_list]
    acc_scratch = [model_results[m]['scratch'] * 100 if model_results[m]['scratch'] else 0 for m in models_list]
    
    y = np.arange(len(models_list))
    height = 0.35
    
    bars_pre = ax.barh(y - height/2, acc_pretrained, height, label='Pretrained', color='#5DA5DA')
    bars_scr = ax.barh(y + height/2, acc_scratch, height, label='Scratch', color='#FAA43A')
    
    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_ylabel('Models', fontsize=14)
    ax.set_title(f'Model Comparison ({samples_str})', fontsize=16, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(models_list)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, f'model_comparison_{samples_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    wandb.log({'model_comparison_bar': wandb.Image(save_path)})
    print(f'Model comparison chart saved to {save_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    seed_everything(args.seed)
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'\nDevice: {device}')
    
    # Create directories
    os.makedirs(args.path_weights, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)
    
    # Load dataset
    print(f'\nLoading dataset from {args.dataset_path}...')
    print(f'Val: {args.val_per_class}/class, Test: {args.test_per_class}/class')
    dataset = load_dataset(
        args.dataset_path, 
        image_size=args.image_size,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class
    )
    
    # Convert to tensors
    train_X = torch.from_numpy(dataset.X_train.astype(np.float32))
    train_y = torch.from_numpy(dataset.y_train).long()
    val_X = torch.from_numpy(dataset.X_val.astype(np.float32))
    val_y = torch.from_numpy(dataset.y_val).long()
    test_X = torch.from_numpy(dataset.X_test.astype(np.float32))
    test_y = torch.from_numpy(dataset.y_test).long()
    
    print(f'Train: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}')
    
    # Limit training samples if specified
    if args.training_samples:
        per_class = args.training_samples // args.num_classes
        X_list, y_list = [], []
        
        for c in range(args.num_classes):
            idx = (train_y == c).nonzero(as_tuple=True)[0]
            if len(idx) < per_class:
                raise ValueError(f'Class {c}: need {per_class}, have {len(idx)}')
            
            g = torch.Generator().manual_seed(args.seed)
            perm = torch.randperm(len(idx), generator=g)[:per_class]
            X_list.append(train_X[idx[perm]])
            y_list.append(train_y[idx[perm]])
        
        train_X = torch.cat(X_list)
        train_y = torch.cat(y_list)
        print(f'Limited to {args.training_samples} training samples ({per_class}/class)')
    
    # Create dataloaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    print(f'\nCreating model: {args.model} (pretrained={args.pretrained})')
    model = get_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # Log model info
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Calculate FLOPs
    flops_info = calculate_flops(
        model, 
        input_size=(1, 3, args.image_size, args.image_size),
        device=str(device)
    )
    print(f'FLOPs: {flops_info["flops_str"]} (MACs: {flops_info["macs_str"]})')
    
    # Initialize WandB
    pretrained_str = 'pretrained' if args.pretrained else 'scratch'
    samples_str = f'_{args.training_samples}samples' if args.training_samples else ''
    run_name = f'{args.model}_{pretrained_str}{samples_str}'
    
    wandb.init(
        project=args.project,
        config=vars(args),
        name=run_name,
        group=run_name,
        job_type=args.mode
    )
    
    # Set config summary (not cluttering the table with per-step logs)
    wandb.run.summary['total_parameters'] = total_params
    wandb.run.summary['model_name'] = args.model
    wandb.run.summary['pretrained'] = args.pretrained
    wandb.run.summary['training_samples'] = args.training_samples if args.training_samples else 'all'
    
    # Log FLOPs to WandB
    wandb.run.summary['flops'] = flops_info['flops']
    wandb.run.summary['macs'] = flops_info['macs']
    wandb.run.summary['flops_gflops'] = flops_info['flops'] / 1e9  # GFLOPs
    wandb.run.summary['macs_gmacs'] = flops_info['macs'] / 1e9  # GMACs
    wandb.config.update({'flops': flops_info['flops'], 'macs': flops_info['macs']})
    
    print(f'\n{"="*60}')
    print(f'Model: {args.model}')
    print(f'{"="*60}')
    print(f'Total Parameters: {total_params:,}')
    print(f'Transfer Learning: {"Enabled" if args.pretrained else "Disabled (from scratch)"}')
    if args.pretrained:
        print(f'  Phase 1 (Freeze): {args.freeze_epochs} epochs, LR={args.lr_classifier}')
        print(f'  Phase 2 (Fine-tune): {args.num_epochs - args.freeze_epochs} epochs')
        print(f'    Backbone LR: {args.lr_backbone}')
        print(f'    Classifier LR: {args.lr_classifier_finetune}')
    print(f'{"="*60}\n')
    
    if args.mode == 'train':
        print(f'Training: {args.model} | {"Pretrained" if args.pretrained else "Scratch"}')
        print(f'Epochs: {args.num_epochs} | Batch: {args.batch_size}')
        print(f'{"="*60}\n')
        
        best_acc, history = train(model, train_loader, val_loader, args, device)
        
        # Load best model and test
        pretrained_str = 'pretrained' if args.pretrained else 'scratch'
        samples_str = f'_{args.training_samples}samples' if args.training_samples else '_allsamples'
        
        # Use experiment_id in filename if provided
        if args.experiment_id:
            filename = f'exp{args.experiment_id:03d}_{args.model}_{pretrained_str}{samples_str}_best.pth'
        else:
            filename = f'{args.model}_{pretrained_str}{samples_str}_best.pth'
        
        best_path = os.path.join(args.path_weights, filename)
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Pass numpy arrays for episode mode
        test(model, test_loader, args, device, 
             test_X=dataset.X_test, test_y=dataset.y_test)
        
    else:  # Test only
        if args.weights:
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Pass numpy arrays for episode mode
            test(model, test_loader, args, device,
                 test_X=dataset.X_test, test_y=dataset.y_test)
        else:
            print('Error: Please specify --weights for test mode')
    
    wandb.finish()


if __name__ == '__main__':
    main()

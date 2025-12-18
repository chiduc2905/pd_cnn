"""PD Scalogram Classification - Standard CNN Training.

Training pipeline for CNN-based PD classification using pretrained or scratch models.
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
from net.models import get_model, get_available_models, count_parameters
from function.function import plot_confusion_matrix, plot_tsne


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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for scratch training (default: 1e-3)')
    parser.add_argument('--lr_pretrained', type=float, default=1e-4,
                        help='Learning rate for pretrained fine-tuning (default: 1e-4)')
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
                        help='Limit training samples (e.g., 30 = 10/class)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (default: 0)')
    parser.add_argument('--project', type=str, default='pd_cnn',
                        help='WandB project name')
    
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
# Training
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
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


def train(model, train_loader, val_loader, args, device):
    """Full training loop with early stopping."""
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use different LR for pretrained vs scratch
    lr = args.lr_pretrained if args.pretrained else args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:  # plateau
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate train-val gap
        train_val_gap = train_acc - val_acc
        
        print(f'Epoch {epoch:3d}/{args.num_epochs} | '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} (gap={train_val_gap:+.4f}) | '
              f'LR: {current_lr:.2e}')
        
        # Log to WandB with grouped metrics
        wandb.log({
            'epoch': epoch,
            # Grouped for combined charts
            'loss/train': train_loss,
            'loss/val': val_loss,
            'accuracy/train': train_acc,
            'accuracy/val': val_acc,
            # Individual metrics  
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_val_gap': train_val_gap,
            'lr': current_lr
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            pretrained_str = 'pretrained' if args.pretrained else 'scratch'
            samples_str = f'_{args.training_samples}samples' if args.training_samples else ''
            save_path = os.path.join(args.path_weights, 
                                     f'{args.model}_{pretrained_str}{samples_str}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, save_path)
            print(f'  → Best model saved ({val_acc:.4f})')
            wandb.run.summary['best_val_acc'] = best_val_acc
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch} (patience={args.patience})')
            break
    
    return best_val_acc


# =============================================================================
# Testing
# =============================================================================

def calculate_p_value(acc, baseline, n):
    """Z-test for proportion significance."""
    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def test(model, test_loader, args, device):
    """Final evaluation on test set with visualizations."""
    criterion = nn.CrossEntropyLoss()
    
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
    
    pretrained_str = 'pretrained' if args.pretrained else 'scratch'
    samples_str = f'_{args.training_samples}samples' if args.training_samples else '_allsamples'
    
    print(f'\n{"="*50}')
    print(f'Test Results: {args.model} | {pretrained_str}{samples_str}')
    print(f'{"="*50}')
    print(f'Accuracy : {test_acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1-Score : {f1:.4f}')
    print(f'p-value  : {p_val:.2e}')
    
    # Log to WandB
    wandb.log({
        'test_accuracy': test_acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'test_p_value': p_val,
        'test_loss': test_loss
    })
    
    # Confusion Matrix
    cm_path = os.path.join(args.path_results, 
                           f'confusion_matrix_{args.model}_{pretrained_str}{samples_str}.png')
    plot_confusion_matrix(all_labels, all_preds, args.num_classes, cm_path)
    wandb.log({'confusion_matrix': wandb.Image(cm_path)})
    
    # t-SNE
    if all_features is not None and len(all_features) > 0:
        tsne_path = os.path.join(args.path_results,
                                 f'tsne_{args.model}_{pretrained_str}{samples_str}.png')
        plot_tsne(all_features, all_labels, args.num_classes, tsne_path)
        wandb.log({'tsne_plot': wandb.Image(tsne_path)})
    
    # Save results to text file
    txt_path = os.path.join(args.path_results,
                            f'results_{args.model}_{pretrained_str}{samples_str}.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Pretrained: {args.pretrained}\n')
        f.write(f'Training Samples: {args.training_samples if args.training_samples else "All"}\n')
        f.write('-' * 30 + '\n')
        f.write(f'Accuracy : {test_acc:.4f}\n')
        f.write(f'Precision: {prec:.4f}\n')
        f.write(f'Recall   : {rec:.4f}\n')
        f.write(f'F1-Score : {f1:.4f}\n')
        f.write(f'p-value  : {p_val:.2e}\n')
    
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
    dataset = load_dataset(args.dataset_path, image_size=args.image_size)
    
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
    
    # Log model parameters
    wandb.log({
        'model/total_parameters': total_params,
        'model/trainable_parameters': trainable_params,
    })
    
    # Log parameters per layer
    layer_params = {}
    for name, param in model.named_parameters():
        layer_params[f'model/layer_params/{name}'] = param.numel()
    wandb.config.update({'layer_parameters': layer_params})
    
    print(f'\n{"="*50}')
    print(f'Model: {args.model}')
    print(f'{"="*50}')
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    print(f'{"="*50}\n')
    
    if args.mode == 'train':
        print(f'Training: {args.model} | {"Pretrained" if args.pretrained else "Scratch"}')
        print(f'Epochs: {args.num_epochs} | Batch: {args.batch_size} | LR: {args.lr_pretrained if args.pretrained else args.lr}')
        print(f'{"="*50}\n')
        
        best_acc = train(model, train_loader, val_loader, args, device)
        
        # Load best model and test
        pretrained_str = 'pretrained' if args.pretrained else 'scratch'
        samples_str = f'_{args.training_samples}samples' if args.training_samples else ''
        best_path = os.path.join(args.path_weights, f'{args.model}_{pretrained_str}{samples_str}_best.pth')
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test(model, test_loader, args, device)
        
    else:  # Test only
        if args.weights:
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model_state_dict'])
            test(model, test_loader, args, device)
        else:
            print('Error: Please specify --weights for test mode')
    
    wandb.finish()


if __name__ == '__main__':
    main()

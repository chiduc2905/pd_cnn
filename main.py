"""PD Scalogram Classification - Standard CNN Training.

Training pipeline for CNN-based PD classification using pretrained or scratch models.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import wandb

from dataset import load_dataset
from net.models import get_model, get_available_models, count_parameters


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
    
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


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
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch:3d}/{args.num_epochs} | '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | '
              f'LR: {current_lr:.2e}')
        
        # Log to WandB
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'lr': current_lr
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            pretrained_str = 'pretrained' if args.pretrained else 'scratch'
            save_path = os.path.join(args.path_weights, 
                                     f'{args.model}_{pretrained_str}_best.pth')
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

def test(model, test_loader, args, device):
    """Final evaluation on test set."""
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    
    # Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print(f'\n{"="*50}')
    print(f'Test Results: {args.model} ({"pretrained" if args.pretrained else "scratch"})')
    print(f'{"="*50}')
    print(f'Accuracy : {test_acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1-Score : {f1:.4f}')
    print(f'Loss     : {test_loss:.4f}')
    
    # Log to WandB
    wandb.log({
        'test/accuracy': test_acc,
        'test/precision': prec,
        'test/recall': rec,
        'test/f1': f1,
        'test/loss': test_loss
    })
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f'\nConfusion Matrix:\n{cm}')
    
    # Save results
    pretrained_str = 'pretrained' if args.pretrained else 'scratch'
    samples_str = f'_{args.training_samples}samples' if args.training_samples else ''
    
    result_path = os.path.join(args.path_results, 
                               f'results_{args.model}_{pretrained_str}{samples_str}.txt')
    with open(result_path, 'w') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Pretrained: {args.pretrained}\n')
        f.write(f'Training Samples: {args.training_samples or "All"}\n')
        f.write('-' * 30 + '\n')
        f.write(f'Accuracy : {test_acc:.4f}\n')
        f.write(f'Precision: {prec:.4f}\n')
        f.write(f'Recall   : {rec:.4f}\n')
        f.write(f'F1-Score : {f1:.4f}\n')
        f.write(f'Loss     : {test_loss:.4f}\n')
        f.write('-' * 30 + '\n')
        f.write(f'Confusion Matrix:\n{cm}\n')
    
    print(f'\nResults saved to {result_path}')
    
    return test_acc


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        group=args.model
    )
    wandb.log({
        'model/total_params': total_params,
        'model/trainable_params': trainable_params
    })
    
    if args.mode == 'train':
        print(f'\n{"="*50}')
        print(f'Training: {args.model} | {"Pretrained" if args.pretrained else "Scratch"}')
        print(f'Epochs: {args.num_epochs} | Batch: {args.batch_size} | LR: {args.lr_pretrained if args.pretrained else args.lr}')
        print(f'{"="*50}\n')
        
        best_acc = train(model, train_loader, val_loader, args, device)
        
        # Load best model and test
        pretrained_str = 'pretrained' if args.pretrained else 'scratch'
        best_path = os.path.join(args.path_weights, f'{args.model}_{pretrained_str}_best.pth')
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

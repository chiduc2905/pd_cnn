"""Test one or multiple checkpoints.

Usage:
    # Test a single checkpoint
    python test_checkpoint.py --checkpoint checkpoints/exp001_resnet18_pretrained_18samples_best.pth

    # Test all checkpoints in a directory
    python test_checkpoint.py --checkpoint_dir checkpoints/

    # Test specific experiment IDs
    python test_checkpoint.py --checkpoint_dir checkpoints/ --exp_ids 1 2 3

    # Test with specific shot settings
    python test_checkpoint.py --checkpoint checkpoints/model.pth --shot_list 1,5
"""
import argparse
import os
import sys
import glob
import re
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from sklearn.metrics import precision_recall_fscore_support

from net.models import get_model, get_available_models
from function.function import plot_confusion_matrix, plot_tsne
from data.load_data import load_dataset


def get_args():
    parser = argparse.ArgumentParser(description='Test checkpoints for PD Classification')
    
    # Checkpoint selection (one of these is required)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory containing checkpoints')
    parser.add_argument('--exp_ids', nargs='+', type=int, default=None,
                        help='Specific experiment IDs to test (e.g., 1 2 3)')
    
    # Data
    parser.add_argument('--dataset_path', type=str, default='/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_cnn/scalogram_v2_split')
    parser.add_argument('--path_results', type=str, default='results/')
    parser.add_argument('--val_per_class', type=int, default=60)
    parser.add_argument('--test_per_class', type=int, default=60)
    
    # Evaluation settings
    parser.add_argument('--eval_mode', type=str, default='episode',
                        choices=['standard', 'episode'],
                        help='Evaluation mode (default: episode)')
    parser.add_argument('--episode_num_test', type=int, default=300,
                        help='Number of test episodes (default: 300)')
    parser.add_argument('--query_per_class', type=int, default=1,
                        help='Query samples per class per episode (default: 1)')
    parser.add_argument('--shot_list', type=str, default='1,5',
                        help='Comma-separated shot settings (default: 1,5)')
    
    # Other
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--project', type=str, default='pd_cnn_test',
                        help='WandB project name for test results')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    
    return parser.parse_args()


def parse_checkpoint_info(checkpoint_path):
    """Extract model info from checkpoint filename."""
    filename = os.path.basename(checkpoint_path)
    
    # Pattern: exp{ID}_{model}_{pretrained/scratch}_{samples}_best.pth
    # or: {model}_{pretrained/scratch}_{samples}_best.pth
    
    exp_match = re.match(r'exp(\d+)_(.+)_(pretrained|scratch)_(\w+)_best\.pth', filename)
    if exp_match:
        exp_id = int(exp_match.group(1))
        model_name = exp_match.group(2)
        pretrained = exp_match.group(3) == 'pretrained'
        samples_str = exp_match.group(4)
    else:
        # Fallback for old naming convention
        old_match = re.match(r'(.+)_(pretrained|scratch)_?(\w*)_?best\.pth', filename)
        if old_match:
            exp_id = None
            model_name = old_match.group(1)
            pretrained = old_match.group(2) == 'pretrained'
            samples_str = old_match.group(3) if old_match.group(3) else 'allsamples'
        else:
            return None
    
    # Parse training samples
    if 'allsamples' in samples_str:
        training_samples = None
    else:
        samples_match = re.search(r'(\d+)samples', samples_str)
        training_samples = int(samples_match.group(1)) if samples_match else None
    
    return {
        'exp_id': exp_id,
        'model_name': model_name,
        'pretrained': pretrained,
        'training_samples': training_samples,
        'samples_str': samples_str
    }


def find_checkpoints(args):
    """Find all checkpoints to test."""
    checkpoints = []
    
    if args.checkpoint:
        # Single checkpoint
        if os.path.exists(args.checkpoint):
            checkpoints.append(args.checkpoint)
        else:
            print(f'Error: Checkpoint not found: {args.checkpoint}')
            sys.exit(1)
    
    elif args.checkpoint_dir:
        # All checkpoints in directory
        pattern = os.path.join(args.checkpoint_dir, '*.pth')
        all_checkpoints = glob.glob(pattern)
        
        if args.exp_ids:
            # Filter by experiment IDs
            for ckpt in all_checkpoints:
                info = parse_checkpoint_info(ckpt)
                if info and info['exp_id'] in args.exp_ids:
                    checkpoints.append(ckpt)
        else:
            checkpoints = all_checkpoints
    
    # Sort by experiment ID if available
    def sort_key(ckpt):
        info = parse_checkpoint_info(ckpt)
        return info['exp_id'] if info and info['exp_id'] else 999
    
    checkpoints.sort(key=sort_key)
    return checkpoints


def calculate_p_value(acc, baseline, n):
    """Z-test for proportion significance."""
    if n <= 0:
        return 1.0
    z = (acc - baseline) / np.sqrt(baseline * (1 - baseline) / n)
    return 2 * norm.sf(abs(z))


def test_checkpoint(checkpoint_path, dataset, args, device):
    """Test a single checkpoint."""
    info = parse_checkpoint_info(checkpoint_path)
    if not info:
        print(f'Warning: Could not parse checkpoint info from {checkpoint_path}')
        return None
    
    print(f'\n{"="*70}')
    if info['exp_id']:
        print(f'Testing Experiment #{info["exp_id"]:03d}')
    print(f'Model: {info["model_name"]} | Pretrained: {info["pretrained"]} | Samples: {info["samples_str"]}')
    print(f'Checkpoint: {os.path.basename(checkpoint_path)}')
    print(f'{"="*70}')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get image size from checkpoint args if available
    ckpt_args = checkpoint.get('args', {})
    image_size = ckpt_args.get('image_size', 224)
    num_classes = ckpt_args.get('num_classes', 3)
    
    # Reload dataset if image size differs
    if image_size != getattr(dataset, '_image_size', 224):
        print(f'Reloading dataset with image_size={image_size}')
        dataset = load_dataset(
            args.dataset_path,
            image_size=image_size,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class
        )
        dataset._image_size = image_size
    
    # Create model
    model = get_model(info['model_name'], num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare test data
    test_X = torch.from_numpy(dataset.X_test.astype(np.float32))
    test_y = torch.from_numpy(dataset.y_test).long()
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    results = {}
    
    if args.eval_mode == 'episode':
        # Episode-based evaluation
        from main import evaluate_multishot
        
        shot_list = [int(s) for s in args.shot_list.split(',')]
        
        # Create args-like object for evaluate_multishot
        class EvalArgs:
            pass
        eval_args = EvalArgs()
        eval_args.num_classes = num_classes
        eval_args.query_per_class = args.query_per_class
        eval_args.shot_list = args.shot_list
        
        multishot_results = evaluate_multishot(
            model, dataset.X_test, dataset.y_test, eval_args, device,
            num_episodes=args.episode_num_test, phase='test'
        )
        
        for shot_num in shot_list:
            mean_acc, std_acc, ci95, all_preds, all_labels = multishot_results[shot_num]
            
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds,
                labels=list(range(num_classes)),
                average='macro',
                zero_division=0
            )
            
            results[f'{shot_num}shot'] = {
                'accuracy': mean_acc,
                'accuracy_std': std_acc,
                'accuracy_ci95': ci95,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
            print(f'{shot_num}-shot: {mean_acc*100:.2f} ± {std_acc*100:.2f}% (F1: {f1:.4f})')
            
            # Save confusion matrix
            cm_path = os.path.join(
                args.path_results,
                f'cm_{info["model_name"]}_{info["samples_str"]}_{shot_num}shot.png'
            )
            plot_confusion_matrix(all_labels, all_preds, num_classes, cm_path)
    
    else:
        # Standard batch evaluation
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = correct / total
        test_loss = total_loss / total
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds,
            labels=list(range(num_classes)),
            average='macro',
            zero_division=0
        )
        
        results['standard'] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        
        print(f'Accuracy: {test_acc*100:.2f}% | F1: {f1:.4f} | Loss: {test_loss:.4f}')
        
        # Save confusion matrix
        cm_path = os.path.join(
            args.path_results,
            f'cm_{info["model_name"]}_{info["samples_str"]}_standard.png'
        )
        plot_confusion_matrix(all_labels, all_preds, num_classes, cm_path)
    
    # Add checkpoint info to results
    results['info'] = info
    results['checkpoint_path'] = checkpoint_path
    
    return results


def main():
    args = get_args()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')
    
    # Create results directory
    os.makedirs(args.path_results, exist_ok=True)
    
    # Find checkpoints
    checkpoints = find_checkpoints(args)
    if not checkpoints:
        print('Error: No checkpoints found!')
        sys.exit(1)
    
    print(f'\nFound {len(checkpoints)} checkpoint(s) to test')
    
    # Load dataset once
    print(f'\nLoading dataset from {args.dataset_path}...')
    dataset = load_dataset(
        args.dataset_path,
        image_size=224,  # Will be reloaded if needed
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class
    )
    dataset._image_size = 224
    print(f'Test set: {len(dataset.X_test)} samples')
    
    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            config=vars(args),
            name=f'test_{len(checkpoints)}_checkpoints'
        )
    
    # Test all checkpoints
    all_results = []
    for i, ckpt in enumerate(checkpoints, 1):
        print(f'\n[{i}/{len(checkpoints)}] ', end='')
        result = test_checkpoint(ckpt, dataset, args, device)
        if result:
            all_results.append(result)
            
            # Log to WandB
            if not args.no_wandb:
                info = result['info']
                log_dict = {
                    'model': info['model_name'],
                    'pretrained': info['pretrained'],
                    'training_samples': info['training_samples'] or 'all'
                }
                
                if args.eval_mode == 'episode':
                    for shot in args.shot_list.split(','):
                        shot_key = f'{shot}shot'
                        if shot_key in result:
                            log_dict[f'{shot_key}_accuracy'] = result[shot_key]['accuracy']
                            log_dict[f'{shot_key}_f1'] = result[shot_key]['f1']
                else:
                    log_dict['accuracy'] = result['standard']['accuracy']
                    log_dict['f1'] = result['standard']['f1']
                
                wandb.log(log_dict)
    
    # Summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    print(f'Tested {len(all_results)}/{len(checkpoints)} checkpoints successfully')
    
    if all_results:
        # Sort by accuracy
        if args.eval_mode == 'episode':
            shot = args.shot_list.split(',')[0]
            all_results.sort(key=lambda x: x.get(f'{shot}shot', {}).get('accuracy', 0), reverse=True)
            
            print(f'\nTop 5 models ({shot}-shot accuracy):')
            for i, r in enumerate(all_results[:5], 1):
                info = r['info']
                acc = r.get(f'{shot}shot', {}).get('accuracy', 0)
                print(f'  {i}. {info["model_name"]} ({info["samples_str"]}): {acc*100:.2f}%')
        else:
            all_results.sort(key=lambda x: x.get('standard', {}).get('accuracy', 0), reverse=True)
            
            print('\nTop 5 models (standard accuracy):')
            for i, r in enumerate(all_results[:5], 1):
                info = r['info']
                acc = r.get('standard', {}).get('accuracy', 0)
                print(f'  {i}. {info["model_name"]} ({info["samples_str"]}): {acc*100:.2f}%')
    
    if not args.no_wandb:
        wandb.finish()
    
    print('\nDone!')


if __name__ == '__main__':
    main()

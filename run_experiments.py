"""Run all CNN experiments for PD classification.

Trains all models with pretrained weights using proper transfer learning:
  - Phase 1: Freeze backbone, train classifier (20 epochs)
  - Phase 2: Fine-tune entire network with discriminative LR (80 epochs)
  
Datasets:
  - Original (scalogram_original): Training samples 18, 30
  - Augmented (scalogram_augmented): Training samples 18, 60, all

Evaluation modes:
  - Standard: Traditional batch evaluation on test set
  - Episode: Episode-based evaluation (for fair comparison with few-shot)
  - Episodic Fine-tuning: Partial fine-tuning baseline (freeze early backbone, 
                          fine-tune last block + classifier per episode)
"""
import subprocess
import sys
from itertools import product

# All available models (sorted by size)
MODELS = [
    # Small models (< 5M params)
    'squeezenet1_1',      # 1.2M
    'shufflenetv2_x0_5',  # 1.4M
    'shufflenetv2_x1_0',  # 2.3M
    'mobilenetv3_small',  # 2.5M
    
    # Medium models (5M - 15M params)
    'efficientnet_b0',    # 5.3M
    'mobilenetv3_large',  # 5.5M
    'efficientnet_b1',    # 7.8M
    'densenet121',        # 8M
    'efficientnet_b2',    # 9.2M
    'resnet18',           # 11.7M
    'efficientnet_b3',    # 12M
    'densenet169',        # 14M
    
    # Large models (15M - 50M params)
    'densenet201',        # 20M
    'resnet34',           # 21.8M
    'resnet50',           # 25.6M
    'inception_v3',       # 27.2M (benchmark)
    'resnet101',          # 44.5M
    
    # Very large models (> 100M params) - Classic benchmarks
    'vgg16_bn',           # 138M
    'vgg19_bn',           # 144M
]

# Dataset configurations with different training sample settings
# Format: {dataset_name: {'path': path, 'samples': [sample_sizes]}}
DATASET_CONFIGS = {
    'original': {
        'path': '/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_original',
        'samples': [18, 30],  # Original dataset: 18 and 30 training samples only
    },
    'augmented': {
        'path': '/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_augmented',
        'samples': [18, 60, None],  # Augmented dataset: 18, 60, and all samples
    },
}

# Default training sample configurations (None = all samples)
# These define how many samples are used for TRANSFER LEARNING training
TRAINING_SAMPLES = [18, 60, None]

# Transfer learning settings (unified for fair comparison)
DEFAULT_FREEZE_EPOCHS = 20
DEFAULT_NUM_EPOCHS = 100

# Per-model input sizes (must match ImageNet pretrained expectations)
# InceptionV3 requires 299x299 minimum, others use 224 or higher
MODEL_INPUT_SIZE = {
    'squeezenet1_1': 224,
    'shufflenetv2_x0_5': 224,
    'shufflenetv2_x1_0': 224,
    'mobilenetv3_small': 224,
    'efficientnet_b0': 224,
    'mobilenetv3_large': 224,
    'efficientnet_b1': 240,
    'densenet121': 224,
    'efficientnet_b2': 260,
    'resnet18': 224,
    'efficientnet_b3': 300,
    'densenet169': 224,
    'densenet201': 224,
    'resnet34': 224,
    'resnet50': 224,
    'inception_v3': 299,  # CRITICAL: InceptionV3 requires 299x299 minimum
    'resnet101': 224,
    'vgg16_bn': 224,
    'vgg19_bn': 224,
}


def run_experiment(model: str, training_samples: int = None, gpu: int = 0, 
                   freeze_epochs: int = DEFAULT_FREEZE_EPOCHS, experiment_id: int = 1, **kwargs):
    """Run a single experiment with pretrained weights and transfer learning.
    
    Args:
        model: Model architecture name
        training_samples: Number of training samples (18, 60, or None for all)
        gpu: GPU id
        freeze_epochs: Epochs with frozen backbone
        experiment_id: Unique experiment number for checkpoint naming
        **kwargs: Additional arguments passed to main.py
    """
    # Get model-specific input size (critical for InceptionV3)
    image_size = MODEL_INPUT_SIZE.get(model, 224)
    
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--mode', 'train',
        '--gpu', str(gpu),
        '--pretrained',  # Always use pretrained
        '--freeze_epochs', str(freeze_epochs),
        '--image_size', str(image_size),  # Use model-specific input size
        '--experiment_id', str(experiment_id),  # Pass experiment ID for checkpoint naming
    ]
    
    if training_samples is not None:
        cmd.extend(['--training_samples', str(training_samples)])
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    samples_str = f'{training_samples}samples' if training_samples else 'all'
    
    print(f'\n{"="*60}')
    print(f'Experiment #{experiment_id}: {model} | pretrained | {samples_str}')
    print(f'Transfer Learning: {freeze_epochs} epochs freeze + {kwargs.get("num_epochs", DEFAULT_NUM_EPOCHS) - freeze_epochs} epochs fine-tune')
    if kwargs.get('episodic_finetune', False):
        print(f'Episodic Fine-tuning: {kwargs.get("shot_num", 5)}-shot, {kwargs.get("finetune_steps", 10)} steps')
    print(f'{"="*60}\n')
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f'WARNING: Experiment #{experiment_id} {model} ({samples_str}) failed with code {result.returncode}')
        return False
    return True


def run_all(models=None, training_samples_list=None, freeze_epochs=DEFAULT_FREEZE_EPOCHS, 
            dataset_path=None, **kwargs):
    """Run all experiments with pretrained weights.
    
    Note: training_samples_list controls how many samples are used for 
    transfer learning training. This is separate from evaluation settings.
    """
    models = models or MODELS
    training_samples_list = training_samples_list or TRAINING_SAMPLES
    
    results = []
    total_experiments = len(models) * len(training_samples_list)
    experiment_id = 0  # Experiment counter starting from 1
    
    for model, samples in product(models, training_samples_list):
        experiment_id += 1
        print(f'\n[{experiment_id}/{total_experiments}] Starting experiment #{experiment_id}...')
        success = run_experiment(model, samples, freeze_epochs=freeze_epochs, 
                                 experiment_id=experiment_id, dataset_path=dataset_path, **kwargs)
        results.append((experiment_id, model, samples, success))
    
    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    if dataset_path:
        print(f'Dataset: {dataset_path}')
    print(f'Transfer Learning: {freeze_epochs} epochs freeze, then fine-tune')
    if kwargs.get('episodic_finetune', False):
        print(f'Episodic Fine-tuning Eval: {kwargs.get("shot_num", 5)}-shot, {kwargs.get("finetune_steps", 10)} steps')
    print(f'{"-"*60}')
    
    for exp_id, model, samples, success in results:
        samples_str = f'{samples}' if samples else 'all'
        status = '✓' if success else '✗'
        print(f'{status} Exp#{exp_id:03d} {model:20s} | pretrained | {samples_str:>4s} samples')
    
    total = len(results)
    passed = sum(1 for _, _, _, s in results if s)
    print(f'\nTotal: {passed}/{total} passed')
    return results


def run_all_datasets(models=None, datasets=None, freeze_epochs=DEFAULT_FREEZE_EPOCHS, **kwargs):
    """Run experiments on multiple datasets with their respective training sample configurations.
    
    Args:
        models: List of model names to run (default: all MODELS)
        datasets: List of dataset names from DATASET_CONFIGS (default: ['original', 'augmented'])
        freeze_epochs: Number of epochs to freeze backbone
        **kwargs: Additional arguments passed to run_experiment
    """
    models = models or MODELS
    datasets = datasets or list(DATASET_CONFIGS.keys())
    
    all_results = {}
    global_experiment_id = 0
    
    # Calculate total experiments across all datasets
    total_experiments = sum(
        len(models) * len(DATASET_CONFIGS[ds]['samples']) 
        for ds in datasets if ds in DATASET_CONFIGS
    )
    
    print(f'\n{"="*80}')
    print(f'RUNNING EXPERIMENTS ON {len(datasets)} DATASET(S)')
    print(f'Total experiments: {total_experiments}')
    print(f'{"="*80}')
    
    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f'WARNING: Unknown dataset "{dataset_name}", skipping...')
            continue
            
        config = DATASET_CONFIGS[dataset_name]
        dataset_path = config['path']
        training_samples = config['samples']
        
        print(f'\n{"#"*80}')
        print(f'DATASET: {dataset_name.upper()}')
        print(f'Path: {dataset_path}')
        print(f'Training samples: {[s if s else "all" for s in training_samples]}')
        print(f'{"#"*80}')
        
        results = []
        num_experiments = len(models) * len(training_samples)
        
        for model, samples in product(models, training_samples):
            global_experiment_id += 1
            print(f'\n[GLOBAL {global_experiment_id}/{total_experiments}] Starting experiment...')
            success = run_experiment(
                model, samples, 
                freeze_epochs=freeze_epochs, 
                experiment_id=global_experiment_id,
                dataset_path=dataset_path,
                **kwargs
            )
            results.append((global_experiment_id, model, samples, success))
        
        # Dataset summary
        print(f'\n{"="*60}')
        print(f'SUMMARY - {dataset_name.upper()}')
        print(f'{"="*60}')
        print(f'Dataset: {dataset_path}')
        print(f'Training samples: {[s if s else "all" for s in training_samples]}')
        print(f'{"-"*60}')
        
        for exp_id, model, samples, success in results:
            samples_str = f'{samples}' if samples else 'all'
            status = '✓' if success else '✗'
            print(f'{status} Exp#{exp_id:03d} {model:20s} | {samples_str:>4s} samples')
        
        passed = sum(1 for _, _, _, s in results if s)
        print(f'\nDataset {dataset_name}: {passed}/{len(results)} passed')
        
        all_results[dataset_name] = results
    
    # Final global summary
    print(f'\n{"="*80}')
    print('FINAL SUMMARY - ALL DATASETS')
    print(f'{"="*80}')
    
    total_passed = 0
    total_run = 0
    for dataset_name, results in all_results.items():
        passed = sum(1 for _, _, _, s in results if s)
        total_passed += passed
        total_run += len(results)
        print(f'{dataset_name:15s}: {passed}/{len(results)} passed')
    
    print(f'{"-"*40}')
    print(f'{"TOTAL":15s}: {total_passed}/{total_run} passed')
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all CNN experiments (pretrained with transfer learning)')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=None,
                        help='Specific models to run (default: all)')
    parser.add_argument('--samples', nargs='+', type=int, default=None,
                        help='Training sample sizes for transfer learning (default: 18, 60, all). Use 0 for all samples.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (default: 0)')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS,
                        help=f'Total epochs (default: {DEFAULT_NUM_EPOCHS})')
    parser.add_argument('--freeze_epochs', type=int, default=DEFAULT_FREEZE_EPOCHS,
                        help=f'Epochs with frozen backbone (default: {DEFAULT_FREEZE_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=16)
    
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Custom dataset path (overrides --datasets if specified)')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIGS.keys()),
                        default=None,
                        help=f'Datasets to run: {list(DATASET_CONFIGS.keys())} (default: both)')
    parser.add_argument('--run_all_datasets', action='store_true',
                        help='Run experiments on all configured datasets (original + augmented)')
    
    # Evaluation modes
    parser.add_argument('--eval_mode', type=str, default='episode',
                        choices=['standard', 'episode'],
                        help='Primary evaluation mode (default: episode for fair comparison with few-shot)')
    parser.add_argument('--episode_num_val', type=int, default=200,
                        help='Number of validation episodes (default: 200)')
    parser.add_argument('--episode_num_test', type=int, default=300,
                        help='Number of test episodes (default: 300)')
    parser.add_argument('--query_per_class', type=int, default=1,
                        help='Query samples per class per episode (default: 1)')
    parser.add_argument('--shot_list', type=str, default='1,5',
                        help='Shot settings to evaluate (default: 1,5)')
    
    # WandB
    parser.add_argument('--project', type=str, default='pd_cnn',
                        help='WandB project name (default: pd_cnn)')
    
    args = parser.parse_args()
    
    # Common kwargs for experiments
    common_kwargs = dict(
        gpu=args.gpu,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_mode=args.eval_mode,
        episode_num_val=args.episode_num_val,
        episode_num_test=args.episode_num_test,
        query_per_class=args.query_per_class,
        shot_list=args.shot_list,
        project=args.project
    )
    
    # Determine which mode to use
    if args.run_all_datasets or args.datasets:
        # Use the new run_all_datasets function
        datasets_to_run = args.datasets if args.datasets else list(DATASET_CONFIGS.keys())
        
        print(f'\n*** MULTI-DATASET MODE ***')
        print(f'Datasets: {datasets_to_run}')
        for ds in datasets_to_run:
            if ds in DATASET_CONFIGS:
                cfg = DATASET_CONFIGS[ds]
                print(f'  - {ds}: {cfg["path"]}')
                print(f'    Training samples: {[s if s else "all" for s in cfg["samples"]]}')
        
        run_all_datasets(
            models=args.models,
            datasets=datasets_to_run,
            freeze_epochs=args.freeze_epochs,
            **common_kwargs
        )
    else:
        # Legacy single-dataset mode
        # Process samples argument (0 means None/all)
        training_samples = None
        if args.samples:
            training_samples = [s if s != 0 else None for s in args.samples]
        
        dataset_path = args.dataset_path if args.dataset_path else './scalogram/'
        
        print(f'\n*** SINGLE-DATASET MODE ***')
        print(f'Dataset path: {dataset_path}')
        print(f'Training samples: {training_samples if training_samples else TRAINING_SAMPLES}')
        
        run_all(
            models=args.models,
            training_samples_list=training_samples,
            freeze_epochs=args.freeze_epochs,
            dataset_path=dataset_path,
            **common_kwargs
        )

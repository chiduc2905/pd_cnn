"""Run all CNN experiments for PD classification.

Trains all 6 models with pretrained weights.
Training samples: 18, 60, and all.
"""
import subprocess
import sys
from itertools import product

# All available models (sorted by size)
MODELS = [
    'squeezenet1_1',      # 1.2M
    'shufflenetv2_x0_5',  # 1.4M
    'mobilenetv3_small',  # 2.5M
    'efficientnet_b0',    # 5.3M
    'densenet121',        # 8M
    'resnet18',           # 11.7M
]

# Training sample configurations (None = all samples)
TRAINING_SAMPLES = [18, 60, None]


def run_experiment(model: str, training_samples: int = None, gpu: int = 0, **kwargs):
    """Run a single experiment with pretrained weights."""
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--mode', 'train',
        '--gpu', str(gpu),
        '--pretrained',  # Always use pretrained
    ]
    
    if training_samples is not None:
        cmd.extend(['--training_samples', str(training_samples)])
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])
    
    samples_str = f'{training_samples}samples' if training_samples else 'all'
    
    print(f'\n{"="*60}')
    print(f'Running: {model} | pretrained | {samples_str}')
    print(f'{"="*60}\n')
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f'WARNING: {model} ({samples_str}) failed with code {result.returncode}')
        return False
    return True


def run_all(models=None, training_samples_list=None, **kwargs):
    """Run all experiments with pretrained weights."""
    models = models or MODELS
    training_samples_list = training_samples_list or TRAINING_SAMPLES
    
    results = []
    total_experiments = len(models) * len(training_samples_list)
    current = 0
    
    for model, samples in product(models, training_samples_list):
        current += 1
        print(f'\n[{current}/{total_experiments}] Starting experiment...')
        success = run_experiment(model, samples, **kwargs)
        results.append((model, samples, success))
    
    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    
    for model, samples, success in results:
        samples_str = f'{samples}' if samples else 'all'
        status = '✓' if success else '✗'
        print(f'{status} {model:20s} | pretrained | {samples_str:>4s} samples')
    
    total = len(results)
    passed = sum(1 for _, _, s in results if s)
    print(f'\nTotal: {passed}/{total} passed')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all CNN experiments (pretrained)')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=None,
                        help='Specific models to run (default: all)')
    parser.add_argument('--samples', nargs='+', type=int, default=None,
                        help='Training sample sizes (default: 18, 60, all). Use 0 for all samples.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (default: 0)')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset_path', type=str, default='./scalogram/')
    
    args = parser.parse_args()
    
    # Process samples argument (0 means None/all)
    training_samples = None
    if args.samples:
        training_samples = [s if s != 0 else None for s in args.samples]
    
    run_all(
        models=args.models,
        training_samples_list=training_samples,
        gpu=args.gpu,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path
    )

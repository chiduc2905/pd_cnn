"""Run all CNN experiments for PD classification.

Simplified version - runs on single dataset (scalogram_official) with multiple training samples.
  
Training modes:
  - Feature Extraction (no-finetune): Freeze backbone, train classifier only
  - Fine-tuning (finetune): Phase 1 freeze (20 epochs) + Phase 2 full fine-tune (80 epochs)
"""
import subprocess
import sys
from itertools import product

# All available models (sorted by size)
MODELS = [
    # Small models (<5M params)
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
    'inception_v3',       # 27.2M
    'resnet101',          # 44.5M
    
    # Very large models (>100M params)
    'vgg16_bn',           # 138M
    'vgg19_bn',           # 144M
]

# Dataset path (pre-split with train/val/test folders)
DATASET_PATH = '/mnt/disk2/nhatnc/res/scalogram_fewshot/proposed_model/smnet/scalogram_official'


# Training sample configurations
TRAINING_SAMPLES = [30, 60, 150, None]  # None = use all training data

# Training modes: True = full fine-tune, False = feature extraction only
FINETUNE_MODES = [True, False]

# Transfer learning settings
DEFAULT_FREEZE_EPOCHS = 20
DEFAULT_NUM_EPOCHS = 100

# Per-model input sizes
MODEL_INPUT_SIZE = {
    'squeezenet1_1': 224, 'shufflenetv2_x0_5': 224, 'shufflenetv2_x1_0': 224,
    'mobilenetv3_small': 224, 'efficientnet_b0': 224, 'mobilenetv3_large': 224,
    'efficientnet_b1': 240, 'densenet121': 224, 'efficientnet_b2': 260,
    'resnet18': 224, 'efficientnet_b3': 300, 'densenet169': 224,
    'densenet201': 224, 'resnet34': 224, 'resnet50': 224,
    'inception_v3': 299,  # CRITICAL: InceptionV3 requires 299x299 minimum
    'resnet101': 224, 'vgg16_bn': 224, 'vgg19_bn': 224,
}


def run_experiment(model, training_samples=None, gpu=0, freeze_epochs=DEFAULT_FREEZE_EPOCHS,
                   experiment_id=1, no_finetune=False, **kwargs):
    """Run a single experiment."""
    image_size = MODEL_INPUT_SIZE.get(model, 224)
    
    cmd = [
        sys.executable, 'main.py',
        '--model', model,
        '--mode', 'train',
        '--gpu', str(gpu),
        '--pretrained',
        '--freeze_epochs', str(freeze_epochs),
        '--image_size', str(image_size),
        '--experiment_id', str(experiment_id),
        '--dataset_path', DATASET_PATH,
        # Episodic evaluation settings (aligned with mamba_glscnet)
        '--episode_num_val', '300',
        '--episode_num_test', '300',
        '--query_per_class', '5',
        '--episodic_finetune',  # Enable true few-shot evaluation
    ]

    
    if no_finetune:
        cmd.append('--no_finetune')
    
    if training_samples is not None:
        cmd.extend(['--training_samples', str(training_samples)])
    
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    samples_str = f'{training_samples}samples' if training_samples else 'all'
    finetune_str = 'no-finetune' if no_finetune else 'finetune'
    
    print(f'\n{"="*60}')
    print(f'Experiment #{experiment_id}: {model} | {samples_str} | {finetune_str}')
    if no_finetune:
        print(f'Mode: Feature Extraction ({freeze_epochs} epochs, classifier only)')
    else:
        print(f'Mode: {freeze_epochs} epochs freeze + {kwargs.get("num_epochs", DEFAULT_NUM_EPOCHS) - freeze_epochs} epochs fine-tune')
    print(f'{"="*60}\n')

    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f'WARNING: Experiment #{experiment_id} failed with code {result.returncode}')
        return False
    return True


def run_all(models=None, training_samples=None, finetune_modes=None,
            freeze_epochs=DEFAULT_FREEZE_EPOCHS, **kwargs):
    """Run all experiments."""
    models = models or MODELS
    training_samples = training_samples or TRAINING_SAMPLES
    finetune_modes = finetune_modes if finetune_modes is not None else FINETUNE_MODES
    
    total_experiments = len(models) * len(training_samples) * len(finetune_modes)
    
    print(f'\n{"="*80}')
    print(f'PD CNN EXPERIMENTS')
    print(f'{"="*80}')
    print(f'Dataset: {DATASET_PATH}')
    print(f'Training samples: {[s if s else "all" for s in training_samples]}')
    print(f'Modes: {["finetune" if m else "no-finetune" for m in finetune_modes]}')
    print(f'Total experiments: {total_experiments}')
    print(f'{"="*80}')
    
    results = []
    experiment_id = 0
    
    for model, samples, do_finetune in product(models, training_samples, finetune_modes):
        experiment_id += 1
        success = run_experiment(
            model, samples,
            freeze_epochs=freeze_epochs,
            experiment_id=experiment_id,
            no_finetune=not do_finetune,
            **kwargs
        )
        results.append((experiment_id, model, samples, do_finetune, success))
    
    # Summary
    print(f'\n{"="*80}')
    print('SUMMARY')
    print(f'{"="*80}')
    for exp_id, model, samples, do_finetune, success in results:
        samples_str = f'{samples}' if samples else 'all'
        mode = 'FT' if do_finetune else 'FE'
        status = '✓' if success else '✗'
        print(f'{status} Exp#{exp_id:03d} {model:20s} | {samples_str:>4s} samples | {mode}')
    
    passed = sum(1 for *_, s in results if s)
    print(f'\nTotal: {passed}/{total_experiments} passed')
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CNN experiments')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=None)
    parser.add_argument('--training_samples', nargs='+', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--freeze_epochs', type=int, default=DEFAULT_FREEZE_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_mode', type=str, default='episode', choices=['standard', 'episode'])
    parser.add_argument('--episode_num_val', type=int, default=150)
    parser.add_argument('--episode_num_test', type=int, default=150)
    parser.add_argument('--query_per_class', type=int, default=5)
    parser.add_argument('--shot_list', type=str, default='1,5')
    parser.add_argument('--project', type=str, default='pd_cnn')
    
    args = parser.parse_args()
    
    run_all(
        models=args.models,
        training_samples=args.training_samples,
        freeze_epochs=args.freeze_epochs,
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

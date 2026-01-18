"""
Helper script to list available models for Transfer Learning fine-tuning.

This script shows all models available in net/models.py MODEL_INFO dict
that can be used with benchmark.py --tl_models argument.
"""
import sys
sys.path.insert(0, '.')

from net.models import MODEL_INFO

print("="*70)
print("AVAILABLE MODELS FOR TRANSFER LEARNING FINE-TUNING")
print("="*70)

# Separate paper models and additional models
paper_models = {k: v for k, v in MODEL_INFO.items() if v.get('paper', False)}
other_models = {k: v for k, v in MODEL_INFO.items() if not v.get('paper', False)}

print("\n📄 PAPER MODELS (default for benchmark):")
print("-" * 70)
print(f"{'Model':<25} {'Params':<12} {'Input':<10} {'Source'}")
print("-" * 70)
for name, info in paper_models.items():
    print(f"{name:<25} {info['params']:<12} {info['input_size']:<10} {info['source']}")

print(f"\nTotal: {len(paper_models)} paper models")

print("\n🔧 ADDITIONAL MODELS (available for custom benchmarks):")
print("-" * 70)
print(f"{'Model':<25} {'Params':<12} {'Input':<10} {'Source'}")
print("-" * 70)
for name, info in other_models.items():
    print(f"{name:<25} {info['params']:<12} {info['input_size']:<10} {info['source']}")

print(f"\nTotal: {len(other_models)} additional models")

print("\n" + "="*70)
print("USAGE EXAMPLES")
print("="*70)

print("\n1. Use default paper models (vgg19, resnet50, densenet201):")
print("   python benchmark.py --run_tl_finetune")

print("\n2. Use specific paper models:")
print("   python benchmark.py --run_tl_finetune --tl_models vgg19 xception densenet201")

print("\n3. Use lightweight models:")
print("   python benchmark.py --run_tl_finetune --tl_models mobilenetv3_small efficientnet_b0")

print("\n4. Use single model:")
print("   python benchmark.py --run_tl_finetune --tl_models resnet50")

print("\n5. Use all paper models:")
paper_model_names = ' '.join(paper_models.keys())
print(f"   python benchmark.py --run_tl_finetune --tl_models {paper_model_names}")

print("\n" + "="*70)

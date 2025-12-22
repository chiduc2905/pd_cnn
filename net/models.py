"""CNN Models for PD Scalogram Classification.

Provides smallest versions of popular architectures with pretrained/scratch support.
All models output logits for num_classes.

Transfer Learning Support:
- freeze_backbone(): Freeze all backbone layers, only train classifier
- unfreeze_backbone(): Unfreeze all layers for fine-tuning
- get_classifier_params(): Get only classifier parameters
- get_backbone_params(): Get only backbone parameters
"""
import torch
import torch.nn as nn
import torchvision.models as models


# Model registry with parameter counts (approximate)
# Models sorted by parameter count (small to large)
MODEL_INFO = {
    # Small models (< 5M params)
    'squeezenet1_1': {'params': '1.2M', 'input_size': 224},
    'shufflenetv2_x0_5': {'params': '1.4M', 'input_size': 224},
    'shufflenetv2_x1_0': {'params': '2.3M', 'input_size': 224},
    'mobilenetv3_small': {'params': '2.5M', 'input_size': 224},
    
    # Medium models (5M - 15M params)
    'efficientnet_b0': {'params': '5.3M', 'input_size': 224},
    'mobilenetv3_large': {'params': '5.5M', 'input_size': 224},
    'efficientnet_b1': {'params': '7.8M', 'input_size': 240},
    'densenet121': {'params': '8M', 'input_size': 224},
    'efficientnet_b2': {'params': '9.2M', 'input_size': 260},
    'resnet18': {'params': '11.7M', 'input_size': 224},
    'efficientnet_b3': {'params': '12M', 'input_size': 300},
    'densenet169': {'params': '14M', 'input_size': 224},
    
    # Large models (15M - 50M params)
    'densenet201': {'params': '20M', 'input_size': 224},
    'resnet34': {'params': '21.8M', 'input_size': 224},
    'resnet50': {'params': '25.6M', 'input_size': 224},
    'inception_v3': {'params': '27.2M', 'input_size': 299},
    'resnet101': {'params': '44.5M', 'input_size': 224},
    
    # Very large models (> 100M params) - Classic benchmarks
    'vgg16_bn': {'params': '138M', 'input_size': 224},
    'vgg19_bn': {'params': '144M', 'input_size': 224},
}


def get_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Create a CNN model with modified classifier for num_classes.
    
    Args:
        model_name: One of the keys in MODEL_INFO
        num_classes: Number of output classes (default: 3 for PD classification)
        pretrained: If True, use ImageNet pretrained weights
        
    Returns:
        nn.Module with classifier head modified for num_classes
    """
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    # =========================================================================
    # SqueezeNet family
    # =========================================================================
    if model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=weights)
        # SqueezeNet uses Conv2d as final classifier
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    
    # =========================================================================
    # ShuffleNet family
    # =========================================================================
    elif model_name == 'shufflenetv2_x0_5':
        model = models.shufflenet_v2_x0_5(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'shufflenetv2_x1_0':
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    # =========================================================================
    # MobileNet family
    # =========================================================================
    elif model_name == 'mobilenetv3_small':
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenetv3_large':
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    # =========================================================================
    # EfficientNet family (B0-B3)
    # =========================================================================
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    # =========================================================================
    # DenseNet family
    # =========================================================================
    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'densenet169':
        model = models.densenet169(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'densenet201':
        model = models.densenet201(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    # =========================================================================
    # ResNet family
    # =========================================================================
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet34':
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet101':
        model = models.resnet101(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    # =========================================================================
    # Inception family
    # =========================================================================
    elif model_name == 'inception_v3':
        # Inception V3 has auxiliary outputs during training
        model = models.inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Also modify auxiliary classifier if present
        if model.aux_logits:
            in_features_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)
    
    # =========================================================================
    # VGG family (with Batch Normalization)
    # =========================================================================
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    else:
        available = list(MODEL_INFO.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    # Store model name for later use
    model._model_name = model_name
    
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all backbone layers, only classifier head remains trainable.
    
    This is Phase 1 of transfer learning: feature extraction.
    """
    model_name = getattr(model, '_model_name', '')
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze classifier head based on model architecture
    # Group models by classifier structure for easier maintenance
    
    # Models with model.classifier (Sequential or single Linear)
    classifier_models = [
        'squeezenet1_1', 'mobilenetv3_small', 'mobilenetv3_large',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        'densenet121', 'densenet169', 'densenet201',
        'vgg16_bn', 'vgg19_bn'
    ]
    
    # Models with model.fc (single Linear)
    fc_models = [
        'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'inception_v3'
    ]
    
    if model_name in classifier_models:
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif model_name in fc_models:
        for param in model.fc.parameters():
            param.requires_grad = True
        # Special case: Inception V3 aux classifier
        if model_name == 'inception_v3' and hasattr(model, 'AuxLogits'):
            for param in model.AuxLogits.parameters():
                param.requires_grad = True
    else:
        # Fallback: try common classifier names
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all layers for fine-tuning.
    
    This is Phase 2 of transfer learning: fine-tuning entire network.
    """
    for param in model.parameters():
        param.requires_grad = True


def freeze_partial_backbone(model: nn.Module) -> None:
    """
    Freeze early backbone, unfreeze last conv block + classifier.
    
    This is the Partial Fine-tuning Baseline for few-shot style evaluation.
    Allows fine-tuning on support set while preserving early features.
    
    Architecture-specific unfreezing:
    - ResNet (all): layer4 + fc
    - EfficientNet (all): last 3 blocks + classifier  
    - DenseNet (all): denseblock4 + classifier
    - SqueezeNet: features[-3:] + classifier
    - ShuffleNet (all): stage4 + conv5 + fc
    - MobileNetV3 (all): last 3 blocks + classifier
    - VGG (all): last 3 conv blocks + classifier
    - Inception V3: Mixed_7c + fc
    """
    model_name = getattr(model, '_model_name', '')
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # =========================================================================
    # ResNet family: layer4 + fc
    # =========================================================================
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    
    # =========================================================================
    # EfficientNet family: last 3 blocks + classifier
    # =========================================================================
    elif model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']:
        # EfficientNet features has varying blocks, unfreeze last 3
        num_features = len(model.features)
        for i in range(max(0, num_features - 3), num_features):
            for param in model.features[i].parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # =========================================================================
    # DenseNet family: denseblock4 + transition3 + norm5 + classifier
    # =========================================================================
    elif model_name in ['densenet121', 'densenet169', 'densenet201']:
        for name, param in model.features.named_parameters():
            if 'denseblock4' in name or 'transition3' in name or 'norm5' in name:
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # =========================================================================
    # SqueezeNet: last fire modules + classifier
    # =========================================================================
    elif model_name == 'squeezenet1_1':
        for i in range(10, 13):
            if i < len(model.features):
                for param in model.features[i].parameters():
                    param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # =========================================================================
    # ShuffleNet family: stage4 + conv5 + fc
    # =========================================================================
    elif model_name in ['shufflenetv2_x0_5', 'shufflenetv2_x1_0']:
        for param in model.stage4.parameters():
            param.requires_grad = True
        for param in model.conv5.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    
    # =========================================================================
    # MobileNetV3 family: last 3 blocks + classifier
    # =========================================================================
    elif model_name in ['mobilenetv3_small', 'mobilenetv3_large']:
        num_features = len(model.features)
        for i in range(max(0, num_features - 3), num_features):
            for param in model.features[i].parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # =========================================================================
    # VGG family: last few conv layers + classifier
    # =========================================================================
    elif model_name in ['vgg16_bn', 'vgg19_bn']:
        # VGG features has conv + bn + relu layers, unfreeze last 6 (2 conv blocks)
        num_features = len(model.features)
        for i in range(max(0, num_features - 6), num_features):
            for param in model.features[i].parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # =========================================================================
    # Inception V3: Mixed_7c + fc
    # =========================================================================
    elif model_name == 'inception_v3':
        for param in model.Mixed_7c.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        if hasattr(model, 'AuxLogits') and model.aux_logits:
            for param in model.AuxLogits.parameters():
                param.requires_grad = True
    
    # =========================================================================
    # Fallback: just unfreeze classifier
    # =========================================================================
    else:
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True


def get_classifier_params(model: nn.Module):
    """
    Get only classifier head parameters.
    
    Returns:
        Generator of classifier parameters
    """
    model_name = getattr(model, '_model_name', '')
    
    # Models with model.classifier
    classifier_models = [
        'squeezenet1_1', 'mobilenetv3_small', 'mobilenetv3_large',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        'densenet121', 'densenet169', 'densenet201',
        'vgg16_bn', 'vgg19_bn'
    ]
    
    # Models with model.fc
    fc_models = [
        'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'inception_v3'
    ]
    
    if model_name in classifier_models:
        return model.classifier.parameters()
    elif model_name in fc_models:
        return model.fc.parameters()
    else:
        # Fallback
        if hasattr(model, 'fc'):
            return model.fc.parameters()
        elif hasattr(model, 'classifier'):
            return model.classifier.parameters()
        return iter([])  # Empty iterator


def get_backbone_params(model: nn.Module):
    """
    Get only backbone parameters (excluding classifier head).
    
    Returns:
        Generator of backbone parameters
    """
    model_name = getattr(model, '_model_name', '')
    classifier_params = set()
    
    # Get classifier parameter IDs
    for p in get_classifier_params(model):
        classifier_params.add(id(p))
    
    # Return all non-classifier parameters
    for param in model.parameters():
        if id(param) not in classifier_params:
            yield param


def get_available_models():
    """Return list of available model names."""
    return list(MODEL_INFO.keys())


def get_model_info(model_name: str = None):
    """Get info about model(s)."""
    if model_name:
        return MODEL_INFO.get(model_name, None)
    return MODEL_INFO


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def calculate_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224), device: str = 'cpu') -> dict:
    """Calculate FLOPs (MACs) for a model using thop.
    
    FLOPs (Floating Point Operations) measure computational complexity.
    Note: thop actually counts MACs (Multiply-Accumulate operations).
    FLOPs ≈ 2 × MACs (one multiply + one add per MAC).
    
    Args:
        model: PyTorch model
        input_size: Input tensor shape (batch, channels, height, width)
        device: Device to run calculation on
        
    Returns:
        dict with keys:
            - 'macs': MACs count (raw number)
            - 'flops': FLOPs count (2 × MACs)
            - 'macs_str': Human-readable MACs string (e.g., '0.39 GMACs')
            - 'flops_str': Human-readable FLOPs string (e.g., '0.78 GFLOPs')
            - 'params': Total parameters
            - 'params_str': Human-readable params string (e.g., '5.29 M')
    """
    try:
        from thop import profile, clever_format
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return {
            'macs': 0, 'flops': 0,
            'macs_str': 'N/A', 'flops_str': 'N/A',
            'params': count_parameters(model, trainable_only=False),
            'params_str': f'{count_parameters(model, trainable_only=False) / 1e6:.2f} M'
        }
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    model = model.to(device)
    
    # Store training mode and set to eval for consistent results
    was_training = model.training
    model.eval()
    
    try:
        # Profile the model
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # FLOPs = 2 × MACs (one multiply + one add)
        flops = 2 * macs
        
        # Format for human readability
        macs_str, params_str = clever_format([macs, params], "%.2f")
        flops_str = clever_format([flops], "%.2f")[0]
        
        result = {
            'macs': int(macs),
            'flops': int(flops),
            'macs_str': macs_str.replace('B', 'G'),  # Use GMACs instead of B
            'flops_str': flops_str.replace('B', 'G') + 'FLOPs',
            'params': int(params),
            'params_str': params_str
        }
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        result = {
            'macs': 0, 'flops': 0,
            'macs_str': 'N/A', 'flops_str': 'N/A',
            'params': count_parameters(model, trainable_only=False),
            'params_str': f'{count_parameters(model, trainable_only=False) / 1e6:.2f} M'
        }
    finally:
        # Restore training mode
        if was_training:
            model.train()
    
    return result


if __name__ == '__main__':
    # Quick test
    print("Available models:")
    for name, info in MODEL_INFO.items():
        model = get_model(name, num_classes=3, pretrained=False)
        total_params = count_parameters(model, trainable_only=False)
        
        # Test freeze/unfreeze
        freeze_backbone(model)
        frozen_trainable = count_parameters(model, trainable_only=True)
        
        unfreeze_backbone(model)
        unfrozen_trainable = count_parameters(model, trainable_only=True)
        
        print(f"  {name}:")
        print(f"    Total: {total_params:,}")
        print(f"    Trainable (frozen backbone): {frozen_trainable:,}")
        print(f"    Trainable (unfrozen): {unfrozen_trainable:,}")

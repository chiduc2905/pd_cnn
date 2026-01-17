"""CNN Models for PD Scalogram Classification.

Combined model registry supporting:
1. Proposed Custom CNN (Pipeline B)
2. Transfer Learning backbones (Pipeline C)  
3. Original pd_cnn models (with 2-phase training)

Architecture specifications follow paper EXACTLY.

Author: PD Analysis Team
"""
import torch
import torch.nn as nn
import torchvision.models as models

# Try to import timm for Xception and NasNetLarge
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# =============================================================================
# MODEL REGISTRY - Paper configurations + original models
# =============================================================================

MODEL_INFO = {
    # =========================================================================
    # PAPER MODELS (Transfer Learning - Pipeline C)
    # =========================================================================
    'vgg19': {'params': '144M', 'input_size': 224, 'source': 'torchvision', 'paper': True},
    'resnet50': {'params': '25.6M', 'input_size': 224, 'source': 'torchvision', 'paper': True},
    'xception': {'params': '22.9M', 'input_size': 299, 'source': 'timm', 'paper': True},
    'densenet201': {'params': '20M', 'input_size': 224, 'source': 'torchvision', 'paper': True},
    'nasnetlarge': {'params': '88.9M', 'input_size': 331, 'source': 'timm', 'paper': True},
    'efficientnetv2s': {'params': '21.5M', 'input_size': 384, 'source': 'torchvision', 'paper': True},
    'convnexttiny': {'params': '28.6M', 'input_size': 224, 'source': 'torchvision', 'paper': True},
    
    # =========================================================================
    # ORIGINAL MODELS (2-phase training supported)
    # =========================================================================
    'squeezenet1_1': {'params': '1.2M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'shufflenetv2_x0_5': {'params': '1.4M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'shufflenetv2_x1_0': {'params': '2.3M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'mobilenetv3_small': {'params': '2.5M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'efficientnet_b0': {'params': '5.3M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'mobilenetv3_large': {'params': '5.5M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'densenet121': {'params': '8M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'resnet18': {'params': '11.7M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'resnet34': {'params': '21.8M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
    'inception_v3': {'params': '27.2M', 'input_size': 299, 'source': 'torchvision', 'paper': False},
    'vgg16_bn': {'params': '138M', 'input_size': 224, 'source': 'torchvision', 'paper': False},
}


def get_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Create a CNN model with modified classifier for num_classes.
    
    Supports both paper models (Pipeline C) and original 2-phase models.
    
    Args:
        model_name: One of the keys in MODEL_INFO
        num_classes: Number of output classes (default: 3 for PD classification)
        pretrained: If True, use ImageNet pretrained weights
        
    Returns:
        nn.Module with classifier head modified for num_classes
    """
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_INFO.keys())}")
    
    info = MODEL_INFO[model_name]
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    # =========================================================================
    # PAPER MODELS
    # =========================================================================
    
    if model_name == 'vgg19':
        model = models.vgg19_bn(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'xception':
        if not HAS_TIMM:
            raise ImportError("Xception requires timm. Install: pip install timm")
        model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'densenet201':
        model = models.densenet201(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'nasnetlarge':
        if not HAS_TIMM:
            raise ImportError("NasNetLarge requires timm. Install: pip install timm")
        model = timm.create_model('nasnetalarge', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'efficientnetv2s':
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'convnexttiny':
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    # =========================================================================
    # ORIGINAL MODELS
    # =========================================================================
    
    elif model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        
    elif model_name == 'shufflenetv2_x0_5':
        model = models.shufflenet_v2_x0_5(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'shufflenetv2_x1_0':
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenetv3_small':
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenetv3_large':
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet34':
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=weights, aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if model.aux_logits:
            in_features_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)
            
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Store model name for transfer learning utilities
    model._model_name = model_name
    
    return model


# =============================================================================
# TRANSFER LEARNING UTILITIES
# =============================================================================

def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone layers, only classifier head remains trainable.
    
    For Pipeline C: Use CNNs ONLY as feature extractors (NO fine-tuning).
    """
    model_name = getattr(model, '_model_name', '')
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze classifier head
    classifier_models = [
        'squeezenet1_1', 'mobilenetv3_small', 'mobilenetv3_large',
        'efficientnet_b0', 'efficientnetv2s', 'convnexttiny',
        'densenet121', 'densenet201', 'vgg16_bn', 'vgg19'
    ]
    
    fc_models = [
        'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
        'resnet18', 'resnet34', 'resnet50',
        'inception_v3'
    ]
    
    timm_models = ['xception', 'nasnetlarge']
    
    if model_name in classifier_models:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name in fc_models:
        for param in model.fc.parameters():
            param.requires_grad = True
        if model_name == 'inception_v3' and hasattr(model, 'AuxLogits'):
            for param in model.AuxLogits.parameters():
                param.requires_grad = True
    elif model_name in timm_models:
        # timm models: find and unfreeze classifier
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all layers for fine-tuning (2-phase training)."""
    for param in model.parameters():
        param.requires_grad = True


def get_classifier_params(model: nn.Module):
    """Get only classifier head parameters."""
    model_name = getattr(model, '_model_name', '')
    
    classifier_models = [
        'squeezenet1_1', 'mobilenetv3_small', 'mobilenetv3_large',
        'efficientnet_b0', 'efficientnetv2s', 'convnexttiny',
        'densenet121', 'densenet201', 'vgg16_bn', 'vgg19'
    ]
    
    if model_name in classifier_models:
        return model.classifier.parameters()
    elif hasattr(model, 'fc'):
        return model.fc.parameters()
    elif hasattr(model, 'head'):
        return model.head.parameters()
    return iter([])


def get_backbone_params(model: nn.Module):
    """Get only backbone parameters (excluding classifier)."""
    classifier_param_ids = set(id(p) for p in get_classifier_params(model))
    for param in model.parameters():
        if id(param) not in classifier_param_ids:
            yield param


# =============================================================================
# UTILITIES
# =============================================================================

def get_available_models():
    """Return list of all available model names."""
    return list(MODEL_INFO.keys())


def get_paper_models():
    """Return list of paper-specified models only."""
    return [name for name, info in MODEL_INFO.items() if info.get('paper', False)]


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
    """Calculate FLOPs using thop."""
    try:
        from thop import profile, clever_format
    except ImportError:
        return {'flops': 0, 'flops_str': 'N/A', 'params': count_parameters(model, False)}
    
    dummy_input = torch.randn(input_size).to(device)
    model = model.to(device).eval()
    
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.2f")
        return {'macs': int(macs), 'flops': int(2*macs), 'macs_str': macs_str, 'params': int(params)}
    except:
        return {'flops': 0, 'flops_str': 'N/A', 'params': count_parameters(model, False)}


if __name__ == '__main__':
    print("Available models:")
    print("\nPaper models (Pipeline C):")
    for name in get_paper_models():
        info = MODEL_INFO[name]
        print(f"  {name}: {info['params']}, input={info['input_size']}")
    
    print("\nOriginal models (2-phase training):")
    for name, info in MODEL_INFO.items():
        if not info.get('paper', False):
            print(f"  {name}: {info['params']}, input={info['input_size']}")

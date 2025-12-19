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
MODEL_INFO = {
    'squeezenet1_1': {'params': '1.2M', 'input_size': 224},
    'shufflenetv2_x0_5': {'params': '1.4M', 'input_size': 224},
    'mobilenetv3_small': {'params': '2.5M', 'input_size': 224},
    'efficientnet_b0': {'params': '5.3M', 'input_size': 224},
    'densenet121': {'params': '8M', 'input_size': 224},
    'resnet18': {'params': '11.7M', 'input_size': 224},
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
    
    if model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=weights)
        # SqueezeNet uses Conv2d as final classifier
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
        
    elif model_name == 'shufflenetv2_x0_5':
        model = models.shufflenet_v2_x0_5(weights=weights)
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
        
    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
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
    if model_name == 'squeezenet1_1':
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif model_name == 'shufflenetv2_x0_5':
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif model_name in ['mobilenetv3_small', 'efficientnet_b0']:
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif model_name == 'densenet121':
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif model_name == 'resnet18':
        for param in model.fc.parameters():
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


def get_classifier_params(model: nn.Module):
    """
    Get only classifier head parameters.
    
    Returns:
        Generator of classifier parameters
    """
    model_name = getattr(model, '_model_name', '')
    
    if model_name == 'squeezenet1_1':
        return model.classifier.parameters()
    elif model_name == 'shufflenetv2_x0_5':
        return model.fc.parameters()
    elif model_name in ['mobilenetv3_small', 'efficientnet_b0', 'densenet121']:
        return model.classifier.parameters()
    elif model_name == 'resnet18':
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

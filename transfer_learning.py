"""Transfer Learning CNN Models with ML Classifiers (Pipeline C).

Implements frozen pretrained backbones + ML classification as per paper:
"Effectiveness of Wavelet Scalogram on Partial Discharge Pattern Classification"

PRETRAINED BACKBONES (ImageNet, FROZEN):
- VGG19
- ResNet50
- Xception (via timm)
- DenseNet201
- NasNetLarge (via timm)
- EfficientNetV2S
- ConvNeXtTiny

TRANSFER LEARNING RULES (STRICT):
- Load ImageNet weights
- Freeze ALL convolutional layers
- Use CNNs ONLY as feature extractors
- NO fine-tuning of backbone

CLASSIFICATION HEADS:
- SVM (RBF)
- Random Forest
- kNN

Author: PD Analysis Team
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Fix random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Try to import timm for Xception and NasNetLarge
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not installed. Xception and NasNetLarge unavailable.")
    print("Install with: pip install timm")


# =============================================================================
# MODEL REGISTRY - Exact paper configurations
# =============================================================================

MODEL_CONFIG = {
    'vgg19': {
        'input_size': 224,
        'source': 'torchvision',
        'feature_dim': 4096  # From classifier[0]
    },
    'resnet50': {
        'input_size': 224,
        'source': 'torchvision',
        'feature_dim': 2048  # After avgpool
    },
    'xception': {
        'input_size': 299,
        'source': 'timm',
        'feature_dim': 2048
    },
    'densenet201': {
        'input_size': 224,
        'source': 'torchvision',
        'feature_dim': 1920  # Before classifier
    },
    'nasnetlarge': {
        'input_size': 331,
        'source': 'timm',
        'feature_dim': 4032
    },
    'efficientnetv2s': {
        'input_size': 384,
        'source': 'torchvision',
        'feature_dim': 1280
    },
    'convnexttiny': {
        'input_size': 224,
        'source': 'torchvision',
        'feature_dim': 768
    }
}


# =============================================================================
# FEATURE EXTRACTOR MODELS
# =============================================================================

class FeatureExtractor(nn.Module):
    """Wrapper for extracting features from pretrained CNN.
    
    Transfer learning rules (STRICT):
    - Load ImageNet weights
    - Freeze ALL convolutional layers
    - Use CNNs ONLY as feature extractors
    - NO fine-tuning
    """
    
    def __init__(self, model_name='vgg19', device='cuda'):
        super(FeatureExtractor, self).__init__()
        
        self.model_name = model_name
        self.device = device
        self.config = MODEL_CONFIG.get(model_name)
        
        if self.config is None:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(MODEL_CONFIG.keys())}")
        
        # Load model based on source
        if self.config['source'] == 'torchvision':
            self.model = self._load_torchvision_model(model_name)
        elif self.config['source'] == 'timm':
            if not HAS_TIMM:
                raise ImportError(f"{model_name} requires timm. Install: pip install timm")
            self.model = self._load_timm_model(model_name)
        
        # Freeze ALL parameters (no fine-tuning as per paper)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.to(device)
        self.model.eval()
        
    def _load_torchvision_model(self, model_name):
        """Load model from torchvision."""
        weights = 'IMAGENET1K_V1'
        
        if model_name == 'vgg19':
            model = models.vgg19_bn(weights=weights)
            # Extract features from classifier[0] (first FC layer output)
            self.feature_layer = nn.Sequential(*list(model.features), 
                                                model.avgpool,
                                                nn.Flatten())
            return model
            
        elif model_name == 'resnet50':
            model = models.resnet50(weights=weights)
            # Remove final FC layer
            self.feature_layer = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
                model.avgpool, nn.Flatten()
            )
            return model
            
        elif model_name == 'densenet201':
            model = models.densenet201(weights=weights)
            self.feature_layer = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            return model
            
        elif model_name == 'efficientnetv2s':
            model = models.efficientnet_v2_s(weights=weights)
            self.feature_layer = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten()
            )
            return model
            
        elif model_name == 'convnexttiny':
            model = models.convnext_tiny(weights=weights)
            self.feature_layer = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten()
            )
            return model
            
        else:
            raise ValueError(f"Unsupported torchvision model: {model_name}")
    
    def _load_timm_model(self, model_name):
        """Load model from timm."""
        if model_name == 'xception':
            model = timm.create_model('xception', pretrained=True, num_classes=0)
        elif model_name == 'nasnetlarge':
            model = timm.create_model('nasnetalarge', pretrained=True, num_classes=0)
        else:
            raise ValueError(f"Unsupported timm model: {model_name}")
        
        # timm models with num_classes=0 return features directly
        self.feature_layer = model
        return model
    
    def forward(self, x):
        """Extract features (no gradients needed)."""
        with torch.no_grad():
            if self.config['source'] == 'timm':
                features = self.feature_layer(x)
            else:
                features = self.feature_layer(x)
        return features
    
    def get_feature_dim(self):
        """Get output feature dimension."""
        return self.config['feature_dim']
    
    def get_input_size(self):
        """Get required input size."""
        return self.config['input_size']


def extract_features(extractor, dataloader, device='cuda'):
    """Extract features from all images using frozen backbone.
    
    Args:
        extractor: FeatureExtractor instance
        dataloader: DataLoader with images
        device: Device to use
        
    Returns:
        features: numpy array of shape (N, feature_dim)
        labels: numpy array of shape (N,)
    """
    all_features = []
    all_labels = []
    
    extractor.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Extracting {extractor.model_name} features"):
            images = images.to(device)
            features = extractor(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)


# =============================================================================
# ML CLASSIFIERS (Same as Pipeline A)
# =============================================================================

def train_svm_on_features(X_train, y_train, cv=5):
    """Train SVM with Gaussian RBF on CNN features.
    
    Paper specification:
    - Kernel: Gaussian RBF
    - Gamma (γ): 0.0001 < γ < 1
    - Regularization (C): 0.1 < C < 1000
    """
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    
    svm = SVC(kernel='rbf', random_state=SEED, probability=True)
    
    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"  SVM Best: C={grid_search.best_params_['C']}, "
          f"gamma={grid_search.best_params_['gamma']}, "
          f"CV acc={grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_rf_on_features(X_train, y_train, cv=5):
    """Train Random Forest on CNN features.
    
    Paper specification:
    - n_estimators: 10 ≤ n ≤ 200
    """
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200]
    }
    
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"  RF Best: n_estimators={grid_search.best_params_['n_estimators']}, "
          f"CV acc={grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_knn_on_features(X_train, y_train, cv=5):
    """Train kNN on CNN features.
    
    Paper specification:
    - K range: 1 ≤ K ≤ 20
    """
    param_grid = {
        'n_neighbors': list(range(1, 21))
    }
    
    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"  kNN Best: K={grid_search.best_params_['n_neighbors']}, "
          f"CV acc={grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_classifier(model, X_test, y_test, name='Model'):
    """Evaluate classifier with paper metrics."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Sensitivity and Specificity
    num_classes = len(np.unique(y_test))
    sensitivities = []
    specificities = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    return {
        'accuracy': acc,
        'sensitivity': np.mean(sensitivities),
        'specificity': np.mean(specificities),
        'confusion_matrix': cm,
        'predictions': y_pred
    }


# =============================================================================
# MAIN PIPELINE C RUNNER
# =============================================================================

def run_pipeline_c(train_loader, test_loader, 
                   model_names=['vgg19', 'resnet50', 'densenet201'],
                   classifiers=['svm', 'rf', 'knn'],
                   device='cuda'):
    """Run Pipeline C: Transfer Learning + Shallow ML.
    
    Args:
        train_loader: DataLoader for training images
        test_loader: DataLoader for test images
        model_names: List of backbone model names
        classifiers: List of classifier types ('svm', 'rf', 'knn')
        device: Device to use
        
    Returns:
        Dict with all results
    """
    print("="*60)
    print("PIPELINE C: Transfer Learning + Shallow ML")
    print("="*60)
    print(f"Backbones: {model_names}")
    print(f"Classifiers: {classifiers}")
    print(f"Note: All backbones are FROZEN (no fine-tuning)")
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'─'*50}")
        print(f"BACKBONE: {model_name.upper()}")
        print(f"{'─'*50}")
        
        try:
            # Create feature extractor
            extractor = FeatureExtractor(model_name, device=device)
            print(f"  Input size: {extractor.get_input_size()}")
            print(f"  Feature dim: {extractor.get_feature_dim()}")
            
            # Extract features
            print("\n  Extracting training features...")
            X_train, y_train = extract_features(extractor, train_loader, device)
            
            print("  Extracting test features...")
            X_test, y_test = extract_features(extractor, test_loader, device)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            results[model_name] = {'extractor': extractor, 'scaler': scaler}
            
            # Train classifiers
            for clf_name in classifiers:
                print(f"\n  Training {clf_name.upper()}...")
                
                if clf_name == 'svm':
                    clf = train_svm_on_features(X_train_scaled, y_train)
                elif clf_name == 'rf':
                    clf = train_rf_on_features(X_train_scaled, y_train)
                elif clf_name == 'knn':
                    clf = train_knn_on_features(X_train_scaled, y_train)
                else:
                    continue
                
                # Evaluate
                eval_result = evaluate_classifier(clf, X_test_scaled, y_test, clf_name)
                
                print(f"    Test Accuracy: {eval_result['accuracy']*100:.2f}%")
                print(f"    Sensitivity: {eval_result['sensitivity']*100:.2f}%")
                print(f"    Specificity: {eval_result['specificity']*100:.2f}%")
                
                results[model_name][clf_name] = {
                    'model': clf,
                    'results': eval_result
                }
                
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            continue
    
    # Summary table
    print(f"\n{'='*60}")
    print("PIPELINE C SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Classifier':<10} {'Accuracy':<10} {'Sen.':<10} {'Spe.':<10}")
    print("-"*55)
    
    for model_name in model_names:
        if model_name in results:
            for clf_name in classifiers:
                if clf_name in results[model_name]:
                    r = results[model_name][clf_name]['results']
                    print(f"{model_name:<15} {clf_name.upper():<10} "
                          f"{r['accuracy']*100:.2f}%    "
                          f"{r['sensitivity']*100:.2f}%    "
                          f"{r['specificity']*100:.2f}%")
    
    return results


# =============================================================================
# AVAILABLE MODELS LIST
# =============================================================================

def get_available_models():
    """Return list of available backbone models."""
    return list(MODEL_CONFIG.keys())


def get_model_info():
    """Return model configuration info."""
    return MODEL_CONFIG.copy()


if __name__ == '__main__':
    print("Pipeline C: Transfer Learning + Shallow ML")
    print("\nAvailable backbones:")
    for name, config in MODEL_CONFIG.items():
        print(f"  {name}: input={config['input_size']}, features={config['feature_dim']}")
    print("\nClassifiers: SVM (RBF), Random Forest, kNN")
    print("\nNote: All backbones are FROZEN (no fine-tuning as per paper)")

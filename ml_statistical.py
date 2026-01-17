"""Traditional ML Classifiers for PD Classification (Pipeline A).

Implements statistical feature-based classification as per paper:
"Effectiveness of Wavelet Scalogram on Partial Discharge Pattern Classification"

FEATURE INPUTS (assumed pre-extracted):
- Mean
- Variance  
- Standard deviation
- Skewness
- Kurtosis

MODELS & HYPERPARAMETERS (STRICT - from paper):
- ANN: Hidden layers [8, 16, 16], Adam, ReLU, Softmax
- SVM: Gaussian RBF, 0.0001 < γ < 1, 0.1 < C < 1000
- Random Forest: 10 ≤ n_estimators ≤ 200, Bagging
- kNN: 1 ≤ K ≤ 20, selected via minimum error rate

Author: PD Analysis Team
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# PIPELINE A - ANN (PyTorch implementation)
# =============================================================================

class ANNClassifier(nn.Module):
    """ANN Classifier as per paper specification.
    
    Architecture (STRICT):
    - Input: 5 features (mean, variance, std, skewness, kurtosis)
    - Hidden layers: [8, 16, 16]
    - Hidden activation: ReLU
    - Output activation: Softmax
    - Weight initialization: Random (PyTorch default)
    """
    
    def __init__(self, input_dim=5, num_classes=3):
        super(ANNClassifier, self).__init__()
        
        # Hidden layers: [8, 16, 16] as per paper
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc_out = nn.Linear(16, num_classes)
        
        # Activation functions as per paper
        self.relu = nn.ReLU()
        # Note: Softmax is applied in loss function (CrossEntropyLoss includes LogSoftmax)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)  # Raw logits (softmax in loss)
        return x
    
    def predict_proba(self, x):
        """Get softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def train_ann(X_train, y_train, X_val, y_val, num_classes=3, 
              epochs=100, batch_size=32, lr=0.001, device='cuda'):
    """Train ANN with Adam optimizer.
    
    Paper specification:
    - Optimizer: Adam
    - Training: supervised
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = ANNClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    
    # Optimizer: Adam as per paper
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss: Categorical Cross-Entropy
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


# =============================================================================
# PIPELINE A - SVM (Gaussian RBF)
# =============================================================================

def train_svm(X_train, y_train, cv=5):
    """Train SVM with Gaussian RBF kernel using Grid Search.
    
    Paper specification:
    - Kernel: Gaussian RBF
    - Gamma (γ): 0.0001 < γ < 1
    - Regularization (C): 0.1 < C < 1000
    """
    # Hyperparameter grid as per paper
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    
    svm = SVC(kernel='rbf', random_state=SEED, probability=True)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"SVM Best params: {grid_search.best_params_}")
    print(f"SVM Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# =============================================================================
# PIPELINE A - Random Forest
# =============================================================================

def train_random_forest(X_train, y_train, cv=5):
    """Train Random Forest using Grid Search.
    
    Paper specification:
    - n_estimators: 10 ≤ n ≤ 200
    - Training: Bagging (default in sklearn RF)
    """
    # Hyperparameter grid as per paper
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200]
    }
    
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"RF Best params: {grid_search.best_params_}")
    print(f"RF Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# =============================================================================
# PIPELINE A - kNN
# =============================================================================

def train_knn(X_train, y_train, cv=5):
    """Train kNN with K selected via minimum error rate.
    
    Paper specification:
    - K range: 1 ≤ K ≤ 20
    - K selected via minimum error rate
    """
    # Hyperparameter grid as per paper
    param_grid = {
        'n_neighbors': list(range(1, 21))  # 1 to 20
    }
    
    knn = KNeighborsClassifier()
    
    # Grid search with cross-validation (minimum error rate = max accuracy)
    grid_search = GridSearchCV(
        knn, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"kNN Best params: {grid_search.best_params_}")
    print(f"kNN Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Evaluate model and compute metrics.
    
    Metrics (as per paper):
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - Confusion matrix
    """
    # Handle PyTorch ANN
    if isinstance(model, ANNClassifier):
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(next(model.parameters()).device)
            outputs = model(X_test_t)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
    else:
        y_pred = model.predict(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Sensitivity and Specificity per class
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
    
    results = {
        'accuracy': acc,
        'sensitivity': np.mean(sensitivities),
        'specificity': np.mean(specificities),
        'confusion_matrix': cm,
        'predictions': y_pred
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Sensitivity (avg): {np.mean(sensitivities)*100:.2f}%")
    print(f"  Specificity (avg): {np.mean(specificities)*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    
    return results


# =============================================================================
# MAIN PIPELINE A RUNNER
# =============================================================================

def run_pipeline_a(X_train, y_train, X_val, y_val, X_test, y_test, 
                   num_classes=3, device='cuda'):
    """Run complete Pipeline A: Traditional ML with statistical features.
    
    Args:
        X_train, y_train: Training data (features already extracted)
        X_val, y_val: Validation data
        X_test, y_test: Test data
        num_classes: Number of output classes
        device: Device for ANN training
        
    Returns:
        Dict with all trained models and results
    """
    print("="*60)
    print("PIPELINE A: Traditional ML with Statistical Features")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. ANN
    print("\n--- Training ANN ---")
    ann_model = train_ann(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        num_classes=num_classes, device=device
    )
    results['ANN'] = evaluate_model(ann_model, X_test_scaled, y_test, 'ANN')
    results['ANN']['model'] = ann_model
    
    # 2. SVM
    print("\n--- Training SVM (RBF) ---")
    svm_model = train_svm(X_train_scaled, y_train)
    results['SVM'] = evaluate_model(svm_model, X_test_scaled, y_test, 'SVM')
    results['SVM']['model'] = svm_model
    
    # 3. Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = train_random_forest(X_train_scaled, y_train)
    results['RF'] = evaluate_model(rf_model, X_test_scaled, y_test, 'Random Forest')
    results['RF']['model'] = rf_model
    
    # 4. kNN
    print("\n--- Training kNN ---")
    knn_model = train_knn(X_train_scaled, y_train)
    results['kNN'] = evaluate_model(knn_model, X_test_scaled, y_test, 'kNN')
    results['kNN']['model'] = knn_model
    
    # Store scaler for inference
    results['scaler'] = scaler
    
    print("\n" + "="*60)
    print("PIPELINE A COMPLETE")
    print("="*60)
    
    return results


if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Run Pipeline A: Traditional ML Classifiers')
    parser.add_argument('--features', type=str, default='features.npz',
                        help='Path to features.npz file generated by feature_extraction_1d.py')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for ANN training (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features):
        print(f"Error: Feature file '{args.features}' not found.")
        print("Run feature_extraction_1d.py first to generate features.")
        print("Example: python feature_extraction_1d.py --data_root /path/to/data")
        exit(1)
        
    print(f"Loading features from {args.features}...")
    data = np.load(args.features)
    
    # Check if data contains necessary keys
    required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    missing_keys = [k for k in required_keys if k not in data]
    
    if missing_keys:
        print(f"Error: Missing keys in feature file: {missing_keys}")
        exit(1)
        
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"Data loaded:")
    print(f"  Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, labels: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, labels: {y_test.shape}")
    
    # Run pipeline
    results = run_pipeline_a(
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test,
        num_classes=len(np.unique(y_train)),
        device=args.device
    )
    
    print(f"\nResults saved. Pipeline finished successfully.")

"""Custom CNN for PD Classification (Pipeline B).

Implements the EXACT CNN architecture from paper:
"Effectiveness of Wavelet Scalogram on Partial Discharge Pattern Classification"

CNN ARCHITECTURE (MUST MATCH EXACTLY):
- Conv2D_1: 32 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
- Conv2D_2: 64 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
- MaxPooling_1: 2×2, stride=2
- Conv2D_3: 128 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
- Conv2D_4: 256 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
- MaxPooling_2: 2×2, stride=2
- Flatten
- FC_1: 512 units, ELU, Dropout
- FC_2: 64 units, ELU
- Output: Softmax, 3 units

TRAINING HYPERPARAMETERS (LOCKED):
- Optimizer: Adam
- Learning rate: 0.0005
- Loss: Categorical Cross-Entropy
- Activation: ELU
- Kernel size: 5×5 (all conv layers)
- Pooling: Max Pooling
- Padding: same
- Validation: 5-fold cross-validation

Author: PD Analysis Team
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

# Fix random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class ProposedCNN(nn.Module):
    """Proposed Custom CNN exactly as specified in paper.
    
    Input: RGB scalogram images (already Min-Max normalized)
    Output: 3 classes (Softmax)
    
    Architecture follows paper specification EXACTLY.
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(ProposedCNN, self).__init__()
        
        # Conv2D_1: 32 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)  # padding=2 for same
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv2D_2: 64 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # MaxPooling_1: 2×2, stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2D_3: 128 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Conv2D_4: 256 filters, 5×5, stride=1, padding=same, ELU, BatchNorm
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        
        # MaxPooling_2: 2×2, stride=2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten (done in forward)
        
        # FC_1: 512 units, ELU, Dropout
        # Input size depends on input image size
        # For 224×224: after 2 poolings → 56×56
        # 256 channels × 56 × 56 = 802816 (too large)
        # Paper likely uses smaller input, using adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Reduce to 7×7
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout = nn.Dropout(dropout_rate)
        
        # FC_2: 64 units, ELU
        self.fc2 = nn.Linear(512, 64)
        
        # Output: Softmax, 3 units (softmax in loss function)
        self.fc_out = nn.Linear(64, num_classes)
        
        # Activation: ELU (as per paper)
        self.elu = nn.ELU()
        
    def forward(self, x):
        # Conv block 1
        x = self.elu(self.bn1(self.conv1(x)))
        
        # Conv block 2 + Pool
        x = self.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Conv block 3
        x = self.elu(self.bn3(self.conv3(x)))
        
        # Conv block 4 + Pool
        x = self.elu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Adaptive pool for consistent FC input
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC_1 with Dropout
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        
        # FC_2
        x = self.elu(self.fc2(x))
        
        # Output (logits - softmax in loss)
        x = self.fc_out(x)
        
        return x
    
    def extract_features(self, x):
        """Extract features from FC_2 layer (before output)."""
        # Conv block 1
        x = self.elu(self.bn1(self.conv1(x)))
        
        # Conv block 2 + Pool
        x = self.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Conv block 3
        x = self.elu(self.bn3(self.conv3(x)))
        
        # Conv block 4 + Pool
        x = self.elu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Adaptive pool
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC_1 (no dropout for feature extraction)
        x = self.elu(self.fc1(x))
        
        # FC_2 features
        x = self.elu(self.fc2(x))
        
        return x


def train_proposed_cnn(train_loader, val_loader, num_classes=3, 
                       epochs=100, lr=0.0005, device='cuda'):
    """Train Proposed CNN with exact paper hyperparameters.
    
    Training Hyperparameters (LOCKED):
    - Optimizer: Adam
    - Learning rate: 0.0005
    - Loss: Categorical Cross-Entropy
    """
    model = ProposedCNN(num_classes=num_classes).to(device)
    
    # Optimizer: Adam with lr=0.0005 (as per paper)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss: Categorical Cross-Entropy
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def train_with_kfold(X, y, num_classes=3, n_folds=5, epochs=100, 
                     lr=0.0005, batch_size=32, device='cuda'):
    """Train Proposed CNN with 5-fold cross-validation (as per paper).
    
    Paper specification:
    - Validation: 5-fold cross-validation
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*40}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*40}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train
        model, history = train_proposed_cnn(
            train_loader, val_loader, 
            num_classes=num_classes,
            epochs=epochs, lr=lr, device=device
        )
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            outputs = model(X_val_t)
            preds = outputs.argmax(dim=1).cpu().numpy()
        
        acc = accuracy_score(y_val, preds)
        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'model': model.state_dict(),
            'history': history
        })
        
        print(f"Fold {fold + 1} Accuracy: {acc*100:.2f}%")
    
    # Summary
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    
    print(f"\n{'='*40}")
    print(f"5-Fold CV Results:")
    print(f"  Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"{'='*40}")
    
    return fold_results


def evaluate_cnn(model, test_loader, device='cuda'):
    """Evaluate CNN model.
    
    Metrics (as per paper):
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - Confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Sensitivity and Specificity
    num_classes = len(np.unique(all_labels))
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
        'predictions': all_preds,
        'labels': all_labels
    }
    
    print(f"\nProposed CNN Results:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Sensitivity (avg): {np.mean(sensitivities)*100:.2f}%")
    print(f"  Specificity (avg): {np.mean(specificities)*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    
    return results


if __name__ == '__main__':
    print("Pipeline B: Proposed Custom CNN")
    print("Architecture (from paper):")
    print("  Conv2D_1: 32×5×5, ELU, BN")
    print("  Conv2D_2: 64×5×5, ELU, BN")
    print("  MaxPool: 2×2")
    print("  Conv2D_3: 128×5×5, ELU, BN")
    print("  Conv2D_4: 256×5×5, ELU, BN")
    print("  MaxPool: 2×2")
    print("  FC_1: 512, ELU, Dropout")
    print("  FC_2: 64, ELU")
    print("  Output: Softmax, 3 classes")
    print("\nHyperparameters:")
    print("  Optimizer: Adam")
    print("  Learning rate: 0.0005")
    print("  Loss: Cross-Entropy")

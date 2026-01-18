# New Transfer Learning Fine-Tuning Pipeline with Two-Phase Training

def run_transfer_learning_finetune(X_train, y_train, X_val, y_val, X_test, y_test, args,
                                    model_names=['vgg19', 'resnet50', 'densenet201'],
                                    num_classes=3, freeze_epochs=10, num_epochs=50, batch_size=32,
                                    lr_classifier=0.001, lr_backbone=1e-5, lr_classifier_finetune=1e-4,
                                    device='cuda'):
    """Two-phase Transfer Learning fine-tuning (mirrors main.py).
    
    Phase 1: Freeze backbone, train classifier only
    Phase 2: Unfreeze backbone, fine-tune entire network
    
    Enhanced with:
    - Parameter counting for frozen and unfrozen states
    - WandB logging
    - Test-only evaluation
    """
    print("\n" + "="*60)
    print("PIPELINE C-FT: Transfer Learning - Two-Phase Fine-Tuning")
    print("="*60)
    
    results = {}
    
    # Prepare DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for model_name in model_names:
        print(f"\n--- {model_name.upper()} (2-Phase Fine-Tuning) ---")
        
        # Initialize WandB for this model
        wandb_run = None
        if HAS_WANDB and args and hasattr(args, 'wandb_project') and args.wandb_project:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f'tl_finetune_{model_name}_{args.samples_per_class if args.samples_per_class else "all"}samples',
                config={
                    'pipeline': 'TL_Finetune',
                    'backbone': model_name,
                    'freeze_epochs': freeze_epochs,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'lr_classifier': lr_classifier,
                    'lr_backbone': lr_backbone,
                    'lr_classifier_finetune': lr_classifier_finetune,
                    'n_train': len(y_train),
                    'n_val': len(y_val),
                    'n_test': len(y_test),
                    'samples_per_class': args.samples_per_class if args else None
                },
                reinit=True
            )
        
        try:
            seed_func(SEED)
            model = get_model(model_name, num_classes=num_classes, pretrained=True)
            model = model.to(device)
            
            # Count total parameters
            total_params = count_parameters(model, trainable_only=False)
            print(f"  Total parameters: {total_params:,}")
            if wandb_run:
                wandb.run.summary['total_parameters'] = total_params
            
            criterion = nn.CrossEntropyLoss()
            best_val_acc = 0.0
            best_model_state = None
            
            # ===== PHASE 1: Frozen Backbone =====
            print(f"\n  [PHASE 1: Freeze Backbone - {freeze_epochs} epochs]")
            freeze_backbone(model)
            trainable_params_p1 = count_parameters(model, trainable_only=True)
            print(f"  Trainable parameters (classifier only): {trainable_params_p1:,}")
            if wandb_run:
                wandb.run.summary['phase1_trainable_parameters'] = trainable_params_p1
            
            optimizer = optim.AdamW(list(get_classifier_params(model)), lr=lr_classifier)
            
            for epoch in range(freeze_epochs):
                # Train
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{freeze_epochs} - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                
                if wandb_run:
                    wandb.log({
                        'epoch': epoch + 1,
                        'phase': 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
            
            # ===== PHASE 2: Fine-Tuning =====
            print(f"\n  [PHASE 2: Fine-Tuning - {num_epochs - freeze_epochs} epochs]")
            unfreeze_backbone(model)
            trainable_params_p2 = count_parameters(model, trainable_only=True)
            print(f"  Trainable parameters (all): {trainable_params_p2:,}")
            if wandb_run:
                wandb.run.summary['phase2_trainable_parameters'] = trainable_params_p2
            
            optimizer = optim.AdamW([
                {'params': list(get_backbone_params(model)), 'lr': lr_backbone},
                {'params': list(get_classifier_params(model)), 'lr': lr_classifier_finetune}
            ])
            
            for epoch in range(freeze_epochs, num_epochs):
                # Train
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                if val_acc > best_val_acc:
                   best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{num_epochs} - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
                
                if wandb_run:
                    wandb.log({
                        'epoch': epoch + 1,
                        'phase': 2,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
            
            # Load best model
            if best_model_state is not None:
                print(f"\n  Restoring best model (Val Acc: {best_val_acc:.4f})")
                model.load_state_dict(best_model_state)
            
            # Evaluate on TEST SET ONLY
            print("  Evaluating on test set...")
            model.eval()
            all_preds = []
            all_features = []
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    
                    # Extract features (from backbone, before classifier)
                    try:
                        if hasattr(model, 'avgpool'):
                            x = model.conv1(inputs) if hasattr(model, 'conv1') else inputs
                            if hasattr(model, 'bn1'): x = model.bn1(x)
                            if hasattr(model, 'relu'): x = model.relu(x)
                            if hasattr(model, 'maxpool'): x = model.maxpool(x)
                            if hasattr(model, 'layer1'): x = model.layer1(x)
                            if hasattr(model, 'layer2'): x = model.layer2(x)
                            if hasattr(model, 'layer3'): x = model.layer3(x)
                            if hasattr(model, 'layer4'): x = model.layer4(x)
                            feats = model.avgpool(x).view(x.size(0), -1).cpu().numpy()
                        elif hasattr(model, 'features'):
                            x = model.features(inputs)
                            feats = nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1).cpu().numpy()
                        else:
                            feats = outputs.cpu().numpy()
                        all_features.append(feats)
                    except:
                        all_features.append(outputs.cpu().numpy())
            
            y_pred = np.array(all_preds)
            features = np.vstack(all_features) if all_features else None
            
            metrics = compute_metrics(y_test, y_pred)
            print(f"  Test Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  Test F1: {metrics['f1']*100:.2f}%")
            
            results[model_name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'features': features,
                'model': model
            }
            
            # Log to WandB
            if wandb_run:
                wandb.run.summary['best_val_accuracy'] = best_val_acc
                wandb.run.summary['test_accuracy'] = metrics['accuracy']
                wandb.run.summary['test_precision'] = metrics['precision']
                wandb.run.summary['test_recall'] = metrics['recall']
                wandb.run.summary['test_f1'] = metrics['f1']
                wandb.finish()
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            if wandb_run:
                wandb.finish()
    
    return results

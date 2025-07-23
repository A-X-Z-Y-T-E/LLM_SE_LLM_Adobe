"""
Improved training pipeline for updated_model_2.pth with fine-tuned class balancing.
Addresses H4/TITLE over-detection and HH1 under-detection specifically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from typing import Dict, List
from collections import Counter
import time
import numpy as np

# Add model_training to path
sys.path.append(str(Path(__file__).parent / "model_training"))

from model_training.models import DocumentGNN
from model_training.dataset import create_data_loaders


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance with better precision control."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedTrainer:
    """Advanced trainer with focal loss and precision-focused metrics."""
    
    def __init__(self, model, device, learning_rate=0.0001, class_weights=None, use_focal_loss=True):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.use_focal_loss = use_focal_loss
        
        # Setup loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights.to(device) if class_weights is not None else None, 
                                     gamma=2.0)
            print("🎯 Using Focal Loss for better precision control")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        
        # Setup optimizer with weight decay
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=8, verbose=True
        )
        
        # Track best model based on balanced F1 score
        self.best_balanced_f1 = 0.0
        self.best_model_state = None
        self.best_epoch = 0
    
    def calculate_class_metrics(self, predictions, targets, num_classes=6):
        """Calculate precision, recall, F1 for each class."""
        label_names = ['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE']
        metrics = {}
        
        for class_id in range(num_classes):
            tp = ((predictions == class_id) & (targets == class_id)).sum().item()
            fp = ((predictions == class_id) & (targets != class_id)).sum().item()
            fn = ((predictions != class_id) & (targets == class_id)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[label_names[class_id]] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': (targets == class_id).sum().item()
            }
        
        # Calculate macro F1 (excluding BODY for structural focus)
        structural_classes = ['HH1', 'HH2', 'HH3', 'H4', 'TITLE']
        structural_f1s = [metrics[cls]['f1'] for cls in structural_classes if metrics[cls]['support'] > 0]
        macro_structural_f1 = sum(structural_f1s) / len(structural_f1s) if structural_f1s else 0.0
        
        # Calculate balanced F1 (weighted by inverse frequency)
        weights = {'BODY': 0.2, 'HH1': 0.3, 'HH2': 0.1, 'HH3': 0.1, 'H4': 0.15, 'TITLE': 0.15}
        balanced_f1 = sum(weights[cls] * metrics[cls]['f1'] for cls in label_names)
        
        return metrics, macro_structural_f1, balanced_f1
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader):
        """Validate with detailed metrics."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                total_correct += (pred == batch.y).sum().item()
                total_samples += batch.y.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        # Calculate detailed metrics
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        class_metrics, structural_f1, balanced_f1 = self.calculate_class_metrics(all_predictions, all_targets)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'structural_f1': structural_f1,
            'balanced_f1': balanced_f1
        }
    
    def train_with_early_stopping(self, train_loader, val_loader, num_epochs=150, patience=25):
        """Train with early stopping based on balanced F1 score."""
        print(f"🚀 Starting advanced training for {num_epochs} epochs...")
        print(f"   🎯 Optimization target: Balanced F1 score")
        print(f"   🛑 Early stopping patience: {patience} epochs")
        
        history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'balanced_f1': [], 'structural_f1': [], 'class_metrics_history': []
        }
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['balanced_f1'])
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['balanced_f1'].append(val_metrics['balanced_f1'])
            history['structural_f1'].append(val_metrics['structural_f1'])
            history['class_metrics_history'].append(val_metrics['class_metrics'])
            
            # Check if this is the best model
            is_best = False
            if val_metrics['balanced_f1'] > self.best_balanced_f1:
                self.best_balanced_f1 = val_metrics['balanced_f1']
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
                is_best = True
                print(f"   📈 New best balanced F1: {self.best_balanced_f1:.4f} at epoch {self.best_epoch}")
            else:
                epochs_without_improvement += 1
            
            # Print detailed progress every 10 epochs
            if (epoch + 1) % 10 == 0 or is_best:
                print(f"   Epoch {epoch+1:3d}/{num_epochs}:")
                print(f"      Acc: {val_metrics['accuracy']:.4f}, Balanced F1: {val_metrics['balanced_f1']:.4f}")
                
                # Show per-class F1 scores
                print(f"      Class F1: ", end="")
                for cls in ['BODY', 'HH1', 'H4', 'TITLE']:
                    f1 = val_metrics['class_metrics'].get(cls, {}).get('f1', 0)
                    print(f"{cls}:{f1:.3f} ", end="")
                print()
                
                # Show precision/recall for problematic classes
                if is_best:
                    for cls in ['HH1', 'H4', 'TITLE']:
                        metrics = val_metrics['class_metrics'].get(cls, {})
                        p, r = metrics.get('precision', 0), metrics.get('recall', 0)
                        print(f"         {cls}: P={p:.3f} R={r:.3f}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"   🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Loaded best model from epoch {self.best_epoch}")
            print(f"   🎯 Best balanced F1: {self.best_balanced_f1:.4f}")
        
        return history


def calculate_fine_tuned_class_weights(ml_vectors_path: str) -> torch.Tensor:
    """Calculate fine-tuned class weights based on current model behavior."""
    print("📊 Calculating fine-tuned class weights based on analysis...")
    
    # Load ML vectors
    with open(ml_vectors_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = data['labels']
    label_counts = Counter(labels)
    
    label_names = ['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE']
    
    print("   Current distribution:")
    for i, name in enumerate(label_names):
        count = label_counts.get(i, 0)
        percentage = (count / len(labels)) * 100
        print(f"      {name}: {count} ({percentage:.1f}%)")
    
    # Fine-tuned weights based on analysis:
    # - BODY: Performing well, keep moderate weight
    # - HH1: Under-detecting (6/22), increase weight significantly
    # - HH2, HH3: Rare classes, standard high weight
    # - H4: Over-detecting (25/1), reduce weight significantly
    # - TITLE: Over-detecting (4/1), reduce weight moderately
    
    fine_tuned_weights = {
        'BODY': 1.0,      # Baseline - good performance
        'HH1': 15.0,      # Increase significantly (was 12.0)
        'HH2': 10.0,      # Standard for rare class
        'HH3': 10.0,      # Standard for rare class
        'H4': 2.0,        # Reduce significantly (was 4.0)
        'TITLE': 2.5      # Reduce moderately (was 3.0)
    }
    
    print("   🔧 Fine-tuned class weights:")
    for name, weight in fine_tuned_weights.items():
        print(f"      {name}: {weight}")
    
    # Convert to tensor
    weight_values = [fine_tuned_weights[name] for name in label_names]
    return torch.tensor(weight_values, dtype=torch.float32)


def train_updated_model_2(ml_vectors_path: str, output_path: str = "updated_model_2.pth"):
    """Train updated_model_2 with fine-tuned class balancing."""
    print("🚀 TRAINING UPDATED_MODEL_2 - FINE-TUNED CLASS BALANCING")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Calculate fine-tuned class weights
    class_weights_tensor = calculate_fine_tuned_class_weights(ml_vectors_path)
    
    # Load dataset
    print("\n📁 Loading training graphs...")
    try:
        # Try different possible graph directories
        for graphs_dir in ["training_data/graphs/raw", "training_data/graphs", "graphs/raw", "graphs"]:
            try:
                train_loader, val_loader, test_loader = create_data_loaders(
                    dataset_path=graphs_dir,
                    batch_size=4
                )
                print(f"   ✅ Loaded from: {graphs_dir}")
                break
            except:
                continue
        else:
            raise FileNotFoundError("No graphs found in any expected directory")
            
    except Exception as e:
        print(f"❌ Failed to load graphs: {e}")
        return None
    
    print(f"   📚 Train batches: {len(train_loader)}")
    print(f"   🔍 Val batches: {len(val_loader)}")
    print(f"   🧪 Test batches: {len(test_loader)}")
    
    # Create improved model with enhanced architecture
    print("\n🏗️ Creating improved model...")
    model = DocumentGNN(
        num_node_features=22,
        hidden_dim=96,      # Increased capacity
        num_classes=6,
        num_layers=3,       # More layers for better learning
        dropout=0.3         # Higher dropout for regularization
    ).to(device)
    
    # Create improved trainer
    trainer = ImprovedTrainer(
        model=model,
        device=device,
        learning_rate=0.0001,
        class_weights=class_weights_tensor,
        use_focal_loss=True
    )
    
    # Train with improved settings
    print(f"\n🚀 Starting fine-tuned training...")
    print(f"   🎯 Target: Better HH1 detection, reduced H4/TITLE over-detection")
    print(f"   📈 Focal Loss: γ=2.0 for hard example focus")
    print(f"   ⚖️ Fine-tuned weights: HH1↑15.0, H4↓2.0, TITLE↓2.5")
    print(f"   🧠 Enhanced architecture: 96 hidden, 3 layers, 0.3 dropout")
    print(f"   📚 Extended training: 150 epochs with 25 patience")
    
    start_time = time.time()
    
    history = trainer.train_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=150,
        patience=25
    )
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time/60:.1f} minutes")
    
    # Test the best model
    print(f"\n🧪 Testing best model...")
    test_metrics = trainer.validate(test_loader)
    
    print(f"\n📊 Final Results:")
    print(f"   🏆 Best balanced F1: {trainer.best_balanced_f1:.4f} (epoch {trainer.best_epoch})")
    print(f"   🎯 Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   📈 Test balanced F1: {test_metrics['balanced_f1']:.4f}")
    
    # Show detailed class performance
    print(f"\n📊 Final Class Performance:")
    for class_name, metrics in test_metrics['class_metrics'].items():
        p, r, f1 = metrics['precision'], metrics['recall'], metrics['f1']
        support = metrics['support']
        print(f"   {class_name:8s}: P={p:.3f} R={r:.3f} F1={f1:.3f} (n={support})")
    
    # Save the best model
    print(f"\n💾 Saving model to: {output_path}")
    
    save_data = {
        'model_state_dict': trainer.best_model_state,
        'model_config': {
            'num_node_features': 22,
            'hidden_dim': 96,
            'num_classes': 6,
            'num_layers': 3,
            'dropout': 0.3
        },
        'training_config': {
            'epochs_trained': trainer.best_epoch,
            'total_epochs': 150,
            'learning_rate': 0.0001,
            'batch_size': 4,
            'class_weights': {name: weight for name, weight in zip(
                ['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE'],
                class_weights_tensor.tolist()
            )},
            'optimizer': 'Adam',
            'loss_function': 'FocalLoss',
            'focal_gamma': 2.0,
            'weight_decay': 1e-4,
            'early_stopping': True,
            'patience': 25
        },
        'metrics': {
            'best_balanced_f1': trainer.best_balanced_f1,
            'best_epoch': trainer.best_epoch,
            'final_test_accuracy': test_metrics['accuracy'],
            'final_test_balanced_f1': test_metrics['balanced_f1'],
            'training_history': history,
            'training_time_minutes': training_time/60,
            'final_class_metrics': test_metrics['class_metrics']
        },
        'label_mapping': {i: name for i, name in enumerate(['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE'])},
        'model_version': 'updated_model_2_fine_tuned_focal_loss'
    }
    
    torch.save(save_data, output_path)
    
    print(f"✅ Model saved successfully!")
    print(f"   📊 Best epoch: {trainer.best_epoch}")
    print(f"   🎯 Best balanced F1: {trainer.best_balanced_f1:.4f}")
    print(f"   💾 File: {output_path}")
    
    return output_path


def quick_retrain():
    """Quick retrain with fine-tuned settings."""
    print("🚀 QUICK FINE-TUNED RETRAIN - Model V2")
    print("=" * 50)
    
    # Use first available ML vectors file
    ml_vectors_dir = Path("training_data/ml_ready_vectors")
    if not ml_vectors_dir.exists():
        print("❌ ML vectors directory not found: training_data/ml_ready_vectors")
        return None
    
    json_files = list(ml_vectors_dir.glob("*.json"))
    if not json_files:
        print("❌ No ML vector files found")
        return None
    
    sample_file = json_files[0]
    print(f"📊 Using {sample_file.name} for fine-tuned weight calculation")
    
    try:
        model_path = train_updated_model_2(str(sample_file))
        return model_path
    except Exception as e:
        print(f"❌ Fine-tuned retrain failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main interface for improved retraining."""
    print("🔧 IMPROVED MODEL TRAINING V2")
    print("=" * 50)
    print("🎯 Addresses specific issues:")
    print("   📈 HH1 under-detection (6/22) → weight ↑ to 15.0")
    print("   📉 H4 over-detection (25/1) → weight ↓ to 2.0")
    print("   📉 TITLE over-detection (4/1) → weight ↓ to 2.5")
    print()
    
    while True:
        print("Training Options:")
        print("1. 🚀 Quick fine-tuned retrain (RECOMMENDED)")
        print("2. 🔧 Custom fine-tuned retrain")
        print("3. 📊 Compare weight strategies")
        print("4. 🧪 Test updated_model_2.pth (NEW)")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            model_path = quick_retrain()
            if model_path:
                print(f"\n🎉 Fine-tuned retraining successful!")
                print(f"   💾 Model saved: {model_path}")
                print(f"   🧪 Test with test_ml_vectors.py to see improvements")
                
                print(f"\n🔍 Expected improvements:")
                print(f"   📈 HH1 detection: Should improve from 6/22 to ~15/22")
                print(f"   📉 H4 detection: Should reduce from 25/1 to ~2-3/1")
                print(f"   📉 TITLE detection: Should reduce from 4/1 to ~2/1")
                print(f"   🎯 Overall balance: Better precision/recall trade-off")
        
        elif choice == "2":
            ml_file = input("Enter path to ML vectors JSON file: ").strip()
            output_file = input("Output model name (default: updated_model_2.pth): ").strip()
            if not output_file:
                output_file = "updated_model_2.pth"
            
            try:
                model_path = train_updated_model_2(ml_file, output_file)
                if model_path:
                    print(f"\n🎉 Custom fine-tuned training successful!")
                    print(f"   💾 Model saved: {model_path}")
            except Exception as e:
                print(f"❌ Custom training failed: {e}")
        
        elif choice == "3":
            print(f"\n📊 WEIGHT STRATEGY COMPARISON:")
            print(f"   🔴 Original Model:")
            print(f"      No class weights → Heavy BODY bias")
            print(f"   🟡 Updated Model 1:")
            print(f"      Standard weights → Over-correction for H4/TITLE")
            print(f"   🟢 Updated Model 2 (This):")
            print(f"      Fine-tuned weights → Balanced detection")
            print()
            print(f"   Weight Evolution:")
            print(f"      BODY:  1.0 → 1.0 → 1.0  (stable)")
            print(f"      HH1:   1.0 → 8.0 → 15.0 (increased for better recall)")
            print(f"      H4:    1.0 → 200 → 2.0  (reduced to prevent over-detection)")
            print(f"      TITLE: 1.0 → 33 → 2.5   (reduced to prevent over-detection)")
                
        elif choice == "4":
            print("\n🧪 TESTING UPDATED_MODEL_2")
            print("=" * 40)
            print("🎯 This will test your newly trained model with improved class weights")
            print("   Expected improvements:")
            print("   📈 HH1: 6/22 → ~15/22 (better recall)")
            print("   📉 H4: 25/1 → ~2-3/1 (better precision)")
            print("   📉 TITLE: 4/1 → ~2/1 (better precision)")
            print("   🎯 Overall: More balanced structural detection")
            print()
            
            # Test the model using the existing test script
            try:
                # Import the ML vector tester
                sys.path.append(str(Path(__file__).parent))
                from test_ml_vectors import MLVectorTester
                
                # Check if updated_model_2.pth exists
                model_path = "updated_model_2.pth"
                if not Path(model_path).exists():
                    print(f"❌ Model not found: {model_path}")
                    print("   Please run option 1 to train the model first")
                    continue
                
                print(f"📦 Testing model: {model_path}")
                
                # Initialize tester with the new model
                tester = MLVectorTester(model_path)
                
                # Use first available ML vectors file for testing
                ml_vectors_dir = Path("training_data/ml_ready_vectors")
                json_files = list(ml_vectors_dir.glob("*.json"))
                
                if not json_files:
                    print("❌ No ML vector files found for testing")
                    continue
                
                test_file = json_files[0]
                print(f"🧪 Testing with: {test_file.name}")
                
                # Run the test
                result = tester.test_with_ml_vectors(str(test_file))
                
                if result:
                    print(f"\n📊 🎉 UPDATED_MODEL_2 RESULTS:")
                    print(f"   🎯 Accuracy: {result['accuracy']:.3f}")
                    print(f"   📈 Structural ratio: {result['structural_ratio']:.3f}")
                    
                    # Detailed analysis
                    expected_structural = sum(count for label, count in result['true_distribution'].items() 
                                            if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
                    predicted_structural = sum(count for label, count in result['prediction_distribution'].items() 
                                             if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
                    
                    print(f"\n🔍 IMPROVEMENT ANALYSIS:")
                    print(f"   📊 Expected structural elements: {expected_structural}")
                    print(f"   🤖 Predicted structural elements: {predicted_structural}")
                    print(f"   📈 Structural prediction ratio: {result['structural_ratio']:.3f}")
                    
                    # Compare with previous results
                    print(f"\n📊 COMPARISON WITH PREVIOUS MODELS:")
                    print(f"   🔴 Original model: ~20% structural detection")
                    print(f"   🟡 Updated model 1: 150% structural detection (over-correction)")
                    print(f"   🟢 Updated model 2: {result['structural_ratio']*100:.1f}% structural detection")
                    
                    # Class-wise performance analysis
                    print(f"\n📊 Class-wise Performance:")
                    for class_name, pred_count in result['prediction_distribution'].items():
                        true_count = result['true_distribution'].get(class_name, 0)
                        if true_count > 0:
                            ratio = pred_count / true_count
                            if 0.8 <= ratio <= 1.2:
                                status = "🎯 Perfect balance"
                            elif ratio > 1.2:
                                status = "📈 Over-detecting (good sensitivity)"
                            elif ratio >= 0.5:
                                status = "⚠️ Under-detecting"
                            else:
                                status = "❌ Poor detection"
                            print(f"   {class_name:8s}: {pred_count:2d}/{true_count:2d} predicted/actual - {status}")
                    
                    # Final assessment
                    if result['structural_ratio'] >= 1.0 and result['accuracy'] >= 0.7:
                        print(f"\n🎉 EXCELLENT RESULTS!")
                        print(f"   ✅ Model successfully balanced structural detection!")
                        print(f"   🚀 Ready for production use in outline generation!")
                    elif result['structural_ratio'] >= 0.8:
                        print(f"\n🎊 GREAT IMPROVEMENT!")
                        print(f"   ✅ Significant improvement over previous models!")
                        print(f"   🔧 Minor fine-tuning could improve it further")
                    elif result['structural_ratio'] >= 0.6:
                        print(f"\n👍 GOOD PROGRESS!")
                        print(f"   ✅ Better than original but still improving")
                        print(f"   💡 Consider further weight adjustments")
                    else:
                        print(f"\n🤔 NEEDS MORE WORK")
                        print(f"   ⚠️ Less improvement than expected")
                        print(f"   💡 May need different training approach")
                
            except ImportError:
                print("❌ test_ml_vectors.py not found. Please run:")
                print("   python test_ml_vectors.py")
                print("   Then choose option 2 to test the updated model")
            except Exception as e:
                print(f"❌ Testing failed: {e}")
                print("\nAlternative: Run manually:")
                print("   python test_ml_vectors.py")
                print("   Choose option 2 (Test with sample ML vectors)")
        
        elif choice == "5":
            print("👋 Fine-tuned training session ended!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

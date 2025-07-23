"""
Complete training pipeline for Document Layout Analysis GNN.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add model_training to path
sys.path.append(str(Path(__file__).parent / "model_training"))

from model_training.models import DocumentGNN, print_model_info
from model_training.dataset import create_data_loaders, analyze_dataset
from model_training.trainer import GNNTrainer, main_training_pipeline


def quick_train():
    """Quick training with default settings."""
    print("🚀 QUICK TRAINING - Document Layout Analysis GNN")
    print("=" * 60)
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   💎 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Dataset path
    dataset_path = "training_data/graphs"
    
    try:
        # Run complete training pipeline with optimized settings for RTX 4050
        history, optimized_model_path = main_training_pipeline(
            dataset_path=dataset_path,
            model_type='standard',
            hidden_dim=64,          # Good balance for RTX 4050
            num_epochs=50,          # Reasonable for initial training
            batch_size=4,           # Increased from 2 (RTX 4050 can handle this)
            learning_rate=0.001
        )
        
        print(f"\n🎉 Training complete!")
        print(f"   📊 Final model: {optimized_model_path}")
        
        return history, optimized_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def custom_train():
    """Custom training with user options."""
    print("🔧 CUSTOM TRAINING SETUP")
    print("=" * 40)
    
    # Show current dataset info
    print("📊 Your dataset:")
    print("   📈 Total graphs: 964")
    print("   🎓 Train: 674 graphs (337 batches)")
    print("   🔍 Val: 144 graphs (72 batches)")
    print("   🧪 Test: 146 graphs (73 batches)")
    print("   🔧 Features per node: 22")
    print()
    
    # Get user preferences
    model_type = input("Model type (standard/compact, default: standard): ").strip() or 'standard'
    hidden_dim = int(input("Hidden dimension (32/64/128, default: 64): ").strip() or '64')
    num_epochs = int(input("Number of epochs (10-100, default: 50): ").strip() or '50')
    batch_size = int(input("Batch size (2/4/8, default: 4): ").strip() or '4')
    learning_rate = float(input("Learning rate (0.0001-0.01, default: 0.001): ").strip() or '0.001')
    
    print(f"\n📋 Training Configuration:")
    print(f"   🏗️  Model: {model_type}")
    print(f"   🔧 Hidden dim: {hidden_dim}")
    print(f"   📚 Epochs: {num_epochs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   📈 Learning rate: {learning_rate}")
    
    # Estimate training time
    estimated_time = (674 // batch_size) * num_epochs * 0.1  # rough estimate
    print(f"   ⏱️  Estimated time: {estimated_time/60:.1f} minutes")
    
    confirm = input("\nProceed with training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    dataset_path = "training_data/graphs"
    
    try:
        history, optimized_model_path = main_training_pipeline(
            dataset_path=dataset_path,
            model_type=model_type,
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        print(f"\n🎉 Custom training complete!")
        print(f"   📊 Model saved: {optimized_model_path}")
        
        return history, optimized_model_path
        
    except Exception as e:
        print(f"❌ Custom training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_model_creation():
    """Test model creation before training."""
    print("🧪 TESTING MODEL CREATION")
    print("=" * 30)
    
    try:
        # Test standard model with correct feature count
        print("🏗️ Creating standard model...")
        model = DocumentGNN(
            num_node_features=22,  # Fixed to 22 features
            hidden_dim=64,
            num_classes=6,
            num_layers=2
        )
        
        print_model_info(model)
        
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        x = torch.randn(10, 22)  # 10 nodes, 22 features (fixed)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # Simple edges
        
        with torch.no_grad():
            output = model(x, edge_index)
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Output classes: {output.argmax(dim=1)}")
        
        print("✅ Model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_data():
    """Analyze dataset before training."""
    print("🔍 DATASET ANALYSIS")
    print("=" * 30)
    
    dataset_path = "training_data/graphs"
    analyze_dataset(dataset_path)


def main():
    """Main training interface."""
    print("🧠 DOCUMENT LAYOUT ANALYSIS - MODEL TRAINING")
    print("=" * 60)
    print("📊 Dataset Ready: 964 graphs (674 train, 144 val, 146 test)")
    print("🎯 Task: Classify text blocks into BODY, HH1, HH2, HH3, H4, TITLE")
    print()
    
    while True:
        print("Training Options:")
        print("1. 🚀 Quick train (recommended settings)")
        print("2. 🔧 Custom train (choose your settings)")
        print("3. 🔍 Analyze dataset")
        print("4. 🧪 Test model creation")
        print("5. 📊 View training history (if available)")
        print("6. 🚪 Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            history, model_path = quick_train()
            if history:
                print(f"\n📈 Training Summary:")
                print(f"   🏆 Best validation accuracy: {max(history.get('val_acc', [0])):.3f}")
                print(f"   📉 Final training loss: {history.get('train_loss', [0])[-1]:.4f}")
                print(f"   💾 Model saved: {model_path}")
            
        elif choice == "2":
            print("\n" + "="*60)
            history, model_path = custom_train()
            if history:
                print(f"\n📈 Training Summary:")
                print(f"   🏆 Best validation accuracy: {max(history.get('val_acc', [0])):.3f}")
                print(f"   📉 Final training loss: {history.get('train_loss', [0])[-1]:.4f}")
                print(f"   💾 Model saved: {model_path}")
            
        elif choice == "3":
            print("\n" + "="*40)
            analyze_data()
            
        elif choice == "4":
            print("\n" + "="*40)
            test_model_creation()
            
        elif choice == "5":
            print("\n📊 Training history analysis not yet implemented.")
            print("   (Will be available after training)")
            
        elif choice == "6":
            print("👋 Training session ended. Good luck with your model!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

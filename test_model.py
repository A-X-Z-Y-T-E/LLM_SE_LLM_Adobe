"""
Complete model testing and evaluation pipeline using the train/val/test split.
Test trained models on the proper test dataset.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

# Add model_training to path
sys.path.append(str(Path(__file__).parent / "model_training"))

from model_training.models import DocumentGNN, CompactDocumentGNN
from model_training.dataset import create_data_loaders, DocumentGraphDataset
from model_training.build_graph import load_processed_document_to_graph


class ModelTester:
    """Comprehensive model testing using proper train/val/test splits."""
    
    def __init__(self, model_path: str, dataset_path: str = "training_data/graphs", device: str = None):
        """Initialize model tester."""
        self.model_path = Path(model_path)
        self.dataset_path = dataset_path
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.label_names = ['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE']
        
        # Load the same data splits used during training
        self.train_loader, self.val_loader, self.test_loader = self._load_data_splits()
        self._load_model()
    
    def _load_data_splits(self):
        """Load the same train/val/test splits used during training."""
        print("📊 Loading dataset with same splits used during training...")
        
        # Use the same random seed and split ratios as training
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_path=self.dataset_path,
            batch_size=1,  # Use batch size 1 for detailed analysis
            train_ratio=0.7,
            val_ratio=0.15,
            num_workers=0,
            random_seed=42  # Same seed as training
        )
        
        print(f"   🎓 Train set: {len(train_loader.dataset)} graphs")
        print(f"   🔍 Validation set: {len(val_loader.dataset)} graphs")
        print(f"   🧪 Test set: {len(test_loader.dataset)} graphs")
        
        return train_loader, val_loader, test_loader
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"📦 Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Check if this is a quantized model
        is_quantized = checkpoint.get('quantized', False)
        
        if is_quantized:
            print("🔧 Loading quantized model...")
            # For quantized models, we need to load differently
            try:
                # Create a regular model first
                self.model = DocumentGNN(
                    num_node_features=22,
                    hidden_dim=64,
                    num_classes=6,
                    num_layers=2
                )
                
                # Prepare for quantization
                self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                self.model = torch.quantization.prepare(self.model, inplace=False)
                
                # Convert to quantized model
                self.model = torch.quantization.convert(self.model, inplace=False)
                
                # Load the quantized state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                print("✅ Quantized model loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Failed to load quantized model: {e}")
                print("🔄 Trying to load as regular model...")
                
                # Fallback to regular model
                self.model = DocumentGNN(
                    num_node_features=22,
                    hidden_dim=64,
                    num_classes=6,
                    num_layers=2
                )
                
                # Try to load without quantized parameters
                filtered_state_dict = {}
                for key, value in checkpoint['model_state_dict'].items():
                    if not any(qname in key for qname in ['scale', 'zero_point', '_packed_params']):
                        filtered_state_dict[key] = value
                
                if filtered_state_dict:
                    self.model.load_state_dict(filtered_state_dict, strict=False)
                    print("✅ Loaded partial model state (quantized -> regular)")
                else:
                    print("❌ Could not extract usable parameters from quantized model")
                    raise ValueError("Unable to load quantized model")
        
        else:
            print("🔧 Loading regular model...")
            # Regular model loading
            self.model = DocumentGNN(
                num_node_features=22,
                hidden_dim=64,
                num_classes=6,
                num_layers=2
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print("✅ Regular model loaded successfully")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def test_on_test_set(self) -> Dict[str, Any]:
        """Test model on the official test set (unseen during training)."""
        print("🧪 TESTING ON OFFICIAL TEST SET")
        print("=" * 40)
        print(f"📊 Test set size: {len(self.test_loader.dataset)} graphs")
        
        return self._evaluate_on_loader(self.test_loader, "Test Set")
    
    def test_on_validation_set(self) -> Dict[str, Any]:
        """Test model on validation set (used during training for early stopping)."""
        print("🔍 TESTING ON VALIDATION SET")
        print("=" * 40)
        print(f"📊 Validation set size: {len(self.val_loader.dataset)} graphs")
        
        return self._evaluate_on_loader(self.val_loader, "Validation Set")
    
    def test_on_train_set(self) -> Dict[str, Any]:
        """Test model on training set (to check for overfitting)."""
        print("🎓 TESTING ON TRAINING SET")
        print("=" * 40)
        print(f"📊 Training set size: {len(self.train_loader.dataset)} graphs")
        
        return self._evaluate_on_loader(self.train_loader, "Training Set")
    
    def _evaluate_on_loader(self, data_loader, set_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific data loader."""
        all_predictions = []
        all_labels = []
        all_confidence_scores = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                predictions = logits.argmax(dim=1)
                
                # Calculate confidence (softmax probabilities)
                probabilities = F.softmax(logits, dim=1)
                confidence_scores = probabilities.max(dim=1)[0]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_confidence_scores.extend(confidence_scores.cpu().numpy())
                
                # Count correct predictions
                correct_predictions += (predictions == batch.y).sum().item()
                total_predictions += batch.y.size(0)
                
                if batch_idx % 50 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(data_loader)} graphs")
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=self.label_names,
            output_dict=True,
            zero_division=0
        )
        
        results = {
            'set_name': set_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'confidence_scores': all_confidence_scores,
            'class_report': class_report,
            'total_samples': total_predictions
        }
        
        print(f"\n📊 {set_name} Results:")
        print(f"   🎯 Accuracy: {accuracy:.3f}")
        print(f"   📈 Precision: {precision:.3f}")
        print(f"   📉 Recall: {recall:.3f}")
        print(f"   🏆 F1 Score: {f1:.3f}")
        print(f"   📝 Total samples: {total_predictions}")
        
        return results
    
    def compare_all_sets(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across train/val/test sets."""
        print("🔬 COMPREHENSIVE EVALUATION ACROSS ALL SETS")
        print("=" * 60)
        
        # Test on all sets
        test_results = self.test_on_test_set()
        val_results = self.test_on_validation_set()
        train_results = self.test_on_train_set()
        
        # Compare results
        comparison = {
            'train': train_results,
            'validation': val_results,
            'test': test_results
        }
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"{'Set':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 55)
        
        for set_name, results in comparison.items():
            print(f"{set_name:<12} {results['accuracy']:<10.3f} "
                  f"{results['precision']:<10.3f} {results['recall']:<10.3f} "
                  f"{results['f1_score']:<10.3f}")
        
        # Check for overfitting
        train_acc = train_results['accuracy']
        val_acc = val_results['accuracy']
        test_acc = test_results['accuracy']
        
        print(f"\n🔍 OVERFITTING ANALYSIS:")
        if train_acc - val_acc > 0.1:
            print(f"   ⚠️  Possible overfitting detected!")
            print(f"   📊 Train accuracy ({train_acc:.3f}) >> Val accuracy ({val_acc:.3f})")
        else:
            print(f"   ✅ Good generalization!")
            print(f"   📊 Train-Val gap: {train_acc - val_acc:.3f}")
        
        print(f"   📊 Final test accuracy: {test_acc:.3f}")
        
        return comparison
    
    def test_single_document(self, document_path: str) -> Dict[str, Any]:
        """Test model on a single document (outside the dataset)."""
        print(f"📄 TESTING SINGLE DOCUMENT: {Path(document_path).name}")
        print("=" * 50)
        
        # Load document graph
        try:
            graph_data, original_blocks = load_processed_document_to_graph(document_path)
            
            if graph_data is None:
                raise ValueError("Failed to create graph from document")
            
            # Move to device
            graph_data = graph_data.to(self.device)
            
            print(f"📊 Document stats:")
            print(f"   Nodes: {graph_data.num_nodes}")
            print(f"   Edges: {graph_data.num_edges}")
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(graph_data.x, graph_data.edge_index)
                predictions = logits.argmax(dim=1)
                probabilities = F.softmax(logits, dim=1)
                confidence_scores = probabilities.max(dim=1)[0]
            
            # Prepare results
            results = {
                'predictions': predictions.cpu().numpy(),
                'true_labels': graph_data.y.cpu().numpy(),
                'confidence_scores': confidence_scores.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'original_blocks': original_blocks,
                'predicted_labels': [self.label_names[pred] for pred in predictions.cpu().numpy()],
                'true_label_names': [self.label_names[label] for label in graph_data.y.cpu().numpy()]
            }
            
            # Calculate accuracy for this document
            accuracy = (predictions == graph_data.y).float().mean().item()
            results['accuracy'] = accuracy
            
            print(f"\n🎯 Document accuracy: {accuracy:.3f}")
            
            # Show predictions vs truth
            print(f"\n🔍 Predictions vs Ground Truth:")
            for i in range(min(10, len(predictions))):  # Show first 10
                pred_name = self.label_names[predictions[i]]
                true_name = self.label_names[graph_data.y[i]]
                confidence = confidence_scores[i].item()
                text_preview = original_blocks[i].get('text', '')[:50] + '...'
                
                status = "✅" if predictions[i] == graph_data.y[i] else "❌"
                print(f"   {i:2d}: {pred_name:6s} vs {true_name:6s} ({confidence:.2f}) {status} | {text_preview}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error testing document: {e}")
            raise
    
    def visualize_confusion_matrix(self, results: Dict[str, Any], save_path: str = None):
        """Create and display confusion matrix."""
        set_name = results.get('set_name', 'Dataset')
        print(f"\n📊 CONFUSION MATRIX - {set_name}")
        print("=" * 40)
        
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title(f'Confusion Matrix - {set_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def save_detailed_results(self, results: Dict[str, Any], output_path: str):
        """Save detailed results to file."""
        # Prepare JSON-serializable results
        json_results = {
            'set_name': results['set_name'],
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'class_report': results['class_report'],
            'total_samples': results['total_samples'],
            'model_path': str(self.model_path),
            'dataset_path': self.dataset_path
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {output_path}")


def find_trained_models() -> List[Path]:
    """Find available trained models."""
    model_dirs = [
        Path("model_checkpoints"),
        Path("optimized_models"),
        Path("training_data/models"),
        Path("."),  # Current directory
    ]
    
    models = []
    for model_dir in model_dirs:
        if model_dir.exists():
            models.extend(list(model_dir.glob("*.pth")))
            models.extend(list(model_dir.glob("*.pt")))
    
    return models


def main():
    """Main testing interface."""
    print("🧪 DOCUMENT LAYOUT ANALYSIS - MODEL TESTING WITH PROPER SPLITS")
    print("=" * 70)
    
    # Find available models
    available_models = find_trained_models()
    
    if not available_models:
        print("❌ No trained models found!")
        print("   Train a model first using: python train_model.py")
        return
    
    print("📦 Available trained models:")
    for i, model_path in enumerate(available_models):
        print(f"   {i}: {model_path}")
    
    # Select model
    try:
        model_idx = int(input(f"\nSelect model (0-{len(available_models)-1}): "))
        selected_model = available_models[model_idx]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return
    
    # Initialize tester
    tester = ModelTester(str(selected_model))
    
    while True:
        print("\n🧪 Testing Options:")
        print("1. 🧪 Test on official test set (recommended)")
        print("2. 🔍 Test on validation set")
        print("3. 🎓 Test on training set (check overfitting)")
        print("4. 🔬 Compare all sets (comprehensive analysis)")
        print("5. 📄 Test on single document (external)")
        print("6. 📊 Visualize performance")
        print("7. 💾 Save results")
        print("8. 🚪 Exit")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            results = tester.test_on_test_set()
            
        elif choice == "2":
            print("\n" + "="*60)
            results = tester.test_on_validation_set()
            
        elif choice == "3":
            print("\n" + "="*60)
            results = tester.test_on_train_set()
            
        elif choice == "4":
            print("\n" + "="*60)
            comparison = tester.compare_all_sets()
            
        elif choice == "5":
            # Test single document
            doc_path = input("Enter path to processed document JSON: ").strip()
            
            try:
                print("\n" + "="*60)
                doc_results = tester.test_single_document(doc_path)
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "6":
            # Visualize performance (need test results first)
            try:
                print("\n📊 Running test set evaluation for visualization...")
                test_results = tester.test_on_test_set()
                
                print("\n🎨 Creating visualizations...")
                tester.visualize_confusion_matrix(test_results, "test_confusion_matrix.png")
                
            except Exception as e:
                print(f"❌ Visualization error: {e}")
        
        elif choice == "7":
            # Save results
            try:
                print("\n💾 Running comprehensive evaluation to save results...")
                comparison = tester.compare_all_sets()
                
                # Save each set's results
                for set_name, results in comparison.items():
                    output_file = f"{set_name}_results.json"
                    tester.save_detailed_results(results, output_file)
                
                print("✅ All results saved!")
                
            except Exception as e:
                print(f"❌ Save error: {e}")
        
        elif choice == "8":
            print("👋 Testing session ended!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
    # The following code is removed because 'tester' is not defined in this scope.


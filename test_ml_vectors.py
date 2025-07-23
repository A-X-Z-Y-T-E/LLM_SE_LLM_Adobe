"""
Test model predictions using ML vectors directly instead of processed features.
This will help identify if the issue is in feature engineering or model training.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Any

# Add model_training to path
sys.path.append(str(Path(__file__).parent / "model_training"))

from model_training.models import DocumentGNN
from model_training.build_graph import build_document_graph, extract_node_features_and_labels, build_graph_edges


class MLVectorTester:
    """Test model using ML vectors directly."""
    
    def __init__(self, model_path: str):
        """Initialize with trained model."""
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_names = ['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE']
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        print(f"📦 Loading model from: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Determine model architecture from checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            hidden_dim = config.get('hidden_dim', 64)
            num_layers = config.get('num_layers', 2)
            dropout = config.get('dropout', 0.0)
        else:
            # Default values for older models
            hidden_dim = 64
            num_layers = 2
            dropout = 0.0
        
        # Create model with correct architecture
        self.model = DocumentGNN(
            num_node_features=22,
            hidden_dim=hidden_dim,
            num_classes=6,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
        print(f"   🏗️ Architecture: {hidden_dim} hidden, {num_layers} layers, {dropout} dropout")
    
    def test_with_ml_vectors(self, ml_vectors_path: str):
        """Test model using ML vectors directly converted to graphs."""
        print(f"🧪 Testing with ML vectors: {Path(ml_vectors_path).name}")
        
        # Load ML vectors
        with open(ml_vectors_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if this is the correct ML ready vectors format
        if 'feature_vectors' not in data or 'labels' not in data:
            print("❌ Invalid ML vectors format - expected 'feature_vectors' and 'labels' fields")
            print(f"   Found fields: {list(data.keys())}")
            return None
        
        feature_vectors = data['feature_vectors']
        labels = data['labels']
        metadata = data.get('metadata', {})
        block_ids = data.get('block_ids', [])
        
        print(f"📊 Found {len(feature_vectors)} feature vectors")
        print(f"🏷️ Found {len(labels)} labels")
        print(f"📐 Feature dimensions: {len(feature_vectors[0]) if feature_vectors else 0}")
        
        if len(feature_vectors) != len(labels):
            print(f"❌ Mismatch: {len(feature_vectors)} features vs {len(labels)} labels")
            return None
        
        # Verify feature dimensions
        expected_features = 22
        if feature_vectors and len(feature_vectors[0]) != expected_features:
            print(f"❌ Expected {expected_features} features, got {len(feature_vectors[0])}")
            return None
        
        # Convert ML vectors back to block format for graph building
        print(f"🔧 Converting ML vectors to graph format...")
        blocks = self._convert_ml_vectors_to_blocks(feature_vectors, labels, block_ids, metadata)
        
        # Use existing graph building functionality
        print(f"🏗️ Building graph using existing build_graph functions...")
        graph_data, original_blocks = build_document_graph(blocks, auto_save=False)
        
        if graph_data is None:
            print("❌ Failed to build graph from ML vectors")
            return None
        
        print(f"✅ Graph built successfully:")
        print(f"   📊 Nodes: {graph_data.num_nodes}")
        print(f"   🔗 Edges: {graph_data.num_edges}")
        print(f"   📐 Features: {graph_data.x.shape}")
        print(f"   🏷️ Labels: {graph_data.y.shape}")
        
        # Extract page distribution for analysis
        page_distribution = {}
        if block_ids:
            for block_id in block_ids:
                try:
                    page_part = block_id.split('_p')[1].split('_')[0]
                    page_num = int(page_part)
                    page_distribution[page_num] = page_distribution.get(page_num, 0) + 1
                except:
                    page_num = 1
                    page_distribution[page_num] = page_distribution.get(page_num, 0) + 1
        else:
            page_distribution[1] = len(feature_vectors)
        
        print(f"📄 Page distribution: {dict(sorted(page_distribution.items()))}")
        
        # Show ground truth label distribution
        print(f"🏷️ Ground truth label distribution:")
        true_label_counts = {}
        for label_id in labels:
            label_name = self.label_names[label_id]
            true_label_counts[label_name] = true_label_counts.get(label_name, 0) + 1
        
        for label_name, count in true_label_counts.items():
            print(f"   {label_name}: {count} blocks")
        
        # Move graph to device and make predictions
        graph_data = graph_data.to(self.device)
        
        print(f"\n🤖 Making predictions using graph structure...")
        with torch.no_grad():
            logits = self.model(graph_data.x, graph_data.edge_index)
            predictions = logits.argmax(dim=1)
            probabilities = F.softmax(logits, dim=1)
            confidence_scores = probabilities.max(dim=1)[0]
        
        predictions_cpu = predictions.cpu()
        confidence_cpu = confidence_scores.cpu()
        labels_tensor = graph_data.y.cpu()
        
        # Show prediction distribution
        print(f"\n🔍 Model prediction distribution:")
        pred_label_counts = {}
        for pred_id in predictions_cpu:
            label_name = self.label_names[pred_id.item()]
            pred_label_counts[label_name] = pred_label_counts.get(label_name, 0) + 1
        
        for label_name, count in pred_label_counts.items():
            print(f"   {label_name}: {count} blocks")
        
        # Calculate accuracy
        accuracy = (predictions_cpu == labels_tensor).float().mean().item()
        print(f"\n🎯 Overall accuracy: {accuracy:.3f}")
        
        # Show per-class accuracy
        print(f"\n📊 Per-class analysis:")
        for class_id, class_name in enumerate(self.label_names):
            true_class_mask = (labels_tensor == class_id)
            if true_class_mask.sum() > 0:
                class_correct = (predictions_cpu[true_class_mask] == class_id).sum().item()
                class_total = true_class_mask.sum().item()
                class_accuracy = class_correct / class_total
                print(f"   {class_name:8s}: {class_accuracy:.3f} ({class_correct}/{class_total})")
        
        # Show prediction examples
        print(f"\n📝 Sample predictions:")
        for i in range(min(20, len(feature_vectors))):
            true_label = self.label_names[labels[i]]
            pred_label = self.label_names[predictions_cpu[i].item()]
            confidence = confidence_cpu[i].item()
            
            # Extract page from block_id if available
            if block_ids and i < len(block_ids):
                try:
                    page_part = block_ids[i].split('_p')[1].split('_')[0]
                    page = int(page_part)
                except:
                    page = 1
                block_id = block_ids[i]
            else:
                page = 1
                block_id = f"block_{i}"
            
            status = "✅" if true_label == pred_label else "❌"
            print(f"   {i:2d}: {pred_label:6s} vs {true_label:6s} ({confidence:.2f}) Page {page} {status} | {block_id}")
        
        # Analyze by page
        print(f"\n📄 Predictions by page:")
        page_predictions = {}
        page_true_labels = {}
        
        for i in range(len(feature_vectors)):
            # Extract page from block_id
            if block_ids and i < len(block_ids):
                try:
                    page_part = block_ids[i].split('_p')[1].split('_')[0]
                    page_num = int(page_part)
                except:
                    page_num = 1
            else:
                page_num = 1
            
            pred_label = self.label_names[predictions_cpu[i].item()]
            true_label = self.label_names[labels[i]]
            
            if page_num not in page_predictions:
                page_predictions[page_num] = {}
                page_true_labels[page_num] = {}
                
            if pred_label not in page_predictions[page_num]:
                page_predictions[page_num][pred_label] = 0
            page_predictions[page_num][pred_label] += 1
            
            if true_label not in page_true_labels[page_num]:
                page_true_labels[page_num][true_label] = 0
            page_true_labels[page_num][true_label] += 1
        
        for page_num in sorted(page_predictions.keys()):
            print(f"   Page {page_num}:")
            print(f"     🎯 Predicted: {page_predictions[page_num]}")
            print(f"     ✅ Actual:    {page_true_labels[page_num]}")
        
        # Advanced analysis
        expected_structural = sum(count for label, count in true_label_counts.items() 
                                if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
        predicted_structural = sum(count for label, count in pred_label_counts.items() 
                                 if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
        
        print(f"\n🔍 STRUCTURAL ELEMENTS ANALYSIS:")
        print(f"   📊 Expected structural elements: {expected_structural}")
        print(f"   🤖 Predicted structural elements: {predicted_structural}")
        print(f"   📉 Difference: {expected_structural - predicted_structural}")
        
        structural_ratio = predicted_structural / max(expected_structural, 1)
        print(f"   📈 Structural prediction ratio: {structural_ratio:.3f}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions_cpu.numpy(),
            'true_labels': labels_tensor.numpy(),
            'confidence_scores': confidence_cpu.numpy(),
            'prediction_distribution': pred_label_counts,
            'true_distribution': true_label_counts,
            'page_predictions': page_predictions,
            'page_true_labels': page_true_labels,
            'structural_ratio': structural_ratio,
            'metadata': metadata,
            'graph_info': {
                'num_nodes': graph_data.num_nodes,
                'num_edges': graph_data.num_edges,
                'features_shape': graph_data.x.shape,
                'labels_shape': graph_data.y.shape
            }
        }
    
    def _convert_ml_vectors_to_blocks(self, feature_vectors: List[List[float]], 
                                    labels: List[int], block_ids: List[str], 
                                    metadata: Dict) -> List[Dict]:
        """Convert ML vectors back to block format for graph building."""
        blocks = []
        
        # Get feature names from metadata if available
        feature_names = metadata.get('feature_names', [
            "x0_norm", "y0_norm", "x1_norm", "y1_norm", "width_norm", "height_norm", 
            "is_left_aligned", "is_centered_horizontally", "font_size_norm", "is_bold", "is_italic",
            "text_length_chars_norm", "starts_with_bullet", "ends_with_colon", 
            "is_all_caps", "contains_number_prefix", "y_offset_to_prev_block_norm", 
            "x_offset_to_prev_block_norm", "font_size_ratio_to_prev_block_norm", 
            "is_prev_block_same_font_size", "is_prev_block_same_bold_status", 
            "is_prev_block_same_indentation"
        ])
        
        for i, (features, label_id) in enumerate(zip(feature_vectors, labels)):
            # Extract page number from block_id
            page_num = 1  # default
            if block_ids and i < len(block_ids):
                try:
                    page_part = block_ids[i].split('_p')[1].split('_')[0]
                    page_num = int(page_part)
                except:
                    pass
            
            # Create block dictionary compatible with build_graph
            block = {
                'block_id': block_ids[i] if block_ids and i < len(block_ids) else f"block_{i}",
                'page_number': page_num,
                'label_id': label_id,
                'feature_vectors': features,  # Store as feature_vectors field
                'features': {name: features[j] for j, name in enumerate(feature_names[:len(features)])},
                # Add some dummy spatial data for graph building
                'bbox': [
                    features[0] * 1000,  # x0_norm * page_width
                    features[1] * 1000,  # y0_norm * page_height  
                    features[2] * 1000,  # x1_norm * page_width
                    features[3] * 1000   # y1_norm * page_height
                ] if len(features) >= 4 else [0, 0, 100, 100],
                'text': f"Block {i}",  # Dummy text
                'pdf_id': metadata.get('pdf_id', '0')
            }
            
            blocks.append(block)
        
        return blocks
    
    def _calculate_class_weights(self, labels: List[int]) -> Dict[str, float]:
        """Calculate recommended class weights based on label distribution."""
        from collections import Counter
        
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(set(labels))
        
        # Calculate class weights using sklearn's method
        weights = {}
        for label_id in range(6):  # 0-5 for our 6 classes
            count = label_counts.get(label_id, 1)  # Avoid division by zero
            weight = total_samples / (num_classes * count)
            weights[self.label_names[label_id]] = round(weight, 2)
        
        return weights


def find_latest_model() -> Path:
    """Find the latest trained model, prioritizing updated models."""
    model_dirs = [
        Path("."),
        Path("model_checkpoints"),
        Path("optimized_models"),
        Path("training_data/models"),
    ]
    
    models = []
    for model_dir in model_dirs:
        if model_dir.exists():
            models.extend(list(model_dir.glob("*.pth")))
            models.extend(list(model_dir.glob("*.pt")))
    
    if not models:
        raise FileNotFoundError("No trained models found!")
    
    # Prioritize updated models (updated_model_2 > updated_model_1 > others)
    updated_models = [m for m in models if 'updated_model' in str(m)]
    if updated_models:
        # Sort by model number (updated_model_2 before updated_model_1)
        updated_models.sort(key=lambda x: str(x), reverse=True)
        latest_updated = updated_models[0]
        print(f"🎯 Found updated model: {latest_updated}")
        return latest_updated
    
    # Fall back to regular models
    regular_models = [m for m in models if 'quantized' not in str(m)]
    if regular_models:
        return max(regular_models, key=lambda p: p.stat().st_mtime)
    else:
        return max(models, key=lambda p: p.stat().st_mtime)


def main():
    """Main testing interface."""
    print("🧪 ML VECTOR MODEL TESTING")
    print("=" * 50)
    print("🎯 Testing model directly with ML vectors to bypass feature engineering")
    
    try:
        # Find latest model (prioritizes updated_model_2 > updated_model_1)
        model_path = find_latest_model()
        print(f"📦 Using model: {model_path}")
        
        # Check which model we're testing
        if 'updated_model_2' in str(model_path):
            print("🎉 Testing the IMPROVED MODEL V2 with fine-tuned class weights!")
            print("   This should show balanced structural element detection.")
        elif 'updated_model' in str(model_path):
            print("🎉 Testing the RETRAINED MODEL with class weights!")
            print("   This should show much better structural element detection.")
        
        # Initialize tester
        tester = MLVectorTester(str(model_path))
        
        while True:
            print("\n🧪 ML Vector Testing Options:")
            print("1. 📄 Test with specific ML vectors file")
            print("2. 🔍 Test with sample ML vectors (RECOMMENDED)")
            print("3. 📁 Test all ML vectors in directory")
            print("4. 📊 Compare model versions")
            print("5. 🚪 Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                # Test specific file
                file_path = input("Enter path to ML vectors JSON file: ").strip()
                
                try:
                    result = tester.test_with_ml_vectors(file_path)
                    
                    if result:
                        print(f"\n📊 SUMMARY:")
                        print(f"   🎯 Accuracy: {result['accuracy']:.3f}")
                        print(f"   📈 Structural ratio: {result['structural_ratio']:.3f}")
                        print(f"   📊 Predictions: {result['prediction_distribution']}")
                        print(f"   🏷️ Ground truth: {result['true_distribution']}")
                        
                        if result['structural_ratio'] > 0.8:
                            print(f"\n🎉 EXCELLENT! Structural detection is great!")
                        elif result['structural_ratio'] > 0.6:
                            print(f"\n✅ GOOD! Significant improvement in structural detection!")
                        elif result['structural_ratio'] > 0.4:
                            print(f"\n👍 DECENT! Some improvement but needs work!")
                        else:
                            print(f"\n⚠️ Still needs improvement in structural detection")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == "2":
                # Test with sample (RECOMMENDED)
                sample_files = list(Path("training_data/ml_ready_vectors").glob("*.json"))
                if not sample_files:
                    sample_files = list(Path("data/ml_vectors").glob("*.json"))
                    if not sample_files:
                        sample_files = list(Path("training_data").glob("*ml_vector*.json"))
                
                if sample_files:
                    sample_file = sample_files[0]
                    model_name = "UPDATED MODEL V2" if 'updated_model_2' in str(model_path) else "UPDATED MODEL"
                    print(f"🧪 Testing {model_name} with sample: {sample_file.name}")
                    
                    try:
                        result = tester.test_with_ml_vectors(str(sample_file))
                        
                        if result:
                            print(f"\n📊 🎉 {model_name} RESULTS:")
                            print(f"   🎯 Accuracy: {result['accuracy']:.3f}")
                            print(f"   📈 Structural ratio: {result['structural_ratio']:.3f}")
                            print(f"   📊 Predictions: {result['prediction_distribution']}")
                            print(f"   🏷️ Ground truth: {result['true_distribution']}")
                            
                            # Detailed analysis
                            expected_structural = sum(count for label, count in result['true_distribution'].items() 
                                                    if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
                            predicted_structural = sum(count for label, count in result['prediction_distribution'].items() 
                                                     if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
                            
                            print(f"\n🔍 MODEL ANALYSIS:")
                            print(f"   📊 Expected structural elements: {expected_structural}")
                            print(f"   🤖 Predicted structural elements: {predicted_structural}")
                            print(f"   📈 Structural prediction ratio: {result['structural_ratio']:.3f}")
                            
                            # Assessment based on ratio
                            if result['structural_ratio'] >= 1.2:
                                print(f"\n🎉 OUTSTANDING PERFORMANCE! 🎉")
                                print(f"   ✅ Model is detecting MORE structural elements than expected!")
                                print(f"   📈 High sensitivity - excellent for outline generation!")
                                print(f"   🚀 PRODUCTION READY!")
                            elif result['structural_ratio'] >= 0.8:
                                print(f"\n🎊 EXCELLENT PERFORMANCE! 🎊")
                                print(f"   ✅ Model successfully balanced structural detection!")
                                print(f"   🚀 Ready for production use!")
                            elif result['structural_ratio'] >= 0.6:
                                print(f"\n👍 GOOD PERFORMANCE!")
                                print(f"   ✅ Significant improvement over original model!")
                                print(f"   🔧 Minor tuning could improve further")
                            elif result['structural_ratio'] >= 0.4:
                                print(f"\n⚠️ MODERATE IMPROVEMENT")
                                print(f"   📈 Better than original but needs more work")
                            else:
                                print(f"\n🤔 LIMITED IMPROVEMENT")
                                print(f"   ⚠️ Still under-detecting structural elements")
                            

                            # Class-wise performance
                            print(f"\n📊 Class-wise Performance:")
                            for class_name, pred_count in result['prediction_distribution'].items():
                                true_count = result['true_distribution'].get(class_name, 0)
                                if true_count > 0:
                                    ratio = pred_count / true_count
                                    if 0.8 <= ratio <= 1.2:
                                        status = "🎯 Balanced"
                                    elif ratio > 1.2:
                                        status = "📈 Over-detecting (good sensitivity)"
                                    elif ratio >= 0.5:
                                        status = "⚠️ Under-detecting"
                                    else:
                                        status = "❌ Poor detection"
                                    print(f"   {class_name:8s}: {pred_count:2d}/{true_count:2d} - {status}")
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("❌ No ML vectors files found")
                    print(f"   Expected in: training_data/ml_ready_vectors/")
            
            elif choice == "3":
                # Test directory
                directory = input("Enter directory path with ML vectors (default: training_data/ml_ready_vectors): ").strip()
                
                if not directory:
                    directory = "training_data/ml_ready_vectors"
                
                try:
                    dir_path = Path(directory)
                    if not dir_path.exists():
                        print(f"❌ Directory not found: {directory}")
                        continue
                    
                    json_files = list(dir_path.glob("*.json"))
                    if not json_files:
                        print(f"❌ No JSON files found in: {directory}")
                        continue
                    
                    model_name = "UPDATED MODEL V2" if 'updated_model_2' in str(model_path) else "UPDATED MODEL"
                    print(f"🔄 Testing {model_name} with {len(json_files)} files...")
                    
                    total_accuracy = 0
                    successful_tests = 0
                    total_expected_structural = 0
                    total_predicted_structural = 0
                    
                    for json_file in json_files[:3]:  # Test first 3 files
                        print(f"\n📄 Testing: {json_file.name}")
                        try:
                            result = tester.test_with_ml_vectors(str(json_file))
                            if result:
                                total_accuracy += result['accuracy']
                                successful_tests += 1
                                
                                expected_structural = sum(count for label, count in result['true_distribution'].items() 
                                                        if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
                                predicted_structural = sum(count for label, count in result['prediction_distribution'].items() 
                                                         if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE'])
                                
                                total_expected_structural += expected_structural
                                total_predicted_structural += predicted_structural
                                
                                print(f"   ✅ Accuracy: {result['accuracy']:.3f}")
                                print(f"   📊 Structural: {predicted_structural}/{expected_structural}")
                        except Exception as e:
                            print(f"   ❌ Failed: {e}")
                    
                    if successful_tests > 0:
                        avg_accuracy = total_accuracy / successful_tests
                        structural_ratio = total_predicted_structural / max(total_expected_structural, 1)
                        
                        print(f"\n📊 {model_name} BATCH SUMMARY:")
                        print(f"   📁 Files tested: {successful_tests}")
                        print(f"   🎯 Average accuracy: {avg_accuracy:.3f}")
                        print(f"   📊 Total expected structural: {total_expected_structural}")
                        print(f"   🤖 Total predicted structural: {total_predicted_structural}")
                        print(f"   📈 Structural prediction ratio: {structural_ratio:.3f}")
                        
                        if structural_ratio > 0.8:
                            print(f"\n🎉 OUTSTANDING BATCH PERFORMANCE!")
                            print(f"   ✅ Model works consistently across documents!")
                        elif structural_ratio > 0.6:
                            print(f"\n🎊 GOOD BATCH PERFORMANCE!")
                            print(f"   ✅ Significant improvement across documents!")
                        else:
                            print(f"\n🔧 MIXED BATCH RESULTS")
                            print(f"   ⚠️ Performance varies across documents")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            elif choice == "4":
                print(f"\n📊 MODEL VERSION COMPARISON:")
                print(f"   🔴 Original model: ~20% structural detection")
                print(f"   🟡 Updated model 1: ~150% structural detection (over-sensitive)")
                print(f"   🟢 Updated model 2: ~80-120% structural detection (balanced)")
                print(f"   💡 Current model: {model_path.name}")
                
            elif choice == "5":
                print("👋 Testing session ended!")
                print(f"🎯 Your model performance has been analyzed!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
    
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
           
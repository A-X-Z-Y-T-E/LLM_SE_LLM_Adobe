"""
Utility functions for graph construction and processing.
Modular components for Document Layout Analysis GNN.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime

def validate_graph_data(graph_data, original_blocks: List[Dict]) -> bool:
    """
    Validate that the constructed graph data is consistent.
    
    Args:
        graph_data: PyTorch Geometric Data object
        original_blocks: Original text blocks
        
    Returns:
        bool: True if validation passes
    """
    try:
        # Check basic properties
        assert graph_data.num_nodes == len(original_blocks), "Node count mismatch"
        assert graph_data.x.shape[0] == len(original_blocks), "Feature tensor size mismatch"
        assert graph_data.y.shape[0] == len(original_blocks), "Label tensor size mismatch"
        assert graph_data.x.shape[1] == 22, "Expected 22 features per node"
        
        # Check edge index bounds
        if graph_data.num_edges > 0:
            max_node_idx = graph_data.edge_index.max().item()
            assert max_node_idx < graph_data.num_nodes, "Edge index out of bounds"
            
        # Check data types
        assert graph_data.x.dtype == torch.float32, "Features should be float32"
        assert graph_data.y.dtype == torch.long, "Labels should be long"
        assert graph_data.edge_index.dtype == torch.long, "Edge indices should be long"
        
        print("✅ Graph validation passed")
        return True
        
    except AssertionError as e:
        print(f"❌ Graph validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def get_graph_statistics(graph_data) -> Dict[str, float]:
    """
    Calculate graph statistics for analysis.
    
    Args:
        graph_data: PyTorch Geometric Data object
        
    Returns:
        Dict with graph statistics
    """
    stats = {
        'num_nodes': int(graph_data.num_nodes),
        'num_edges': int(graph_data.num_edges),
        'avg_degree': graph_data.num_edges / max(graph_data.num_nodes, 1),
        'edge_density': graph_data.num_edges / max(graph_data.num_nodes * (graph_data.num_nodes - 1), 1),
    }
    
    # Label distribution
    if hasattr(graph_data, 'y') and graph_data.y is not None:
        label_counts = torch.bincount(graph_data.y)
        stats['label_distribution'] = label_counts.tolist()
        stats['num_classes'] = len(label_counts)
    
    # Feature statistics
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        stats['feature_mean'] = float(graph_data.x.mean())
        stats['feature_std'] = float(graph_data.x.std())
        stats['feature_min'] = float(graph_data.x.min())
        stats['feature_max'] = float(graph_data.x.max())
    
    return stats

def load_label_mappings() -> Dict[str, Dict]:
    """
    Load label mappings from the project's label_mappings.json file.
    
    Returns:
        Dict containing label mappings
    """
    try:
        # Get project root and label mappings file
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # Go up from model_training to project root
        mappings_file = project_root / "data" / "label_mappings.json"
        
        if not mappings_file.exists():
            print(f"⚠️ Label mappings file not found: {mappings_file}")
            # Return default mappings
            return {
                "label_to_id": {"BODY": 0, "HH1": 1, "HH2": 2, "HH3": 3, "H4": 4, "TITLE": 5},
                "id_to_label": {0: "BODY", 1: "HH1", 2: "HH2", 3: "HH3", 4: "H4", 5: "TITLE"},
                "labels_list": ["BODY", "HH1", "HH2", "HH3", "H4", "TITLE"]
            }
        
        with open(mappings_file, 'r', encoding='utf-8') as f:
            import json
            data = json.load(f)
        
        mappings = data.get("mappings", {})
        
        # Convert string keys to int for id_to_label
        id_to_label_raw = mappings.get("id_to_label", {})
        id_to_label = {int(k): v for k, v in id_to_label_raw.items()}
        
        return {
            "label_to_id": mappings.get("label_to_id", {}),
            "id_to_label": id_to_label,
            "labels_list": mappings.get("labels_list", [])
        }
        
    except Exception as e:
        print(f"❌ Error loading label mappings: {e}")
        # Return default mappings
        return {
            "label_to_id": {"BODY": 0, "HH1": 1, "HH2": 2, "HH3": 3, "H4": 4, "TITLE": 5},
            "id_to_label": {0: "BODY", 1: "HH1", 2: "HH2", 3: "HH3", 4: "H4", 5: "TITLE"},
            "labels_list": ["BODY", "HH1", "HH2", "HH3", "H4", "TITLE"]
        }

def save_graph_to_disk(graph_data, original_blocks: List[Dict], output_path: str) -> None:
    """
    Save graph data and metadata to disk.
    
    Args:
        graph_data: PyTorch Geometric Data object
        original_blocks: Original text blocks
        output_path: Path to save the graph
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'graph_data': graph_data,
        'original_blocks': original_blocks,
        'statistics': get_graph_statistics(graph_data),
        'label_mappings': load_label_mappings()
    }
    
    # Save graph
    torch.save(save_data, output_path)
    
    print(f"💾 Saved graph to: {output_path}")
    print(f"   📊 Nodes: {graph_data.num_nodes}")
    print(f"   🔗 Edges: {graph_data.num_edges}")

def load_graph_from_disk(graph_path: str):
    """
    Load graph data from disk.
    
    Args:
        graph_path: Path to saved graph file
        
    Returns:
        Tuple of (graph_data, original_blocks, metadata)
    """
    graph_path = Path(graph_path)
    
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    # Load data with weights_only=False for our custom structures
    data = torch.load(graph_path, weights_only=False)
    
    graph_data = data['graph_data']
    original_blocks = data['original_blocks']
    metadata = {
        'statistics': data.get('statistics', {}),
        'label_mappings': data.get('label_mappings', {})
    }
    
    print(f"✅ Loaded graph from disk:")
    print(f"   📊 Nodes: {graph_data.num_nodes}")
    print(f"   🔗 Edges: {graph_data.num_edges}")
    
    return graph_data, original_blocks, metadata

def load_combined_graphs(combined_path: str):
    """
    Load combined graphs file.
    
    Args:
        combined_path: Path to combined graphs file
        
    Returns:
        Tuple of (graphs_list, metadata)
    """
    combined_path = Path(combined_path)
    
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined graphs file not found: {combined_path}")
    
    # Load data with weights_only=False for our custom structures
    data = torch.load(combined_path, weights_only=False)
    
    graphs = data.get('graphs', [])
    metadata = data.get('metadata', {})
    
    print(f"✅ Loaded combined graphs from disk:")
    print(f"   📊 Documents: {len(graphs)}")
    print(f"   📈 Total nodes: {metadata.get('total_nodes', 'unknown')}")
    print(f"   🔗 Total edges: {metadata.get('total_edges', 'unknown')}")
    
    return graphs, metadata

def create_training_data_splits(graphs: List[Tuple], train_ratio: float = 0.7, 
                               val_ratio: float = 0.15, random_seed: int = 42):
    """
    Split graphs into training, validation, and test sets.
    
    Args:
        graphs: List of (graph_data, original_blocks, file_path) tuples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    import random
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Shuffle graphs
    graphs_copy = graphs.copy()
    random.shuffle(graphs_copy)
    
    # Calculate split indices
    total_graphs = len(graphs_copy)
    train_end = int(total_graphs * train_ratio)
    val_end = int(total_graphs * (train_ratio + val_ratio))
    
    # Split data
    train_graphs = graphs_copy[:train_end]
    val_graphs = graphs_copy[train_end:val_end]
    test_graphs = graphs_copy[val_end:]
    
    print(f"📊 Data split created:")
    print(f"   🎓 Training: {len(train_graphs)} graphs ({len(train_graphs)/total_graphs*100:.1f}%)")
    print(f"   🔍 Validation: {len(val_graphs)} graphs ({len(val_graphs)/total_graphs*100:.1f}%)")
    print(f"   🧪 Testing: {len(test_graphs)} graphs ({len(test_graphs)/total_graphs*100:.1f}%)")
    
    return train_graphs, val_graphs, test_graphs

def save_data_splits(train_graphs: List, val_graphs: List, test_graphs: List, 
                    output_dir: str = None):
    """
    Save training data splits to disk.
    
    Args:
        train_graphs: Training graphs
        val_graphs: Validation graphs  
        test_graphs: Test graphs
        output_dir: Directory to save splits
    """
    if output_dir is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        output_dir = project_root / "training_data" / "splits"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits_data = {
        'train': train_graphs,
        'val': val_graphs,
        'test': test_graphs,
        'metadata': {
            'train_count': len(train_graphs),
            'val_count': len(val_graphs),
            'test_count': len(test_graphs),
            'total_count': len(train_graphs) + len(val_graphs) + len(test_graphs),
            'split_timestamp': datetime.now().isoformat()
        }
    }
    
    splits_path = output_dir / "data_splits.pt"
    torch.save(splits_data, splits_path)
    
    print(f"💾 Saved data splits to: {splits_path}")

def check_cuda_availability():
    """
    Check CUDA availability and GPU information.
    """
    print("🔍 CUDA & GPU Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️ CUDA not available - will use CPU")

"""
PyTorch Geometric Dataset classes for Document Layout Analysis.
Handles loading and batching of graph data for training.
"""

import os
import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import json


class DocumentGraphDataset(InMemoryDataset):
    """
    In-memory dataset for document graphs.
    Loads all graphs into memory for fast access during training.
    """
    
    def __init__(self, 
                 root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 load_combined: bool = False):
        """
        Initialize DocumentGraphDataset.
        
        Args:
            root (str): Root directory containing graph files
            transform (Callable, optional): Transform to apply to data
            pre_transform (Callable, optional): Pre-transform to apply
            load_combined (bool): Whether to load from combined_graphs.pt
        """
        self.load_combined = load_combined
        super().__init__(root, transform, pre_transform)
        # Use weights_only=False for our custom graph data (we trust our own files)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw graph files."""
        if self.load_combined:
            return ['combined_graphs.pt']
        
        raw_dir = Path(self.root) / 'raw'
        if not raw_dir.exists():
            return []
        
        return [f for f in os.listdir(raw_dir) if f.endswith('_graph.pt')]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed files."""
        return ['data.pt']
    
    def download(self):
        """Download data - not applicable for local files."""
        pass
    
    def process(self):
        """Process raw graph files into PyTorch Geometric format."""
        data_list = []
        
        if self.load_combined:
            # Load from combined file
            combined_path = Path(self.root) / 'raw' / 'combined_graphs.pt'
            if combined_path.exists():
                # Use weights_only=False for our custom data structures
                combined_data = torch.load(combined_path, weights_only=False)
                graphs = combined_data.get('graphs', [])
                
                for graph_data, _, _ in graphs:
                    if self.pre_filter is not None and not self.pre_filter(graph_data):
                        continue
                    if self.pre_transform is not None:
                        graph_data = self.pre_transform(graph_data)
                    data_list.append(graph_data)
        else:
            # Load individual graph files
            raw_dir = Path(self.root) / 'raw'
            for filename in self.raw_file_names:
                file_path = raw_dir / filename
                
                try:
                    # Load graph data with weights_only=False for custom objects
                    loaded_data = torch.load(file_path, weights_only=False)
                    
                    # Handle different file formats
                    if isinstance(loaded_data, Data):
                        graph_data = loaded_data
                    elif isinstance(loaded_data, dict):
                        graph_data = loaded_data.get('graph_data')
                    else:
                        print(f"⚠️ Unknown format in {filename}")
                        continue
                    
                    if graph_data is None:
                        continue
                    
                    # Apply filters and transforms
                    if self.pre_filter is not None and not self.pre_filter(graph_data):
                        continue
                    if self.pre_transform is not None:
                        graph_data = self.pre_transform(graph_data)
                    
                    data_list.append(graph_data)
                    
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"✅ Processed {len(data_list)} graphs")

class OnDemandDocumentDataset(Dataset):
    """
    On-demand dataset for large collections of document graphs.
    Loads graphs from disk as needed to save memory.
    """
    
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """
        Initialize on-demand dataset.
        
        Args:
            root (str): Root directory containing graph files
            transform (Callable, optional): Transform to apply
            pre_transform (Callable, optional): Pre-transform to apply
        """
        self.graph_files = []
        self._load_file_list(root)
        super().__init__(root, transform, pre_transform)
    
    def _load_file_list(self, root: str):
        """Load list of available graph files."""
        root_path = Path(root)
        
        # Look for graph files in raw directory
        raw_dir = root_path / 'raw'
        if raw_dir.exists():
            self.graph_files = [
                raw_dir / f for f in os.listdir(raw_dir) 
                if f.endswith('_graph.pt')
            ]
        else:
            # Look directly in root
            self.graph_files = [
                root_path / f for f in os.listdir(root_path) 
                if f.endswith('_graph.pt')
            ]
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return raw file names."""
        return [f.name for f in self.graph_files]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return processed file names."""
        return []  # No preprocessing needed
    
    def download(self):
        """Download - not applicable."""
        pass
    
    def process(self):
        """Process - not needed for on-demand loading."""
        pass
    
    def len(self) -> int:
        """Return dataset size."""
        return len(self.graph_files)
    
    def get(self, idx: int) -> Data:
        """Get graph by index."""
        file_path = self.graph_files[idx]
        
        try:
            # Use weights_only=False for our custom graph data
            loaded_data = torch.load(file_path, weights_only=False)
            
            # Handle different file formats
            if isinstance(loaded_data, Data):
                graph_data = loaded_data
            elif isinstance(loaded_data, dict):
                graph_data = loaded_data.get('graph_data')
                if graph_data is None:
                    raise ValueError("No 'graph_data' key found")
            else:
                raise ValueError(f"Unknown data format in {file_path}")
            
            return graph_data
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            # Return empty graph as fallback
            return Data(x=torch.zeros((1, 22)),  # Updated to 22 features
                       y=torch.zeros(1, dtype=torch.long),
                       edge_index=torch.empty((2, 0), dtype=torch.long))


def create_data_loaders(dataset_path: str,
                       batch_size: int = 2,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       num_workers: int = 0,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_path (str): Path to dataset
        batch_size (int): Batch size
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        num_workers (int): Number of workers for data loading
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load dataset
    dataset = DocumentGraphDataset(dataset_path)
    
    # Set random seed
    torch.manual_seed(random_seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"📊 Dataset split:")
    print(f"   🎓 Train: {len(train_dataset)} graphs")
    print(f"   🔍 Val: {len(val_dataset)} graphs")
    print(f"   🧪 Test: {len(test_dataset)} graphs")
    
    return train_loader, val_loader, test_loader


def analyze_dataset(dataset_path: str):
    """
    Analyze dataset statistics.
    
    Args:
        dataset_path (str): Path to dataset
    """
    print("🔍 ANALYZING DATASET")
    print("=" * 40)
    
    try:
        dataset = DocumentGraphDataset(dataset_path)
        
        print(f"📊 Dataset size: {len(dataset)} graphs")
        
        # Analyze first few graphs
        if len(dataset) > 0:
            sample_sizes = []
            label_counts = {}
            
            for i in range(min(5, len(dataset))):
                data = dataset[i]
                sample_sizes.append(data.num_nodes)
                
                # Count labels
                for label in data.y.tolist():
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"📈 Sample graph sizes: {sample_sizes}")
            print(f"🏷️  Label distribution (sample): {label_counts}")
            
            # Feature information
            first_graph = dataset[0]
            print(f"🔧 Feature dimension: {first_graph.x.shape[1]}")
            print(f"📊 Node count range: {min(sample_sizes)}-{max(sample_sizes)}")
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")


# ========================================================================
# UTILITIES
# ========================================================================

def validate_dataset(dataset_path: str) -> bool:
    """
    Validate dataset integrity.
    
    Args:
        dataset_path (str): Path to dataset
        
    Returns:
        bool: True if dataset is valid
    """
    try:
        dataset = DocumentGraphDataset(dataset_path)
        
        if len(dataset) == 0:
            print("❌ Empty dataset")
            return False
        
        # Check first graph
        data = dataset[0]
        
        # Validate features
        if data.x.shape[1] != 22:  # Updated to 22 features
            print(f"❌ Expected 22 features, got {data.x.shape[1]}")
            return False
        
        # Validate labels
        if data.y.max() >= 6 or data.y.min() < 0:
            print(f"❌ Invalid label range: {data.y.min()}-{data.y.max()}")
            return False
        
        print("✅ Dataset validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        return False

"""
GNN Model Architecture for Document Layout Analysis.
Modular components optimized for <200MB size and <10s inference constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Optional


class DocumentGNN(nn.Module):
    """
    Graph Neural Network for Document Layout Analysis.
    Classifies text blocks into structural categories: TITLE, HH1, HH2, HH3, H4, BODY.
    
    Optimized for:
    - CPU inference (<10s)
    - Model size (<200MB)
    - High accuracy on document structure classification
    """
    
    def __init__(self, 
                 num_node_features: int = 22,  # Updated to 22 features
                 hidden_dim: int = 64,
                 num_classes: int = 6,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        """
        Initialize DocumentGNN model.
        
        Args:
            num_node_features (int): Number of input node features (22 from feature engineering)
            hidden_dim (int): Hidden dimension size (64 for compact model)
            num_classes (int): Number of output classes (6: BODY, HH1, HH2, HH3, H4, TITLE)
            num_layers (int): Number of GNN layers (2-3 for optimal performance)
            dropout (float): Dropout rate for regularization
            use_batch_norm (bool): Whether to use batch normalization
            activation (str): Activation function ('relu', 'gelu', 'elu')
        """
        super(DocumentGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # 1. Input embedding layer
        self.embedding = nn.Linear(num_node_features, hidden_dim)
        
        # 2. GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 3. Classifier layers
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            batch (torch.Tensor, optional): Batch assignment for each node
            
        Returns:
            torch.Tensor: Node predictions [num_nodes, num_classes]
        """
        # 1. Input embedding
        x = self.embedding(x)
        x = self.activation(x)
        
        # 2. GNN layers
        for i in range(self.num_layers):
            # Apply GNN convolution
            x = self.convs[i](x, edge_index)
            
            # Apply batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout (except last layer)
            if i < self.num_layers - 1:
                x = self.dropout_layer(x)
        
        # 3. Final classification
        x = self.dropout_layer(x)
        logits = self.classifier(x)
        
        return logits
    
    def get_model_size(self) -> float:
        """
        Calculate model size in MB.
        
        Returns:
            float: Model size in megabytes
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CompactDocumentGNN(nn.Module):
    """
    Ultra-compact version of DocumentGNN for extreme size constraints.
    Uses smaller dimensions and fewer layers.
    """
    
    def __init__(self,
                 num_node_features: int = 22,  # Updated to 22 features
                 hidden_dim: int = 32,
                 num_classes: int = 6,
                 dropout: float = 0.1):
        """
        Initialize compact DocumentGNN model.
        
        Args:
            num_node_features (int): Number of input features (22)
            hidden_dim (int): Hidden dimension (32 for ultra-compact)
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super(CompactDocumentGNN, self).__init__()
        
        # Single layer architecture for minimal size
        self.embedding = nn.Linear(num_node_features, hidden_dim)
        self.conv = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.embedding(x))
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
    
    def get_model_size(self) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def create_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_type (str): Type of model ('standard', 'compact')
        **kwargs: Model parameters
        
    Returns:
        nn.Module: Created model
    """
    if model_type == 'standard':
        return DocumentGNN(**kwargs)
    elif model_type == 'compact':
        return CompactDocumentGNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_model_info(model: nn.Module):
    """
    Print detailed model information.
    
    Args:
        model (nn.Module): Model to analyze
    """
    print(f"🔍 MODEL INFORMATION:")
    print(f"   📊 Model type: {model.__class__.__name__}")
    
    if hasattr(model, 'count_parameters'):
        print(f"   🔢 Parameters: {model.count_parameters():,}")
    else:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   🔢 Parameters: {param_count:,}")
    
    if hasattr(model, 'get_model_size'):
        print(f"   💾 Model size: {model.get_model_size():.2f} MB")
    
    # Architecture details
    if hasattr(model, 'hidden_dim'):
        print(f"   🏗️  Hidden dim: {model.hidden_dim}")
    if hasattr(model, 'num_layers'):
        print(f"   📚 Layers: {model.num_layers}")
    if hasattr(model, 'dropout'):
        print(f"   🎲 Dropout: {model.dropout}")


# ========================================================================
# QUANTIZATION UTILITIES
# ========================================================================

def prepare_model_for_quantization(model: nn.Module) -> nn.Module:
    """
    Prepare model for quantization by adding QuantStub and DeQuantStub.
    
    Args:
        model (nn.Module): Model to prepare
        
    Returns:
        nn.Module: Prepared model
    """
    # Add quantization stubs
    model.quant = torch.quantization.QuantStub()
    model.dequant = torch.quantization.DeQuantStub()
    
    # Modify forward method to include quantization
    original_forward = model.forward
    
    def quantized_forward(self, x, edge_index, batch=None):
        x = self.quant(x)
        x = original_forward(x, edge_index, batch)
        x = self.dequant(x)
        return x
    
    model.forward = quantized_forward.__get__(model, model.__class__)
    return model


def quantize_model(model: nn.Module, calibration_data) -> nn.Module:
    """
    Apply post-training quantization to model.
    
    Args:
        model (nn.Module): Trained model
        calibration_data: Data for calibration
        
    Returns:
        nn.Module: Quantized model
    """
    # Prepare for quantization
    model.eval()
    model = prepare_model_for_quantization(model)
    
    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with sample data
    with torch.no_grad():
        for data in calibration_data:
            model_prepared(data.x, data.edge_index)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized

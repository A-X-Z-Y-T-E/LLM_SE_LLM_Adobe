"""
Training pipeline for Document Layout Analysis GNN.
Handles model training, validation, and optimization for deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

from .models import DocumentGNN, CompactDocumentGNN, print_model_info
from .dataset import create_data_loaders


class GNNTrainer:
    """
    Trainer class for Document Layout Analysis GNN.
    Handles training loop, validation, early stopping, and model optimization.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 patience: int = 10,
                 save_dir: str = "model_checkpoints"):
        """
        Initialize GNN trainer.
        
        Args:
            model (nn.Module): Model to train
            device (torch.device): Device for training
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            patience (int): Early stopping patience
            save_dir (str): Directory to save models
        """
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_nodes = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(outputs, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * batch.num_nodes
            predicted = outputs.argmax(dim=1)
            correct_predictions += (predicted == batch.y).sum().item()
            total_nodes += batch.num_nodes
        
        avg_loss = total_loss / total_nodes
        accuracy = correct_predictions / total_nodes
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Validate model for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy, detailed_metrics)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_nodes = 0
        
        # For detailed metrics
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(outputs, batch.y)
                
                total_loss += loss.item() * batch.num_nodes
                predicted = outputs.argmax(dim=1)
                correct_predictions += (predicted == batch.y).sum().item()
                total_nodes += batch.num_nodes
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().tolist())
                all_targets.extend(batch.y.cpu().tolist())
        
        avg_loss = total_loss / total_nodes
        accuracy = correct_predictions / total_nodes
        
        # Calculate per-class metrics
        detailed_metrics = self._calculate_detailed_metrics(all_predictions, all_targets)
        
        return avg_loss, accuracy, detailed_metrics
    
    def _calculate_detailed_metrics(self, predictions: List, targets: List) -> Dict:
        """
        Calculate detailed per-class metrics.
        
        Args:
            predictions (List): Predicted labels
            targets (List): True labels
            
        Returns:
            Dict: Detailed metrics
        """
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Label names
        label_names = ['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE']
        
        metrics = {
            'per_class': {
                label_names[i]: {
                    'precision': float(precision[i]) if i < len(precision) else 0.0,
                    'recall': float(recall[i]) if i < len(recall) else 0.0,
                    'f1': float(f1[i]) if i < len(f1) else 0.0,
                    'support': int(support[i]) if i < len(support) else 0
                }
                for i in range(len(label_names))
            },
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              scheduler_type: str = 'reduce_lr_on_plateau') -> Dict:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Maximum number of epochs
            scheduler_type (str): Type of learning rate scheduler
            
        Returns:
            Dict: Training history
        """
        print(f"🚀 STARTING TRAINING")
        print("=" * 50)
        
        # Print model info
        print_model_info(self.model)
        print()
        
        # Initialize scheduler
        if scheduler_type == 'reduce_lr_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            scheduler = None
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc, detailed_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if scheduler:
                if scheduler_type == 'reduce_lr_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping and best model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Save best model
                self._save_model('best_model.pth', detailed_metrics)
                print(f"   💾 New best model saved (Val Loss: {val_loss:.4f})")
                
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"   🛑 Early stopping triggered after {epoch+1} epochs")
                    print(f"   🏆 Best epoch: {self.best_epoch} (Val Loss: {self.best_val_loss:.4f})")
                    break
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETE")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best epoch: {self.best_epoch}")
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        print(f"🎯 Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history
    
    def _save_model(self, filename: str, metrics: Dict = None):
        """
        Save model and training state.
        
        Args:
            filename (str): Filename to save
            metrics (Dict, optional): Validation metrics
        """
        save_path = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'model_config': {
                'model_class': self.model.__class__.__name__,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if metrics:
            checkpoint['validation_metrics'] = metrics
        
        torch.save(checkpoint, save_path)
    
    def load_model(self, filename: str):
        """
        Load model from checkpoint.
        
        Args:
            filename (str): Filename to load
        """
        load_path = self.save_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', {})
        
        print(f"✅ Model loaded from {filename}")
        print(f"   🏆 Best validation loss: {self.best_val_loss:.4f}")
        print(f"   🎯 Best validation accuracy: {self.best_val_acc:.4f}")


def optimize_model_for_deployment(model_path: str, 
                                 calibration_loader: DataLoader,
                                 output_dir: str = "optimized_models") -> str:
    """
    Optimize trained model for deployment with quantization.
    
    Args:
        model_path (str): Path to trained model
        calibration_loader (DataLoader): Data for calibration
        output_dir (str): Output directory
        
    Returns:
        str: Path to optimized model
    """
    print("🔧 OPTIMIZING MODEL FOR DEPLOYMENT")
    print("=" * 40)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Recreate model with correct feature count (22)
    model = DocumentGNN(num_node_features=22, hidden_dim=64, num_classes=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"📊 Original model size: {model.get_model_size():.2f} MB")
    
    # Apply quantization
    try:
        # Prepare for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model, inplace=False)
        
        # Calibration
        print("🎯 Calibrating model...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= 10:  # Limit calibration samples
                    break
                _ = model_prepared(batch.x, batch.edge_index, batch.batch)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)
        
        # Save quantized model
        quantized_path = output_dir / "model_quantized.pth"
        torch.save({
            'model_state_dict': model_quantized.state_dict(),
            'model_config': checkpoint.get('model_config', {}),
            'quantized': True,
            'timestamp': datetime.now().isoformat()
        }, quantized_path)
        
        print(f"✅ Quantized model saved: {quantized_path}")
        
        return str(quantized_path)
        
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        print("💾 Saving original model instead...")
        
        # Save original model
        original_path = output_dir / "model_optimized.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': checkpoint.get('model_config', {}),
            'quantized': False,
            'timestamp': datetime.now().isoformat()
        }, original_path)
        
        return str(original_path)


def main_training_pipeline(dataset_path: str,
                          model_type: str = 'standard',
                          hidden_dim: int = 64,
                          num_epochs: int = 100,
                          batch_size: int = 2,
                          learning_rate: float = 0.001):
    """
    Complete training pipeline for Document Layout Analysis.
    
    Args:
        dataset_path (str): Path to dataset
        model_type (str): Type of model ('standard', 'compact')
        hidden_dim (int): Hidden dimension size
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
    """
    print("🚀 DOCUMENT LAYOUT ANALYSIS - TRAINING PIPELINE")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create data loaders
    print("\n📊 Loading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size
    )
    
    # Create model with correct feature count (22)
    print(f"\n🏗️  Creating {model_type} model...")
    if model_type == 'standard':
        model = DocumentGNN(
            num_node_features=22,  # Fixed to 22 features
            hidden_dim=hidden_dim,
            num_classes=6,
            num_layers=2
        )
    else:
        model = CompactDocumentGNN(
            num_node_features=22,  # Fixed to 22 features
            hidden_dim=32,
            num_classes=6
        )
    
    # Initialize trainer
    trainer = GNNTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train model
    print(f"\n🎓 Starting training...")
    history = trainer.train(train_loader, val_loader, num_epochs, 
                          scheduler_type='reduce_lr_on_plateau')
    
    # Optimize for deployment
    print(f"\n🔧 Optimizing for deployment...")
    best_model_path = trainer.save_dir / "best_model.pth"
    optimized_path = optimize_model_for_deployment(
        str(best_model_path),
        val_loader
    )
    
    print(f"\n🎉 Training pipeline complete!")
    print(f"   💾 Best model: {best_model_path}")
    print(f"   ⚡ Optimized model: {optimized_path}")
    
    return history, str(optimized_path)


if __name__ == "__main__":
    # Example usage
    dataset_path = "training_data/graphs"
    history, optimized_model = main_training_pipeline(
        dataset_path=dataset_path,
        model_type='standard',
        hidden_dim=64,
        num_epochs=50,
        batch_size=2,
        learning_rate=0.001
    )

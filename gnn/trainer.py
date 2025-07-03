import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from utils import get_device, ensure_device_consistency, check_device_consistency, move_data_to_model_device


class GNNTrainer:
    """
    Trainer class for Graph Neural Networks for Finite Element Analysis
    """
    
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Initialize the trainer
        
        Args:
            model: MPNN model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration dictionary
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Device - explicitly use CUDA device 0 if available
        self.device = get_device()
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get('batch_size', 1), 
            shuffle=False  # No shuffling for single mesh
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config.get('batch_size', 1), 
            shuffle=False
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Create output directory - use absolute path relative to script location
        output_dir_name = config.get('output_dir', 'gnn_outputs')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, output_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device and ensure all tensors are on the same device
            batch = ensure_device_consistency(batch, self.device)
            
            # Debug: Check device consistency (only in first few iterations)
            if num_batches < 3:
                check_device_consistency(batch, self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            loss = self.criterion(predictions, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        # For single mesh training, use training mode to avoid BatchNorm differences
        # since we're using the same data for both training and validation
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device and ensure all tensors are on the same device
                batch = ensure_device_consistency(batch, self.device)
                
                # Forward pass
                predictions = self.model(batch)
                loss = self.criterion(predictions, batch.y)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs):
        """Train the model for specified number of epochs"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss**0.5) # RMSE
            
            # Validation (using same model state as training for fair comparison)
            val_loss = self.validate()
            self.val_losses.append(val_loss**0.5) # RMSE
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                print(f"New best validation loss: {val_loss:.6f}")
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch + 1)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final model and plots
        self.save_final_results()
    
    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_results(self):
        """Save final model and training plots"""
        # Save model
        model_path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Plot training history
        self.plot_training_history()
        
        print(f"Final results saved to: {self.output_dir}")
    
    def plot_training_history(self):
        """Plot training and validation loss history"""
        plt.figure(figsize=(6, 5))  
        plt.plot(self.train_losses, '-', label='Training Loss', color='k', linewidth=2)
        plt.plot(self.val_losses, '--', label='Validation Loss', color='g', linewidth=2)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        
        # Safely calculate minimum loss with additional checks
        try:
            train_min = min(self.train_losses) if self.train_losses else 1e-4
            val_min = min(self.val_losses) if self.val_losses else 1e-4
            min_loss = min(train_min, val_min)
            
            # Ensure min_loss is a valid number
            if not np.isfinite(min_loss) or min_loss <= 0:
                min_loss = 1e-4
        except (ValueError, TypeError):
            min_loss = 1e-4
        
        if min_loss < 1e-4:
            plt.ylim(min_loss * 0.5, 1e0)
        else:
            plt.ylim(1e-4, 1e0)
        # plt.title('Training History', fontsize=16, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, test_dataset=None):
        """Evaluate the model on test dataset"""
        if test_dataset is None:
            test_dataset = self.val_dataset
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = ensure_device_consistency(batch, self.device)
                predictions = self.model(batch)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(batch.y.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        mse_loss = nn.MSELoss()(all_predictions, all_targets)
        mae_loss = nn.L1Loss()(all_predictions, all_targets)
        rmse_loss = torch.sqrt(mse_loss)  # Calculate RMSE
        
        # Calculate relative errors for each output
        relative_errors = []
        for i in range(all_predictions.shape[1]):
            rel_error = torch.mean(torch.abs(
                (all_predictions[:, i] - all_targets[:, i]) / (all_targets[:, i] + 1e-8)
            ))
            relative_errors.append(rel_error.item())
        
        results = {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'rmse_loss': rmse_loss.item(),  # Add RMSE loss
            'relative_errors': {
                'u': relative_errors[0],
                'v': relative_errors[1], 
                'w': relative_errors[2],
                'von_mises': relative_errors[3]
            }
        }
        
        print("\nEvaluation Results:")
        print(f"MSE Loss: {mse_loss.item():.6f}")
        print(f"RMSE Loss: {rmse_loss.item():.6f}")
        print(f"MAE Loss: {mae_loss.item():.6f}")
        print("Relative Errors:")
        for name, error in results['relative_errors'].items():
            print(f"  {name}: {error:.4f}")
        
        return results, all_predictions, all_targets
    
    def predict_single_mesh(self, data):
        """Predict on a single mesh"""
        self.model.eval()
        with torch.no_grad():
            data = move_data_to_model_device(data, self.model)
            predictions = self.model(data)
            return predictions.cpu()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Model loaded on device: {self.device}")


def create_trainer(model, train_dataset, val_dataset, config):
    """Create a trainer instance with default configuration"""
    default_config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'output_dir': 'gnn_outputs',
        'save_interval': 10
    }
    
    # Update with provided config
    default_config.update(config)
    
    return GNNTrainer(model, train_dataset, val_dataset, default_config) 
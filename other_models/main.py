import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import StepLR
import datetime

from dataset import HeatDataset
from utils.callback_wrapper import callback_warpper

def get_args():
    parser = argparse.ArgumentParser(description='Train a PyTorch Lightning Model')
    parser.add_argument('--root', type=str, default='/mnt/a/jgz1751/icml_parse_dataset/', help='Root directory for dataset')
    parser.add_argument('--is_train', type=int, default=1, help='Training or evaluation')
    parser.add_argument('--model_name', type=str, default='mlp', help='Model name')
    parser.add_argument('--dataset_type', type=str, default='train_full', help='How much data to use for training')
    parser.add_argument('--clean_noise_type', type=str, default='clean', help='Clean or noisy data')
    parser.add_argument('--input_dim', type=int, default=5, help='Input dimension')
    parser.add_argument('--hidden_layers', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('--hidden_dim', type=int, default=50, help='Hidden dimension size')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=200, help='Max training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', choices=['StepLR', 'ReduceLROnPlateau'], help='Learning rate scheduler')
    # ReduceLROnPlateau
    parser.add_argument('--lr_reduce_factor', type=float, default=0.8, help='lr_reduce_factor')
    parser.add_argument('--lr_reduce_patience', type=int, default=1, help='lr_reduce_patience')    
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID to use')
    return parser.parse_args()

class PLModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        if args.model_name == 'siren':
            from models.siren import Siren
            self.model = Siren(
                input_dim=args.input_dim, 
                hidden_layers=args.hidden_layers, 
                hidden_dim=args.hidden_dim, 
                output_dim=args.output_dim
            )
        elif 'mlp' in args.model_name:
            from models.mlp import MLP
            self.model = MLP(
                input_dim=args.input_dim, 
                hidden_layers=args.hidden_layers, 
                hidden_dim=args.hidden_dim, 
                output_dim=args.output_dim
            )
        else:
            raise ValueError(f"Invalid model_name: {args.model_name}")
    
    def forward(self, x):
        return self.model(x)
    
    def rmse_loss(self, y_hat, y):
        mse = torch.mean((y_hat - y) ** 2)
        return torch.sqrt(mse)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.rmse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.rmse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.rmse_loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        
        if self.args.lr_scheduler == 'StepLR':        
            scheduler = StepLR(optimizer, step_size=self.args.max_epochs//5, gamma=0.3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                }
            }
        elif self.args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.args.lr_reduce_factor, patience=self.args.lr_reduce_patience)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss', 
                    'interval': 'epoch',
                    'frequency': 1,
                    'name': 'lr',  
                }
            }
        else:
            return optimizer

def evaluate_model(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    mse_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            # deormalize y_hat and y
            y_hat = dataloader.dataset.denormalize_u(y_hat)
            y = dataloader.dataset.denormalize_u(y)
            
            mse_loss += torch.sum((y_hat - y) ** 2).item()
            total_samples += len(y)
    
    rmse = (mse_loss / total_samples) ** 0.5
    return rmse

if __name__ == "__main__":
    args = get_args()

    # Load datasets
    if args.dataset_type == 'train_full':
        train_dataset = HeatDataset(args.root + f'train_{args.clean_noise_type}.csv', dataset_type='train_full', clean_noise_type=args.clean_noise_type)
        if args.clean_noise_type == 'clean':
            if "siren" in args.model_name:
                best_model_path = "best_model-2025-01-23_00-06-20-siren_small_datatrain_full_hl6_hd50_lr0.0001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
            else:
                best_model_path = "best_model-2025-01-25_00-52-58-mlp_silu_datatrain_full_hl5_hd100_lr0.001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
        else:
            if "siren" in args.model_name:
                best_model_path = "best_model-2025-01-29_22-47-28-siren_datatrain_full_hl6_hd50_lr0.0001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
            else:
                best_model_path = "best_model-2025-01-29_22-40-37-mlp_silu_datatrain_full_hl5_hd100_lr0.001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
    elif args.dataset_type == 'train_10':
        train_dataset = HeatDataset(args.root + f'train_10_{args.clean_noise_type}.csv', dataset_type='train_10', clean_noise_type=args.clean_noise_type)
        if args.clean_noise_type == 'clean':
            if "siren" in args.model_name:
                best_model_path = "best_model-2025-01-23_00-06-17-siren_small_datatrain_10_hl6_hd50_lr0.0001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
            else:
                best_model_path = "best_model-2025-01-25_20-16-48-mlp_silu_datatrain_10_hl5_hd100_lr0.001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
        else:
            if "siren" in args.model_name:
                best_model_path = "best_model-2025-01-29_22-47-25-siren_datatrain_10_hl6_hd50_lr0.0001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
            else:
                best_model_path = "best_model-2025-01-28_22-56-11-mlp_silu_datatrain_10_hl5_hd100_lr0.001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
    elif args.dataset_type == 'train_30':
        train_dataset = HeatDataset(args.root + f'train_30_{args.clean_noise_type}.csv', dataset_type='train_30', clean_noise_type=args.clean_noise_type)
        if args.clean_noise_type == 'clean':
            if "siren" in args.model_name:
                best_model_path = "best_model-2025-01-23_00-06-18-siren_small_datatrain_30_hl6_hd50_lr0.0001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
            else:
                best_model_path = "best_model-2025-01-25_20-16-49-mlp_silu_datatrain_30_hl5_hd100_lr0.001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
        else:
            if "siren" in args.model_name:
                best_model_path = "best_model-2025-01-29_22-47-26-siren_datatrain_30_hl6_hd50_lr0.0001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
            else:
                best_model_path = "best_model-2025-01-28_22-56-12-mlp_silu_datatrain_30_hl5_hd100_lr0.001_bs128_epochs2000_lr_schedulerReduceLROnPlateau_lr_reduce_factor0.9_lr_reduce_patience1.ckpt"
    else:
        raise ValueError(f"Invalid dataset_type: {args.dataset_type}")
    val_dataset = HeatDataset(args.root + f'val_{args.clean_noise_type}.csv', clean_noise_type=args.clean_noise_type)
    test_dataset = HeatDataset(args.root + f'test_{args.clean_noise_type}.csv', clean_noise_type=args.clean_noise_type)
    test_clean_dataset = HeatDataset(args.root + f'test_clean.csv', clean_noise_type=args.clean_noise_type, is_for_test=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_clean_loader = DataLoader(test_clean_dataset, batch_size=args.batch_size)

    # Initialize logger
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.lr_scheduler == 'ReduceLROnPlateau':
        log_name = (
            f"{time_str}-{args.model_name}"
			f"_data{args.dataset_type}"
            f"_hl{args.hidden_layers}"
            f"_hd{args.hidden_dim}"
            f"_lr{args.lr}"
            f"_bs{args.batch_size}"
            f"_epochs{args.max_epochs}"
            f"_lr_scheduler{args.lr_scheduler}"
            f"_lr_reduce_factor{args.lr_reduce_factor}"
            f"_lr_reduce_patience{args.lr_reduce_patience}"
        )
    elif args.lr_scheduler == 'StepLR':
        log_name = (
            f"{time_str}-{args.model_name}"
			f"_data{args.dataset_type}"
            f"_hl{args.hidden_layers}"
            f"_hd{args.hidden_dim}"
            f"_lr{args.lr}"
            f"_bs{args.batch_size}"
            f"_epochs{args.max_epochs}"
            f"_lr_scheduler{args.lr_scheduler}"
        )
    logger = TensorBoardLogger(args.log_dir, name=log_name)
    callbacks = callback_warpper(log_name, args)

    # Train model with checkpointing and early stopping
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        logger=logger, 
        log_every_n_steps=10, 
        devices=[args.gpu_id], 
        accelerator="gpu",
        callbacks=callbacks
    )

    if args.is_train:
        model = PLModel(args)
        trainer.fit(model, train_loader, val_loader)

        # Test model using the best checkpoint
        best_model_path = callbacks[0].best_model_path
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            model = PLModel.load_from_checkpoint(best_model_path, args=args)

        trainer.test(model, test_loader)
    else:
        # best_model_path = args.best_model_path
        model = PLModel.load_from_checkpoint("./checkpoints/" + best_model_path, args=args).to("cuda")

        train_rmse = evaluate_model(model, train_loader)
        val_rmse   = evaluate_model(model, val_loader)
        test_rmse  = evaluate_model(model, test_loader)
        test_clean_rmse = evaluate_model(model, test_clean_loader)

        print(f"Train RMSE: {train_rmse:.2e}")
        print(f"Val RMSE:   {val_rmse:.2e}")
        print(f"Test RMSE:  {test_rmse:.2e}")
        print(f"Test Clean RMSE:  {test_clean_rmse:.2e}")
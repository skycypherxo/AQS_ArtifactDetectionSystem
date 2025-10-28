"""
Training Script for EfficientNet-Based Video Artifact Detection Model
====================================================================

Uses EfficientNet-B0 as backbone with transfer learning for better performance.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import os
import json
import time
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from efficientnet_artifact_detector import (
    EfficientNetArtifactDetector, 
    ArtifactDetectionLoss,
    create_data_loaders,
    calculate_metrics,
    ArtifactType
)

class EfficientNetTrainer:
    """Trainer for EfficientNet-based artifact detection model"""
    
    def __init__(self, 
                 model: EfficientNetArtifactDetector,
                 train_loader,
                 test_loader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 save_dir: str = 'checkpoints',
                 use_mixed_precision: bool = True):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.use_mixed_precision = use_mixed_precision
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = ArtifactDetectionLoss(
            classification_weight=1.0,
            intensity_weight=1.2,
            quality_weight=0.8
        )
        
        # Optimizer with different learning rates for backbone and new layers
        backbone_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name or 'feature_extractor' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pre-trained backbone
            {'params': new_params, 'lr': learning_rate}  # Higher LR for new layers
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=learning_rate * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Tensorboard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/efficientnet_artifact_detector_{timestamp}')
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        self.max_patience = 10
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Metrics accumulation
        epoch_metrics = {
            'classification_loss': 0.0,
            'intensity_loss': 0.0,
            'quality_loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'intensity_mae': 0.0,
            'quality_mae': 0.0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            frames = batch['frames'].to(self.device, non_blocking=True)
            artifact_labels = batch['artifact_labels'].to(self.device, non_blocking=True)
            intensity_labels = batch['intensity_labels'].to(self.device, non_blocking=True)
            quality_labels = batch['quality_score'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with autocast():
                    predictions = self.model(frames)
                    
                    targets = {
                        'artifact_labels': artifact_labels,
                        'intensity_labels': intensity_labels,
                        'quality_score': quality_labels
                    }
                    
                    loss_dict = self.criterion(predictions, targets)
                    total_loss_batch = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(frames)
                
                targets = {
                    'artifact_labels': artifact_labels,
                    'intensity_labels': intensity_labels,
                    'quality_score': quality_labels
                }
                
                loss_dict = self.criterion(predictions, targets)
                total_loss_batch = loss_dict['total_loss']
                
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update metrics
            batch_size = frames.size(0)
            total_loss += total_loss_batch.item() * batch_size
            total_samples += batch_size
            
            # Calculate batch metrics
            with torch.no_grad():
                batch_metrics = calculate_metrics(predictions, targets)
                
                for key in epoch_metrics:
                    if key in loss_dict:
                        epoch_metrics[key] += loss_dict[key].item() * batch_size
                    elif key in batch_metrics:
                        epoch_metrics[key] += batch_metrics[key].item() * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Avg Loss': f'{total_loss / total_samples:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard every 100 batches
            if batch_idx % 100 == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', total_loss_batch.item(), global_step)
        
        # Average metrics over epoch
        avg_loss = total_loss / total_samples
        for key in epoch_metrics:
            epoch_metrics[key] /= total_samples
        
        return avg_loss, epoch_metrics
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # Metrics accumulation
        epoch_metrics = {
            'classification_loss': 0.0,
            'intensity_loss': 0.0,
            'quality_loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'intensity_mae': 0.0,
            'quality_mae': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Validation'):
                frames = batch['frames'].to(self.device, non_blocking=True)
                artifact_labels = batch['artifact_labels'].to(self.device, non_blocking=True)
                intensity_labels = batch['intensity_labels'].to(self.device, non_blocking=True)
                quality_labels = batch['quality_score'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        predictions = self.model(frames)
                        
                        targets = {
                            'artifact_labels': artifact_labels,
                            'intensity_labels': intensity_labels,
                            'quality_score': quality_labels
                        }
                        
                        loss_dict = self.criterion(predictions, targets)
                        total_loss_batch = loss_dict['total_loss']
                else:
                    predictions = self.model(frames)
                    
                    targets = {
                        'artifact_labels': artifact_labels,
                        'intensity_labels': intensity_labels,
                        'quality_score': quality_labels
                    }
                    
                    loss_dict = self.criterion(predictions, targets)
                    total_loss_batch = loss_dict['total_loss']
                
                # Update metrics
                batch_size = frames.size(0)
                total_loss += total_loss_batch.item() * batch_size
                total_samples += batch_size
                
                # Calculate batch metrics
                batch_metrics = calculate_metrics(predictions, targets)
                
                for key in epoch_metrics:
                    if key in loss_dict:
                        epoch_metrics[key] += loss_dict[key].item() * batch_size
                    elif key in batch_metrics:
                        epoch_metrics[key] += batch_metrics[key].item() * batch_size
        
        # Average metrics over epoch
        avg_loss = total_loss / total_samples
        for key in epoch_metrics:
            epoch_metrics[key] /= total_samples
        
        return avg_loss, epoch_metrics
    
    def save_checkpoint(self, is_best: bool = False, is_best_f1: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoints
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_loss_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"New best loss model saved: {self.best_loss:.4f}")
        
        if is_best_f1:
            best_f1_path = os.path.join(self.save_dir, 'best_f1_checkpoint.pth')
            torch.save(checkpoint, best_f1_path)
            print(f"New best F1 model saved: {self.best_f1:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.best_f1 = checkpoint.get('best_f1', 0.0)
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"Checkpoint loaded from epoch {self.epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    def train(self, num_epochs: int, resume: bool = False):
        """Main training loop"""
        
        if resume:
            checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
            self.load_checkpoint(checkpoint_path)
        
        print(f"EfficientNet-based Training Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Mixed precision: {self.use_mixed_precision}")
        print(f"  Backbone LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  New layers LR: {self.optimizer.param_groups[1]['lr']:.2e}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate/Backbone', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Learning_Rate/New_Layers', self.optimizer.param_groups[1]['lr'], epoch)
            
            # Log metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            # Print epoch results
            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train F1: {train_metrics['f1_score']:.4f} | Val F1: {val_metrics['f1_score']:.4f}")
            print(f"Train Quality MAE: {train_metrics['quality_mae']:.4f} | Val Quality MAE: {val_metrics['quality_mae']:.4f}")
            
            # Save checkpoints
            is_best_loss = val_loss < self.best_loss
            is_best_f1 = val_metrics['f1_score'] > self.best_f1
            
            if is_best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
            
            if is_best_f1:
                self.best_f1 = val_metrics['f1_score']
            
            self.save_checkpoint(is_best_loss, is_best_f1)
            
            # Early stopping
            if not is_best_loss:
                self.patience_counter += 1
                
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Best validation F1: {self.best_f1:.4f}")
        
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet-based Video Artifact Detection Model')
    parser.add_argument('--data_root', type=str, default='../vimeo_settuplet_1',
                       help='Path to Vimeo dataset root')
    parser.add_argument('--train_split', type=str, default='../vimeo_settuplet_1/sep_trainlist.txt',
                       help='Path to train split file')
    parser.add_argument('--test_split', type=str, default='../vimeo_settuplet_1/sep_testlist.txt',
                       help='Path to test split file')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.mixed_precision = False
    
    print("EfficientNet-based Training Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_data_loaders(
        data_root=args.data_root,
        train_split=args.train_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create EfficientNet-based model
    print("\nCreating EfficientNet-based model...")
    num_artifact_types = len(ArtifactType.get_all_types())
    model = EfficientNetArtifactDetector(
        num_artifact_types=num_artifact_types,
        sequence_length=7
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = EfficientNetTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        use_mixed_precision=args.mixed_precision
    )
    
    # Start training
    trainer.train(num_epochs=args.num_epochs, resume=args.resume)

if __name__ == "__main__":
    main()
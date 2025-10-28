"""
EfficientNet-Based Video Artifact Detection Model
================================================

Uses EfficientNet-B0 as the backbone for better feature extraction
and builds temporal analysis on top for video artifact detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import cv2
import os
from PIL import Image
import json
import random
from typing import Dict, List, Tuple, Optional
import math

class ArtifactType:
    """Enumeration of different artifact types"""
    COMPRESSION_BLOCKING = 0
    COMPRESSION_RINGING = 1
    MOTION_GHOSTING = 2
    MOTION_JUDDER = 3
    NOISE_GAUSSIAN = 4
    NOISE_IMPULSE = 5
    BLUR_MOTION = 6
    BLUR_DEFOCUS = 7
    COLOR_BANDING = 8
    COLOR_SATURATION = 9
    
    @classmethod
    def get_all_types(cls):
        return [cls.COMPRESSION_BLOCKING, cls.COMPRESSION_RINGING, cls.MOTION_GHOSTING,
                cls.MOTION_JUDDER, cls.NOISE_GAUSSIAN, cls.NOISE_IMPULSE,
                cls.BLUR_MOTION, cls.BLUR_DEFOCUS, cls.COLOR_BANDING, cls.COLOR_SATURATION]
    
    @classmethod
    def get_type_name(cls, artifact_type):
        names = {
            cls.COMPRESSION_BLOCKING: "compression_blocking",
            cls.COMPRESSION_RINGING: "compression_ringing", 
            cls.MOTION_GHOSTING: "motion_ghosting",
            cls.MOTION_JUDDER: "motion_judder",
            cls.NOISE_GAUSSIAN: "noise_gaussian",
            cls.NOISE_IMPULSE: "noise_impulse",
            cls.BLUR_MOTION: "blur_motion",
            cls.BLUR_DEFOCUS: "blur_defocus",
            cls.COLOR_BANDING: "color_banding",
            cls.COLOR_SATURATION: "color_saturation"
        }
        return names.get(artifact_type, "unknown")

class ArtifactGenerator:
    """Simplified but effective artifact generator"""
    
    @staticmethod
    def add_compression_blocking(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add JPEG compression artifacts"""
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
        processed_frames = []
        
        for frame in frames_np:
            frame_uint8 = (frame * 255).astype(np.uint8)
            quality = max(10, int(100 - intensity * 80))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', frame_uint8, encode_param)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            processed_frame = decoded.astype(np.float32) / 255.0
            processed_frames.append(processed_frame)
        
        return torch.tensor(np.stack(processed_frames)).permute(0, 3, 1, 2)
    
    @staticmethod
    def add_compression_ringing(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add ringing artifacts"""
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        
        processed_frames = []
        for frame in frames:
            ringing = F.conv2d(frame.unsqueeze(0), kernel, padding=1, groups=3).squeeze(0)
            processed_frame = frame + intensity * 0.1 * ringing
            processed_frames.append(torch.clamp(processed_frame, 0, 1))
        
        return torch.stack(processed_frames)
    
    @staticmethod
    def add_motion_ghosting(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add ghosting effects"""
        processed_frames = [frames[0]]
        
        for i in range(1, len(frames)):
            alpha = 0.7 + intensity * 0.2
            beta = intensity * 0.3
            ghosted_frame = alpha * frames[i] + beta * frames[i-1]
            processed_frames.append(torch.clamp(ghosted_frame, 0, 1))
        
        return torch.stack(processed_frames)
    
    @staticmethod
    def add_motion_judder(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add judder by frame duplication"""
        if intensity < 0.3:
            return frames
            
        processed_frames = []
        skip_pattern = max(1, int(intensity * 3))
        
        for i in range(len(frames)):
            if i % skip_pattern == 0 and i > 0:
                processed_frames.append(processed_frames[-1])
            else:
                processed_frames.append(frames[i])
        
        return torch.stack(processed_frames[:len(frames)])
    
    @staticmethod
    def add_gaussian_noise(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add Gaussian noise"""
        noise_std = intensity * 0.08
        noise = torch.randn_like(frames) * noise_std
        return torch.clamp(frames + noise, 0, 1)
    
    @staticmethod
    def add_impulse_noise(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add salt and pepper noise"""
        noise_prob = intensity * 0.03
        mask = torch.rand_like(frames) < noise_prob
        salt_pepper = torch.rand_like(frames)
        noise_values = torch.where(salt_pepper > 0.5, torch.ones_like(frames), torch.zeros_like(frames))
        return torch.where(mask, noise_values, frames)
    
    @staticmethod
    def add_motion_blur(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add motion blur"""
        kernel_size = int(3 + intensity * 8)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        kernel[center, :] = 1.0 / kernel_size
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        
        processed_frames = []
        for frame in frames:
            blurred = F.conv2d(frame.unsqueeze(0), kernel, padding=kernel_size//2, groups=3)
            processed_frames.append(blurred.squeeze(0))
        
        return torch.stack(processed_frames)
    
    @staticmethod
    def add_defocus_blur(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add defocus blur"""
        kernel_size = int(3 + intensity * 6)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        sigma = intensity * 2.0
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        
        processed_frames = []
        for frame in frames:
            blurred = F.conv2d(frame.unsqueeze(0), kernel, padding=kernel_size//2, groups=3)
            processed_frames.append(blurred.squeeze(0))
        
        return torch.stack(processed_frames)
    
    @staticmethod
    def add_color_banding(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add color banding"""
        levels = max(8, int(256 - intensity * 200))
        quantized = torch.round(frames * (levels - 1)) / (levels - 1)
        return torch.clamp(quantized, 0, 1)
    
    @staticmethod
    def add_color_saturation_issues(frames: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Add color saturation issues"""
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
        processed_frames = []
        
        for frame in frames_np:
            frame_uint8 = (frame * 255).astype(np.uint8)
            hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV)
            
            saturation_factor = 1.0 + intensity * random.choice([-1, 1]) * 0.6
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            processed_frame = rgb.astype(np.float32) / 255.0
            processed_frames.append(processed_frame)
        
        return torch.tensor(np.stack(processed_frames)).permute(0, 3, 1, 2)

class TemporalAttention(nn.Module):
    """Temporal attention for video sequences"""
    
    def __init__(self, feature_dim: int, sequence_length: int = 7):
        super().__init__()
        self.sequence_length = sequence_length
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, feature_dim)
        attention_weights = self.attention(x)  # (batch, sequence_length, 1)
        attention_weights = self.softmax(attention_weights)
        
        # Apply attention
        attended = torch.sum(x * attention_weights, dim=1)  # (batch, feature_dim)
        return attended

class EfficientNetArtifactDetector(nn.Module):
    """
    EfficientNet-based Video Artifact Detection Model
    """
    
    def __init__(self, num_artifact_types: int = 10, sequence_length: int = 7):
        super().__init__()
        
        self.num_artifact_types = num_artifact_types
        self.sequence_length = sequence_length
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Remove the classifier to get features
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get the feature dimension from EfficientNet-B0
        self.feature_dim = 1280  # EfficientNet-B0 output features
        
        # This will be set after feature_processor is defined
        
        # Enhanced feature processing
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Update temporal attention to use correct feature dimension
        self.temporal_attention = TemporalAttention(256, sequence_length)
        
        # Multi-task heads
        self.artifact_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_artifact_types)
        )
        
        self.intensity_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_artifact_types)
        )
        
        self.quality_scorer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize new layers
        self._initialize_new_layers()
        
    def _initialize_new_layers(self):
        """Initialize newly added layers"""
        for module in [self.temporal_attention, self.feature_processor, 
                      self.artifact_classifier, self.intensity_regressor, self.quality_scorer]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process each frame through EfficientNet backbone
        frame_features = []
        for i in range(seq_len):
            frame = x[:, i]  # (batch, channels, height, width)
            
            # Extract features using EfficientNet
            features = self.feature_extractor(frame)  # (batch, 1280, H', W')
            features = self.feature_processor(features)  # (batch, 256)
            frame_features.append(features)
        
        # Stack temporal features
        temporal_features = torch.stack(frame_features, dim=1)  # (batch, sequence_length, 256)
        
        # Apply temporal attention
        attended_features = self.temporal_attention(temporal_features)  # (batch, 256)
        
        # Multi-task outputs
        artifact_logits = self.artifact_classifier(attended_features)
        intensity_scores = torch.sigmoid(self.intensity_regressor(attended_features))
        quality_score = self.quality_scorer(attended_features)
        
        return {
            'artifact_logits': artifact_logits,
            'intensity_scores': intensity_scores,
            'quality_score': quality_score,
            'features': attended_features
        }

class VimeoArtifactDataset(Dataset):
    """
    Dataset for Vimeo septuplet data with artifact generation
    """
    
    def __init__(self, 
                 data_root: str,
                 split_file: str,
                 sequence_length: int = 7,
                 image_size: Tuple[int, int] = (224, 224),  # EfficientNet standard size
                 artifact_probability: float = 0.8,
                 max_artifacts_per_sequence: int = 3):
        
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.artifact_probability = artifact_probability
        self.max_artifacts_per_sequence = max_artifacts_per_sequence
        
        # Load sequence list
        with open(split_file, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines()]
        
        # Artifact generator
        self.artifact_generator = ArtifactGenerator()
        
        # Transform for EfficientNet (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Artifact application methods
        self.artifact_methods = {
            ArtifactType.COMPRESSION_BLOCKING: self.artifact_generator.add_compression_blocking,
            ArtifactType.COMPRESSION_RINGING: self.artifact_generator.add_compression_ringing,
            ArtifactType.MOTION_GHOSTING: self.artifact_generator.add_motion_ghosting,
            ArtifactType.MOTION_JUDDER: self.artifact_generator.add_motion_judder,
            ArtifactType.NOISE_GAUSSIAN: self.artifact_generator.add_gaussian_noise,
            ArtifactType.NOISE_IMPULSE: self.artifact_generator.add_impulse_noise,
            ArtifactType.BLUR_MOTION: self.artifact_generator.add_motion_blur,
            ArtifactType.BLUR_DEFOCUS: self.artifact_generator.add_defocus_blur,
            ArtifactType.COLOR_BANDING: self.artifact_generator.add_color_banding,
            ArtifactType.COLOR_SATURATION: self.artifact_generator.add_color_saturation_issues
        }
    
    def __len__(self):
        return len(self.sequences)
    
    def load_sequence(self, sequence_path: str) -> torch.Tensor:
        """Load a 7-frame sequence"""
        frames = []
        sequence_dir = os.path.join(self.data_root, 'sequences', sequence_path)
        
        for i in range(1, self.sequence_length + 1):
            frame_path = os.path.join(sequence_dir, f'im{i}.png')
            
            if not os.path.exists(frame_path):
                # Use previous frame if current frame is missing
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create dummy frame
                    dummy_frame = torch.zeros(3, *self.image_size)
                    frames.append(dummy_frame)
                continue
            
            try:
                image = Image.open(frame_path).convert('RGB')
                frame_tensor = self.transform(image)
                frames.append(frame_tensor)
            except:
                # Use previous frame on error
                if frames:
                    frames.append(frames[-1])
                else:
                    dummy_frame = torch.zeros(3, *self.image_size)
                    frames.append(dummy_frame)
        
        return torch.stack(frames)  # (sequence_length, channels, height, width)
    
    def apply_artifacts(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply artifacts to frames (before normalization)"""
        
        # Denormalize for artifact application
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        denorm_frames = frames * std + mean
        denorm_frames = torch.clamp(denorm_frames, 0, 1)
        
        # Initialize labels
        artifact_labels = torch.zeros(len(ArtifactType.get_all_types()))
        intensity_labels = torch.zeros(len(ArtifactType.get_all_types()))
        
        # Decide whether to apply artifacts
        if random.random() > self.artifact_probability:
            # No artifacts - return normalized frames
            quality_score = torch.tensor([0.9 + random.random() * 0.1])
            return frames, artifact_labels, intensity_labels, quality_score
        
        # Select random artifacts
        num_artifacts = random.randint(1, self.max_artifacts_per_sequence)
        selected_artifacts = random.sample(ArtifactType.get_all_types(), num_artifacts)
        
        processed_frames = denorm_frames.clone()
        total_intensity = 0.0
        
        for artifact_type in selected_artifacts:
            # Random intensity
            intensity = random.uniform(0.2, 0.8)
            
            # Apply artifact
            try:
                artifact_method = self.artifact_methods[artifact_type]
                processed_frames = artifact_method(processed_frames, intensity)
                
                # Update labels
                artifact_labels[artifact_type] = 1.0
                intensity_labels[artifact_type] = intensity
                total_intensity += intensity
            except:
                continue
        
        # Renormalize processed frames
        processed_frames = torch.clamp(processed_frames, 0, 1)
        processed_frames = (processed_frames - mean) / std
        
        # Calculate quality score
        if total_intensity > 0:
            avg_intensity = total_intensity / len(selected_artifacts)
            quality_score = torch.tensor([max(0.1, 1.0 - avg_intensity)])
        else:
            quality_score = torch.tensor([0.9])
        
        return processed_frames, artifact_labels, intensity_labels, quality_score
    
    def __getitem__(self, idx):
        sequence_path = self.sequences[idx]
        
        try:
            # Load clean sequence (already normalized)
            clean_frames = self.load_sequence(sequence_path)
            
            # Apply artifacts
            artifact_frames, artifact_labels, intensity_labels, quality_score = self.apply_artifacts(clean_frames)
            
            return {
                'frames': artifact_frames,
                'clean_frames': clean_frames,
                'artifact_labels': artifact_labels,
                'intensity_labels': intensity_labels,
                'quality_score': quality_score,
                'sequence_path': sequence_path
            }
            
        except Exception as e:
            # Return dummy sample
            dummy_frames = torch.zeros(self.sequence_length, 3, *self.image_size)
            dummy_labels = torch.zeros(len(ArtifactType.get_all_types()))
            dummy_quality = torch.tensor([0.5])
            
            return {
                'frames': dummy_frames,
                'clean_frames': dummy_frames,
                'artifact_labels': dummy_labels,
                'intensity_labels': dummy_labels,
                'quality_score': dummy_quality,
                'sequence_path': sequence_path
            }

class ArtifactDetectionLoss(nn.Module):
    """
    Multi-task loss function for artifact detection
    """
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 intensity_weight: float = 1.0,
                 quality_weight: float = 0.5):
        super().__init__()
        
        self.classification_weight = classification_weight
        self.intensity_weight = intensity_weight
        self.quality_weight = quality_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        # Unpack predictions
        artifact_logits = predictions['artifact_logits']
        intensity_scores = predictions['intensity_scores']
        quality_score = predictions['quality_score']
        
        # Unpack targets
        artifact_labels = targets['artifact_labels']
        intensity_labels = targets['intensity_labels']
        quality_labels = targets['quality_score']
        
        # Classification loss
        classification_loss = self.bce_loss(artifact_logits, artifact_labels)
        
        # Intensity regression loss (only for positive artifacts)
        mask = artifact_labels > 0
        if mask.sum() > 0:
            intensity_loss = self.mse_loss(
                intensity_scores[mask], 
                intensity_labels[mask]
            )
        else:
            intensity_loss = torch.tensor(0.0, device=artifact_logits.device)
        
        # Quality score loss
        quality_loss = self.mse_loss(quality_score.squeeze(), quality_labels.squeeze())
        
        # Total loss
        total_loss = (self.classification_weight * classification_loss + 
                     self.intensity_weight * intensity_loss + 
                     self.quality_weight * quality_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'intensity_loss': intensity_loss,
            'quality_loss': quality_loss
        }

def create_data_loaders(data_root: str, 
                       train_split: str,
                       test_split: str,
                       batch_size: int = 4,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders"""
    
    # Create datasets
    train_dataset = VimeoArtifactDataset(
        data_root=data_root,
        split_file=train_split,
        artifact_probability=0.8
    )
    
    test_dataset = VimeoArtifactDataset(
        data_root=data_root,
        split_file=test_split,
        artifact_probability=0.9
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    
    # Unpack predictions and targets
    artifact_logits = predictions['artifact_logits']
    intensity_scores = predictions['intensity_scores']
    quality_score = predictions['quality_score']
    
    artifact_labels = targets['artifact_labels']
    intensity_labels = targets['intensity_labels']
    quality_labels = targets['quality_score']
    
    # Convert logits to probabilities
    artifact_probs = torch.sigmoid(artifact_logits)
    artifact_preds = (artifact_probs > 0.5).float()
    
    # Classification metrics
    tp = (artifact_preds * artifact_labels).sum(dim=1)
    fp = (artifact_preds * (1 - artifact_labels)).sum(dim=1)
    fn = ((1 - artifact_preds) * artifact_labels).sum(dim=1)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Intensity MAE (only for positive artifacts)
    mask = artifact_labels > 0
    if mask.sum() > 0:
        intensity_mae = torch.abs(intensity_scores[mask] - intensity_labels[mask]).mean()
    else:
        intensity_mae = torch.tensor(0.0)
    
    # Quality score MAE
    quality_mae = torch.abs(quality_score.squeeze() - quality_labels.squeeze()).mean()
    
    return {
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1_score': f1.mean(),
        'intensity_mae': intensity_mae,
        'quality_mae': quality_mae
    }

if __name__ == "__main__":
    # Test the EfficientNet-based model
    print("EfficientNet-Based Video Artifact Detection Model")
    print("=" * 50)
    
    # Model parameters
    num_artifact_types = len(ArtifactType.get_all_types())
    sequence_length = 7
    
    # Create model
    model = EfficientNetArtifactDetector(
        num_artifact_types=num_artifact_types,
        sequence_length=sequence_length
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with dummy input (EfficientNet expects 224x224)
    dummy_input = torch.randn(2, sequence_length, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Artifact logits shape: {output['artifact_logits'].shape}")
    print(f"Intensity scores shape: {output['intensity_scores'].shape}")
    print(f"Quality score shape: {output['quality_score'].shape}")
    print(f"Features shape: {output['features'].shape}")
    
    print("\nEfficientNet-based model advantages:")
    print("- Pre-trained on ImageNet for better feature extraction")
    print("- Efficient architecture optimized for mobile/edge devices")
    print("- Better spatial feature representation")
    print("- Proven performance on image classification tasks")
    print("- Reduced training time with transfer learning")
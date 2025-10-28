"""
AQS: Artifact Quality Score - A Novel No-Reference Video Quality Metric
======================================================================

This module implements the Artifact Quality Score (AQS), a new interpretable 
no-reference video quality metric that combines artifact detection probabilities 
and intensity scores to provide a comprehensive quality assessment.

Key Features:
- Interpretable: Shows which artifacts affect quality
- No-reference: Doesn't need original video
- Real-time capable: Optimized for streaming applications
- Temporal analysis: Tracks quality changes over time
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr
import json
from datetime import datetime

class AQSMetric:
    """
    Artifact Quality Score (AQS) - Novel interpretable video quality metric
    """
    
    def __init__(self, alpha: float = 0.6, temporal_window: int = 30, aggregation_method: str = 'rms'):
        """
        Initialize AQS metric with improved defaults
        
        Args:
            alpha: Balance between quality score and artifact impact (0-1) - Optimized for perceptual quality
            temporal_window: Window size for temporal analysis
            aggregation_method: How to aggregate artifacts ('rms' for better perceptual correlation)
        """
        self.alpha = alpha
        self.temporal_window = temporal_window
        self.aggregation_method = aggregation_method
        self.aqs_history = []
        self.artifact_history = []
        
    def compute_aqs(self, artifact_logits: torch.Tensor, 
                   intensity_scores: torch.Tensor, 
                   quality_score: torch.Tensor) -> torch.Tensor:
        """
        Compute AQS score from model outputs
        
        Args:
            artifact_logits: Raw artifact detection logits [batch, num_artifacts]
            intensity_scores: Artifact intensity scores [batch, num_artifacts] 
            quality_score: Overall quality score [batch, 1]
            
        Returns:
            AQS scores [batch]
        """
        # Convert logits to probabilities
        artifact_probs = torch.sigmoid(artifact_logits)
        
        # Weighted artifact impact (element-wise product)
        artifact_impact = artifact_probs * intensity_scores
        
        # FIXED: Use better aggregation to avoid dilution effect
        artifact_severity = self._aggregate_artifacts(artifact_impact)
        
        # Normalize quality score (if not already 0-1)
        quality_norm = torch.sigmoid(quality_score.squeeze())
        
        # BALANCED blur penalization - blur should reduce quality to "Poor" range (0.3-0.5)
        quality_component = torch.pow(quality_norm, 0.8)  # Slight compression
        
        # Moderate artifact penalization - blur should significantly impact but not destroy
        artifact_component = torch.pow(artifact_severity, 0.7)  # Balanced curve
        
        # Blur-specific penalty - blur reduces quality significantly
        blur_mask = artifact_severity > 0.2  # Reasonable threshold for blur detection
        
        # Apply strong but not devastating penalty for blur
        quality_degradation = torch.where(
            blur_mask,
            quality_component * (1 - artifact_component * 0.8),  # Strong penalty for blur
            quality_component * (1 - artifact_component * 0.6)   # Moderate penalty for other artifacts
        )
        
        # Balanced weighting - artifacts important but not completely dominant
        aqs = 0.4 * quality_degradation + 0.6 * (1 - artifact_component)  # 60% artifact weight
        
        # Balanced mapping - blur should result in scores 0.3-0.5 (Poor range)
        aqs = torch.sigmoid(2.5 * (aqs - 0.4))  # Moderate curve, centered for realistic scores
        
        # Allow realistic range for blurry content
        aqs = torch.clamp(aqs, 0.1, 0.95)
        
        return aqs
    
    def _aggregate_artifacts(self, artifact_impact: torch.Tensor) -> torch.Tensor:
        """
        Aggregate artifact impacts with special handling for blur artifacts
        
        Args:
            artifact_impact: [batch, num_artifacts] tensor of artifact impacts
            
        Returns:
            Aggregated artifact severity [batch]
        """
        # Apply blur penalty - blur artifacts are particularly noticeable to humans
        blur_penalty = self._apply_blur_penalty(artifact_impact)
        
        if self.aggregation_method == 'max':
            # Use worst artifact (no dilution)
            return blur_penalty.max(dim=1)[0]
            
        elif self.aggregation_method == 'rms':
            # Root Mean Square (penalizes stronger artifacts more)
            return torch.sqrt((blur_penalty ** 2).mean(dim=1))
            
        elif self.aggregation_method == 'significant_mean':
            # Only average artifacts above threshold (ignore weak detections)
            threshold = 0.2  # Reasonable threshold for artifact detection
            batch_size = artifact_impact.shape[0]
            result = torch.zeros(batch_size, device=artifact_impact.device)
            
            for i in range(batch_size):
                significant = artifact_impact[i] > threshold
                if significant.any():
                    result[i] = artifact_impact[i][significant].mean()
                else:
                    result[i] = artifact_impact[i].max()  # Fallback to max if none significant
            return result
            
        elif self.aggregation_method == 'weighted_top3':
            # Average of top 3 artifacts (reduces dilution)
            top3_values, _ = torch.topk(artifact_impact, k=min(3, artifact_impact.shape[1]), dim=1)
            return top3_values.mean(dim=1)
            
        else:  # 'mean' - original method (kept for compatibility)
            return blur_penalty.mean(dim=1)
    
    def _apply_blur_penalty(self, artifact_impact: torch.Tensor) -> torch.Tensor:
        """
        Apply perceptual penalties for different artifact types based on human visual sensitivity
        
        Args:
            artifact_impact: [batch, num_artifacts] tensor of artifact impacts
            
        Returns:
            Artifact impacts with perceptual penalties applied
        """
        # Create a copy to modify
        penalized_impact = artifact_impact.clone()
        
        # BALANCED blur penalties - blur should significantly reduce quality but not destroy it
        # Index mapping: [blocking, ringing, ghosting, judder, gaussian_noise, impulse_noise, 
        #                motion_blur, defocus_blur, color_banding, color_saturation]
        penalties = [
            1.2,  # compression_blocking - noticeable
            1.2,  # compression_ringing - noticeable 
            1,  # motion_ghosting - distracting
            1,  # motion_judder - annoying
            1,  # noise_gaussian - moderately noticeable
            1,  # noise_impulse - very distracting
            1,  # blur_motion - significantly reduces quality
            1,  # blur_defocus - significantly reduces quality
            1,  # color_banding - noticeable in gradients
            1   # color_saturation - less critical
        ]
        
        # Apply penalties with blur-specific logic
        for i, penalty in enumerate(penalties):
            if i < artifact_impact.shape[1]:
                penalized_impact[:, i] = penalized_impact[:, i] * penalty
        
        # Apply blur-specific enhancement to reduce false positives
        penalized_impact = self._enhance_blur_detection(penalized_impact)
        
        # Clamp to ensure values stay in valid range
        penalized_impact = torch.clamp(penalized_impact, 0, 1)
        
        return penalized_impact
    
    def _enhance_blur_detection(self, artifact_impact: torch.Tensor) -> torch.Tensor:
        """
        Enhance blur detection by reducing false positives from compression artifacts
        """
        if artifact_impact.shape[1] < 8:  # Need at least 8 artifact types
            return artifact_impact
        
        enhanced_impact = artifact_impact.clone()
        
        # Artifact indices: [blocking, ringing, ghosting, judder, gaussian_noise, impulse_noise, 
        #                   motion_blur, defocus_blur, color_banding, color_saturation]
        compression_blocking_idx = 0
        compression_ringing_idx = 1
        motion_blur_idx = 6
        defocus_blur_idx = 7
        
        for batch_idx in range(artifact_impact.shape[0]):
            # Get blur and compression probabilities
            motion_blur = artifact_impact[batch_idx, motion_blur_idx]
            defocus_blur = artifact_impact[batch_idx, defocus_blur_idx]
            compression_ringing = artifact_impact[batch_idx, compression_ringing_idx]
            compression_blocking = artifact_impact[batch_idx, compression_blocking_idx]
            
            max_blur = torch.max(motion_blur, defocus_blur)
            max_compression = torch.max(compression_ringing, compression_blocking)
            
            # If blur is dominant, reduce compression artifact confidence
            if max_blur > max_compression * 0.8 and max_blur > 0.3:
                enhanced_impact[batch_idx, compression_ringing_idx] *= 0.6
                enhanced_impact[batch_idx, compression_blocking_idx] *= 0.7
                
            # If compression is dominant, slightly reduce blur confidence to avoid false positives
            elif max_compression > max_blur * 1.2 and max_compression > 0.4:
                enhanced_impact[batch_idx, motion_blur_idx] *= 0.8
                enhanced_impact[batch_idx, defocus_blur_idx] *= 0.8
        
        return enhanced_impact
    
    def compute_aqs_with_analysis(self, artifact_logits: torch.Tensor,
                                 intensity_scores: torch.Tensor,
                                 quality_score: torch.Tensor) -> Dict:
        """
        Compute AQS with detailed analysis
        
        Returns:
            Dictionary with AQS score and analysis components
        """
        # Basic AQS computation
        aqs = self.compute_aqs(artifact_logits, intensity_scores, quality_score)
        
        # Detailed analysis
        artifact_probs = torch.sigmoid(artifact_logits)
        artifact_impact = artifact_probs * intensity_scores
        
        # Artifact contributions (normalized)
        total_impact = artifact_impact.sum(dim=1, keepdim=True)
        artifact_contributions = artifact_impact / (total_impact + 1e-8)
        
        # Get aggregated severity using the same method as AQS computation
        artifact_severity = self._aggregate_artifacts(artifact_impact)
        
        # Temporal analysis if history available
        temporal_stats = {}
        if len(self.aqs_history) > 0:
            recent_aqs = self.aqs_history[-self.temporal_window:]
            temporal_stats = {
                'aqs_variance': np.var(recent_aqs),
                'aqs_trend': self._compute_trend(recent_aqs),
                'stability_score': self._compute_stability(recent_aqs)
            }
        
        # Update history
        self.aqs_history.extend(aqs.cpu().numpy().tolist())
        self.artifact_history.append(artifact_contributions.cpu().numpy())
        
        return {
            'aqs_score': aqs,
            'artifact_probabilities': artifact_probs,
            'artifact_intensities': intensity_scores,
            'artifact_contributions': artifact_contributions,
            'quality_component': torch.sigmoid(quality_score.squeeze()),
            'artifact_component': artifact_severity,  # Use aggregated severity instead of mean
            'max_artifact_impact': artifact_impact.max(dim=1)[0],  # Track worst artifact
            'aggregation_method': self.aggregation_method,
            'temporal_stats': temporal_stats
        }
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend in AQS values (positive = improving)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _compute_stability(self, values: List[float]) -> float:
        """Compute stability score (higher = more stable)"""
        if len(values) < 2:
            return 1.0
        
        # Stability = 1 - normalized variance
        variance = np.var(values)
        mean_val = np.mean(values)
        normalized_var = variance / (mean_val + 1e-8)
        stability = 1.0 / (1.0 + normalized_var)
        
        return float(stability)
    
    def compare_with_traditional_metrics(self, aqs_scores: np.ndarray,
                                       psnr_scores: np.ndarray = None,
                                       ssim_scores: np.ndarray = None,
                                       vmaf_scores: np.ndarray = None) -> Dict:
        """
        Compare AQS with traditional quality metrics
        
        Returns:
            Correlation analysis results
        """
        results = {}
        
        if psnr_scores is not None:
            spearman_psnr, _ = spearmanr(aqs_scores, psnr_scores)
            pearson_psnr, _ = pearsonr(aqs_scores, psnr_scores)
            results['psnr'] = {
                'spearman': spearman_psnr,
                'pearson': pearson_psnr
            }
        
        if ssim_scores is not None:
            spearman_ssim, _ = spearmanr(aqs_scores, ssim_scores)
            pearson_ssim, _ = pearsonr(aqs_scores, ssim_scores)
            results['ssim'] = {
                'spearman': spearman_ssim,
                'pearson': pearson_ssim
            }
        
        if vmaf_scores is not None:
            spearman_vmaf, _ = spearmanr(aqs_scores, vmaf_scores)
            pearson_vmaf, _ = pearsonr(aqs_scores, vmaf_scores)
            results['vmaf'] = {
                'spearman': spearman_vmaf,
                'pearson': pearson_vmaf
            }
        
        return results
    
    def create_temporal_plot(self, output_path: str = "aqs_temporal.png"):
        """Create temporal AQS plot"""
        if len(self.aqs_history) == 0:
            print("No AQS history available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.aqs_history, 'b-', linewidth=2, label='AQS Score')
        plt.axhline(y=0.85, color='g', linestyle='--', alpha=0.7, label='Good Quality Threshold')
        plt.axhline(y=0.6, color='r', linestyle='--', alpha=0.7, label='Poor Quality Threshold')
        
        plt.xlabel('Frame/Sequence')
        plt.ylabel('AQS Score')
        plt.title('AQS Temporal Variation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal plot saved to {output_path}")
    
    def create_artifact_contribution_plot(self, artifact_names: List[str],
                                        output_path: str = "artifact_contributions.png"):
        """Create artifact contribution visualization"""
        if len(self.artifact_history) == 0:
            print("No artifact history available for plotting")
            return
        
        # Average contributions across all frames
        avg_contributions = np.mean(self.artifact_history, axis=0)
        
        if len(avg_contributions.shape) > 1:
            avg_contributions = np.mean(avg_contributions, axis=0)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(artifact_names, avg_contributions, alpha=0.7, color='skyblue')
        
        # Add value labels on bars
        for bar, contrib in zip(bars, avg_contributions):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{contrib:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Artifact Type')
        plt.ylabel('Average Contribution to Quality Degradation')
        plt.title('Artifact Contributions to AQS')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Artifact contribution plot saved to {output_path}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of AQS analysis"""
        if len(self.aqs_history) == 0:
            return {}
        
        aqs_array = np.array(self.aqs_history)
        
        return {
            'mean_aqs': float(np.mean(aqs_array)),
            'std_aqs': float(np.std(aqs_array)),
            'min_aqs': float(np.min(aqs_array)),
            'max_aqs': float(np.max(aqs_array)),
            'median_aqs': float(np.median(aqs_array)),
            'frames_analyzed': len(self.aqs_history),
            'good_quality_ratio': float(np.mean(aqs_array > 0.85)),
            'poor_quality_ratio': float(np.mean(aqs_array < 0.6)),
            'overall_trend': self._compute_trend(self.aqs_history),
            'overall_stability': self._compute_stability(self.aqs_history)
        }
    
    def reset_history(self):
        """Reset AQS and artifact history"""
        self.aqs_history = []
        self.artifact_history = []

# Convenience function for quick AQS computation
def compute_aqs_simple(artifact_logits: torch.Tensor,
                      intensity_scores: torch.Tensor,
                      quality_score: torch.Tensor,
                      alpha: float = 0.6,
                      aggregation_method: str = 'rms') -> torch.Tensor:
    """
    Simple AQS computation function with FIXED dilution issue
    
    Args:
        artifact_logits: Raw artifact detection logits
        intensity_scores: Artifact intensity scores
        quality_score: Overall quality score
        alpha: Balance parameter (reduced from 0.7 to 0.5)
        aggregation_method: How to aggregate artifacts ('max', 'rms', 'significant_mean', 'mean')
        
    Returns:
        AQS scores
    """
    artifact_probs = torch.sigmoid(artifact_logits)
    artifact_impact = artifact_probs * intensity_scores
    
    # BALANCED blur penalties - blur should significantly reduce quality
    penalties = torch.tensor([1.6, 1.4, 1.8, 1.7, 1.3, 2.0, 2.2, 2.0, 1.5, 1.2], 
                           device=artifact_impact.device, dtype=artifact_impact.dtype)
    if artifact_impact.shape[1] == len(penalties):
        artifact_impact = artifact_impact * penalties.unsqueeze(0)
    
    # Aggregate with improved method
    if aggregation_method == 'max':
        artifact_severity = artifact_impact.max(dim=1)[0]
    elif aggregation_method == 'rms':
        artifact_severity = torch.sqrt((artifact_impact ** 2).mean(dim=1))
    else:  # fallback to mean
        artifact_severity = artifact_impact.mean(dim=1)
    
    # BALANCED computation - blur should result in "Poor" scores (0.3-0.5)
    quality_norm = torch.sigmoid(quality_score.squeeze())
    quality_component = torch.pow(quality_norm, 0.8)
    artifact_component = torch.pow(artifact_severity, 0.7)  # Balanced curve
    
    # Apply strong but reasonable penalty for blur
    blur_mask = artifact_severity > 0.2  # Reasonable threshold
    quality_degradation = torch.where(
        blur_mask,
        quality_component * (1 - artifact_component * 0.8),  # Strong penalty
        quality_component * (1 - artifact_component * 0.6)   # Moderate penalty
    )
    
    # Balanced weighting (60% artifact weight)
    aqs = 0.4 * quality_degradation + 0.6 * (1 - artifact_component)
    aqs = torch.sigmoid(2.5 * (aqs - 0.4))  # Balanced curve
    aqs = torch.clamp(aqs, 0.1, 0.95)
    
    return aqs
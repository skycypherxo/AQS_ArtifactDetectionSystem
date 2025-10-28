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
    
    def __init__(self, temporal_window: int = 30):
        """
        Initialize AQS metric
        
        Args:
            temporal_window: Window size for temporal analysis
        """
        self.temporal_window = temporal_window
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
        
        # Weighted artifact impact
        artifact_impact = artifact_probs * intensity_scores
        
        # CRITICAL FIX: Use MAX aggregation instead of RMS
        # This ensures the worst artifact dominates the score
        artifact_severity = artifact_impact.max(dim=1)[0]
        
        # CRITICAL FIX: Amplify artifact impact (square root makes it stronger)
        artifact_severity = torch.sqrt(artifact_severity)  # 0.36 → 0.60
        
        # Normalize quality score
        quality_norm = torch.sigmoid(quality_score.squeeze())
        
        # CRITICAL FIX: Increase artifact weight to 80%
        aqs = 0.2 * quality_norm + 0.8 * (1 - artifact_severity)
        
        return torch.clamp(aqs, 0, 1)
    
    def _aggregate_artifacts(self, artifact_impact: torch.Tensor) -> torch.Tensor:
        """
        Aggregate artifact impacts using MAX (worst artifact dominates)
        
        Args:
            artifact_impact: [batch, num_artifacts] tensor of artifact impacts
            
        Returns:
            Aggregated artifact severity [batch]
        """
        # Use max instead of RMS
        artifact_severity = artifact_impact.max(dim=1)[0]
        
        # Amplify to make artifacts more impactful
        return torch.sqrt(artifact_severity)
    

    
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
            'artifact_severity': artifact_severity,
            'max_artifact_impact': artifact_impact.max(dim=1)[0],  # Track worst artifact
            'aggregation_method': 'max',
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
    
    def get_quality_label(self, aqs_score: float) -> str:
        """Get quality label from AQS score"""
        if aqs_score >= 0.70:
            return "Excellent"
        elif aqs_score >= 0.50:
            return "Good"
        elif aqs_score >= 0.35:
            return "Fair"
        elif aqs_score >= 0.20:
            return "Poor"
        else:
            return "Bad"
    
    def reset_history(self):
        """Reset AQS and artifact history"""
        self.aqs_history = []
        self.artifact_history = []

# Convenience function for quick AQS computation
def compute_aqs_simple(artifact_logits: torch.Tensor,
                      intensity_scores: torch.Tensor,
                      quality_score: torch.Tensor) -> torch.Tensor:
    """
    Simple AQS computation function
    
    Args:
        artifact_logits: Raw artifact detection logits
        intensity_scores: Artifact intensity scores
        quality_score: Overall quality score
        
    Returns:
        AQS scores
    """
    artifact_probs = torch.sigmoid(artifact_logits)
    artifact_impact = artifact_probs * intensity_scores
    
    # Use MAX aggregation - worst artifact dominates
    artifact_severity = artifact_impact.max(dim=1)[0]
    
    # Amplify impact
    artifact_severity = torch.sqrt(artifact_severity)
    
    # Normalize quality
    quality_norm = torch.sigmoid(quality_score.squeeze())
    
    # 80% artifacts, 20% quality
    aqs = 0.2 * quality_norm + 0.8 * (1 - artifact_severity)
    
    return torch.clamp(aqs, 0, 1)
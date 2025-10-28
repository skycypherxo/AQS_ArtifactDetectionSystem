"""
AQS Research Framework
=====================

Comprehensive research framework for evaluating the AQS (Artifact Quality Score)
metric against traditional video quality metrics and generating publication-ready results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from realtime_aqs_analyzer import RealTimeAQSAnalyzer
from aqs_metric import AQSMetric
from efficientnet_artifact_detector import ArtifactType

class AQSResearchFramework:
    """
    Research framework for AQS metric evaluation and comparison
    """
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize research framework
        
        Args:
            model_path: Path to trained artifact detection model
            device: Device for computation
        """
        self.analyzer = RealTimeAQSAnalyzer(
            model_path=model_path,
            device=device,
            batch_size=8  # Larger batch for research
        )
        
        self.results_database = []
        self.comparison_metrics = {}
        
    def evaluate_video_dataset(self, video_paths: List[str], 
                              reference_metrics: Dict[str, List[float]] = None,
                              output_dir: str = "aqs_research_results") -> Dict:
        """
        Evaluate AQS on a dataset of videos
        
        Args:
            video_paths: List of video file paths
            reference_metrics: Dictionary of reference metrics (PSNR, SSIM, VMAF, etc.)
            output_dir: Output directory for results
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"Evaluating AQS on {len(video_paths)} videos...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_aqs_scores = []
        all_video_results = []
        
        for i, video_path in enumerate(video_paths):
            print(f"Processing video {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
            
            try:
                # Analyze video
                video_output_dir = os.path.join(output_dir, f"video_{i:03d}")
                results = self.analyzer.analyze_video_file(video_path, video_output_dir)
                
                # Extract AQS scores
                aqs_scores = [r['aqs_score'] for r in results]
                all_aqs_scores.extend(aqs_scores)
                
                # Video-level statistics
                video_stats = {
                    'video_path': video_path,
                    'video_name': os.path.basename(video_path),
                    'num_sequences': len(results),
                    'mean_aqs': np.mean(aqs_scores),
                    'std_aqs': np.std(aqs_scores),
                    'min_aqs': np.min(aqs_scores),
                    'max_aqs': np.max(aqs_scores),
                    'aqs_scores': aqs_scores,
                    'detailed_results': results
                }
                
                all_video_results.append(video_stats)
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        # Overall dataset statistics
        dataset_stats = {
            'total_videos': len(all_video_results),
            'total_sequences': len(all_aqs_scores),
            'overall_mean_aqs': np.mean(all_aqs_scores),
            'overall_std_aqs': np.std(all_aqs_scores),
            'overall_min_aqs': np.min(all_aqs_scores),
            'overall_max_aqs': np.max(all_aqs_scores),
            'video_results': all_video_results
        }
        
        # Compare with reference metrics if provided
        correlation_results = {}
        if reference_metrics:
            correlation_results = self._compare_with_reference_metrics(
                all_aqs_scores, reference_metrics, output_dir
            )
        
        # Generate comprehensive report
        evaluation_results = {
            'dataset_statistics': dataset_stats,
            'correlation_analysis': correlation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_file = os.path.join(output_dir, 'aqs_evaluation_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_list(evaluation_results)
            json.dump(json_results, f, indent=2)
        
        # Create visualizations
        self._create_research_visualizations(evaluation_results, output_dir)
        
        # Generate research report
        self._generate_research_report(evaluation_results, output_dir)
        
        print(f"Evaluation complete. Results saved to {output_dir}")
        return evaluation_results
    
    def _compare_with_reference_metrics(self, aqs_scores: List[float],
                                      reference_metrics: Dict[str, List[float]],
                                      output_dir: str) -> Dict:
        """Compare AQS with reference quality metrics"""
        print("Comparing AQS with reference metrics...")
        
        correlation_results = {}
        
        # Ensure all metrics have same length as AQS scores
        min_length = min(len(aqs_scores), 
                        min(len(scores) for scores in reference_metrics.values()))
        
        aqs_array = np.array(aqs_scores[:min_length])
        
        for metric_name, metric_scores in reference_metrics.items():
            metric_array = np.array(metric_scores[:min_length])
            
            # Calculate correlations
            spearman_corr, spearman_p = spearmanr(aqs_array, metric_array)
            pearson_corr, pearson_p = pearsonr(aqs_array, metric_array)
            
            # Calculate error metrics
            mse = mean_squared_error(aqs_array, metric_array)
            mae = mean_absolute_error(aqs_array, metric_array)
            
            correlation_results[metric_name] = {
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'mse': float(mse),
                'mae': float(mae),
                'num_samples': min_length
            }
            
            print(f"{metric_name} - Spearman: {spearman_corr:.3f}, Pearson: {pearson_corr:.3f}")
        
        # Create correlation matrix visualization
        self._create_correlation_matrix(aqs_array, reference_metrics, output_dir)
        
        return correlation_results
    
    def _create_correlation_matrix(self, aqs_scores: np.ndarray,
                                 reference_metrics: Dict[str, List[float]],
                                 output_dir: str):
        """Create correlation matrix visualization"""
        # Prepare data for correlation matrix
        data_dict = {'AQS': aqs_scores}
        
        min_length = len(aqs_scores)
        for metric_name, scores in reference_metrics.items():
            data_dict[metric_name] = np.array(scores[:min_length])
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('AQS vs Traditional Metrics - Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plots
        n_metrics = len(reference_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric_name, scores) in enumerate(reference_metrics.items()):
            metric_array = np.array(scores[:min_length])
            
            axes[i].scatter(aqs_scores, metric_array, alpha=0.6, s=20)
            axes[i].set_xlabel('AQS Score')
            axes[i].set_ylabel(f'{metric_name} Score')
            axes[i].set_title(f'AQS vs {metric_name}')
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(aqs_scores, metric_array, 1)
            p = np.poly1d(z)
            axes[i].plot(aqs_scores, p(aqs_scores), "r--", alpha=0.8)
            
            # Add correlation coefficient
            corr, _ = pearsonr(aqs_scores, metric_array)
            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', 
                        transform=axes[i].transAxes, 
                        bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aqs_vs_traditional_metrics.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_research_visualizations(self, results: Dict, output_dir: str):
        """Create comprehensive research visualizations"""
        dataset_stats = results['dataset_statistics']
        video_results = dataset_stats['video_results']
        
        # 1. AQS Distribution across videos
        plt.figure(figsize=(12, 8))
        
        # Overall distribution
        plt.subplot(2, 2, 1)
        all_aqs = []
        for video in video_results:
            all_aqs.extend(video['aqs_scores'])
        
        plt.hist(all_aqs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(all_aqs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_aqs):.3f}')
        plt.xlabel('AQS Score')
        plt.ylabel('Frequency')
        plt.title('Overall AQS Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Per-video mean AQS
        plt.subplot(2, 2, 2)
        video_means = [video['mean_aqs'] for video in video_results]
        video_names = [video['video_name'][:15] + '...' if len(video['video_name']) > 15 
                      else video['video_name'] for video in video_results]
        
        bars = plt.bar(range(len(video_means)), video_means, alpha=0.7, color='lightcoral')
        plt.xlabel('Video')
        plt.ylabel('Mean AQS')
        plt.title('Mean AQS per Video')
        plt.xticks(range(len(video_names)), video_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # AQS variance per video
        plt.subplot(2, 2, 3)
        video_stds = [video['std_aqs'] for video in video_results]
        plt.bar(range(len(video_stds)), video_stds, alpha=0.7, color='lightgreen')
        plt.xlabel('Video')
        plt.ylabel('AQS Standard Deviation')
        plt.title('AQS Variability per Video')
        plt.xticks(range(len(video_names)), video_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Quality categories
        plt.subplot(2, 2, 4)
        good_quality = sum(1 for score in all_aqs if score > 0.7)
        medium_quality = sum(1 for score in all_aqs if 0.5 <= score <= 0.7)
        poor_quality = sum(1 for score in all_aqs if score < 0.5)
        
        categories = ['Good\n(>0.7)', 'Medium\n(0.5-0.7)', 'Poor\n(<0.5)']
        counts = [good_quality, medium_quality, poor_quality]
        colors = ['green', 'orange', 'red']
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.7)
        plt.ylabel('Number of Sequences')
        plt.title('Quality Category Distribution')
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = count / total * 100
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + total*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aqs_research_overview.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal analysis for selected videos
        self._create_temporal_analysis_plots(video_results, output_dir)
        
        # 3. Artifact analysis
        self._create_artifact_analysis_plots(video_results, output_dir)
    
    def _create_temporal_analysis_plots(self, video_results: List[Dict], output_dir: str):
        """Create temporal analysis plots"""
        # Select up to 4 videos for temporal analysis
        selected_videos = video_results[:min(4, len(video_results))]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, video in enumerate(selected_videos):
            if i >= 4:
                break
                
            aqs_scores = video['aqs_scores']
            
            axes[i].plot(aqs_scores, 'b-', linewidth=2, alpha=0.8)
            axes[i].axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='Good Quality')
            axes[i].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Poor Quality')
            axes[i].fill_between(range(len(aqs_scores)), aqs_scores, alpha=0.3)
            
            axes[i].set_xlabel('Sequence Number')
            axes[i].set_ylabel('AQS Score')
            axes[i].set_title(f'{video["video_name"][:20]}...' if len(video["video_name"]) > 20 
                             else video["video_name"])
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
            
            if i == 0:
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(selected_videos), 4):
            axes[i].axis('off')
        
        plt.suptitle('AQS Temporal Analysis - Selected Videos', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aqs_temporal_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_artifact_analysis_plots(self, video_results: List[Dict], output_dir: str):
        """Create artifact analysis plots"""
        # Aggregate artifact statistics across all videos
        artifact_names = [ArtifactType.get_type_name(t) for t in ArtifactType.get_all_types()]
        artifact_counts = {name: 0 for name in artifact_names}
        artifact_intensities = {name: [] for name in artifact_names}
        
        total_sequences = 0
        
        for video in video_results:
            for result in video['detailed_results']:
                total_sequences += 1
                for artifact_name, artifact_data in result['artifacts'].items():
                    artifact_counts[artifact_name] += 1
                    artifact_intensities[artifact_name].append(artifact_data['intensity'])
        
        # Create artifact analysis plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Artifact occurrence frequency
        names = list(artifact_counts.keys())
        counts = list(artifact_counts.values())
        percentages = [count / total_sequences * 100 for count in counts]
        
        bars1 = ax1.bar(names, percentages, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Artifact Type')
        ax1.set_ylabel('Occurrence Percentage (%)')
        ax1.set_title('Artifact Detection Frequency')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, percentage in zip(bars1, percentages):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Average artifact intensity
        avg_intensities = []
        for name in names:
            if artifact_intensities[name]:
                avg_intensities.append(np.mean(artifact_intensities[name]))
            else:
                avg_intensities.append(0)
        
        bars2 = ax2.bar(names, avg_intensities, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Artifact Type')
        ax2.set_ylabel('Average Intensity')
        ax2.set_title('Average Artifact Intensity')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, intensity in zip(bars2, avg_intensities):
            if intensity > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{intensity:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'artifact_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_research_report(self, results: Dict, output_dir: str):
        """Generate comprehensive research report"""
        report_path = os.path.join(output_dir, 'aqs_research_report.md')
        
        dataset_stats = results['dataset_statistics']
        correlation_results = results['correlation_analysis']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# AQS (Artifact Quality Score) Research Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents a comprehensive evaluation of the AQS (Artifact Quality Score) ")
            f.write("metric, a novel no-reference video quality assessment method based on artifact ")
            f.write("detection and intensity analysis.\n\n")
            
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Videos Analyzed:** {dataset_stats['total_videos']}\n")
            f.write(f"- **Total Sequences:** {dataset_stats['total_sequences']}\n")
            f.write(f"- **Overall Mean AQS:** {dataset_stats['overall_mean_aqs']:.3f}\n")
            f.write(f"- **AQS Standard Deviation:** {dataset_stats['overall_std_aqs']:.3f}\n")
            f.write(f"- **AQS Range:** {dataset_stats['overall_min_aqs']:.3f} - {dataset_stats['overall_max_aqs']:.3f}\n\n")
            
            if correlation_results:
                f.write("## Correlation Analysis with Traditional Metrics\n\n")
                f.write("| Metric | Spearman Correlation | Pearson Correlation | P-value | MSE | MAE |\n")
                f.write("|--------|---------------------|--------------------|---------|----|-----|\n")
                
                for metric_name, stats in correlation_results.items():
                    f.write(f"| {metric_name} | {stats['spearman_correlation']:.3f} | ")
                    f.write(f"{stats['pearson_correlation']:.3f} | {stats['spearman_p_value']:.3e} | ")
                    f.write(f"{stats['mse']:.3f} | {stats['mae']:.3f} |\n")
                f.write("\n")
            
            f.write("## Key Findings\n\n")
            
            # Quality distribution analysis
            all_aqs = []
            for video in dataset_stats['video_results']:
                all_aqs.extend(video['aqs_scores'])
            
            good_quality_ratio = sum(1 for score in all_aqs if score > 0.7) / len(all_aqs)
            poor_quality_ratio = sum(1 for score in all_aqs if score < 0.5) / len(all_aqs)
            
            f.write(f"- **Good Quality Content:** {good_quality_ratio:.1%} of sequences have AQS > 0.7\n")
            f.write(f"- **Poor Quality Content:** {poor_quality_ratio:.1%} of sequences have AQS < 0.5\n")
            
            if correlation_results:
                # Find best correlating metric
                best_metric = max(correlation_results.items(), 
                                key=lambda x: abs(x[1]['spearman_correlation']))
                f.write(f"- **Best Correlation:** AQS shows strongest correlation with {best_metric[0]} ")
                f.write(f"(ρ = {best_metric[1]['spearman_correlation']:.3f})\n")
            
            f.write("\n## Research Implications\n\n")
            f.write("### Advantages of AQS\n")
            f.write("1. **Interpretability:** AQS provides insight into which artifacts affect quality\n")
            f.write("2. **No-Reference:** Does not require original video for comparison\n")
            f.write("3. **Real-time Capable:** Can be computed in real-time for streaming applications\n")
            f.write("4. **Multi-dimensional:** Considers multiple artifact types simultaneously\n\n")
            
            f.write("### Applications\n")
            f.write("- Video streaming quality monitoring\n")
            f.write("- Content creation quality assurance\n")
            f.write("- Adaptive bitrate control\n")
            f.write("- Video compression optimization\n\n")
            
            f.write("## Methodology\n\n")
            f.write("AQS is computed as a weighted combination of:\n")
            f.write("1. **Artifact Detection:** Probability of each artifact type being present\n")
            f.write("2. **Intensity Assessment:** Severity of detected artifacts\n")
            f.write("3. **Quality Prediction:** Overall quality score from the model\n\n")
            f.write("Formula: `AQS = alpha × quality_score + (1-alpha) × (1 - artifact_impact)`\n\n")
            
            f.write("## Future Work\n\n")
            f.write("- Validation on larger datasets\n")
            f.write("- Comparison with human perceptual studies\n")
            f.write("- Optimization for specific video content types\n")
            f.write("- Integration with adaptive streaming systems\n")
        
        print(f"Research report saved to {report_path}")
    
    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

def main():
    """Main function for research evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AQS Research Framework')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained artifact detection model')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing test videos')
    parser.add_argument('--output_dir', type=str, default='aqs_research_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    
    args = parser.parse_args()
    
    # Get video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_paths = []
    
    for file in os.listdir(args.video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_paths.append(os.path.join(args.video_dir, file))
    
    if not video_paths:
        print(f"No video files found in {args.video_dir}")
        return
    
    print(f"Found {len(video_paths)} video files")
    
    # Create research framework
    framework = AQSResearchFramework(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run evaluation
    results = framework.evaluate_video_dataset(
        video_paths=video_paths,
        output_dir=args.output_dir
    )
    
    print("Research evaluation completed!")

if __name__ == "__main__":
    main()
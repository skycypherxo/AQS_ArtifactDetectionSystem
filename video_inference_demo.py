"""
Video Inference Demo
===================

A demo where you can input your own video file and get artifact detection results.
The model will analyze the video frame by frame and provide detailed artifact detection.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import argparse
from typing import List, Dict, Tuple
import json
from datetime import datetime

from efficientnet_artifact_detector import EfficientNetArtifactDetector, ArtifactType
import torchvision.transforms as transforms

class VideoArtifactAnalyzer:
    """Analyzes videos for artifacts using the trained model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        # Analysis parameters - define first
        self.sequence_length = 7
        self.overlap = 3  # Overlap between sequences for smoother analysis
        
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> EfficientNetArtifactDetector:
        """Load the trained model"""
        model = EfficientNetArtifactDetector(
            num_artifact_types=len(ArtifactType.get_all_types()),
            sequence_length=self.sequence_length
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """Get the preprocessing transform"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    def preprocess_sequence(self, frames: List[np.ndarray], start_idx: int) -> torch.Tensor:
        """Preprocess a sequence of frames"""
        sequence_frames = []
        
        for i in range(self.sequence_length):
            frame_idx = min(start_idx + i, len(frames) - 1)
            frame = frames[frame_idx]
            
            # Apply transform
            frame_tensor = self.transform(frame)
            sequence_frames.append(frame_tensor)
        
        return torch.stack(sequence_frames).unsqueeze(0)  # Add batch dimension
    
    def analyze_video(self, video_path: str, output_dir: str = "analysis_results") -> Dict:
        """Analyze entire video for artifacts"""
        print(f"Analyzing video: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames
        frames = self.extract_frames(video_path, max_frames=200)  # Limit for demo
        
        if len(frames) < self.sequence_length:
            raise ValueError(f"Video too short. Need at least {self.sequence_length} frames.")
        
        # Analyze sequences
        results = []
        step_size = self.sequence_length - self.overlap
        
        for start_idx in range(0, len(frames) - self.sequence_length + 1, step_size):
            # Preprocess sequence
            sequence_tensor = self.preprocess_sequence(frames, start_idx)
            sequence_tensor = sequence_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(sequence_tensor)
            
            # Process results
            artifact_probs = torch.sigmoid(output['artifact_logits']).cpu().numpy()[0]
            intensity_scores = output['intensity_scores'].cpu().numpy()[0]
            quality_score = output['quality_score'].cpu().numpy()[0][0]
            
            # Store results
            sequence_result = {
                'start_frame': start_idx,
                'end_frame': start_idx + self.sequence_length - 1,
                'quality_score': float(quality_score),
                'artifacts': {}
            }
            
            # Process each artifact type
            for i, artifact_type in enumerate(ArtifactType.get_all_types()):
                artifact_name = ArtifactType.get_type_name(artifact_type)
                prob = float(artifact_probs[i])
                intensity = float(intensity_scores[i])
                impact = prob * intensity  # ACTUAL IMPACT (probability × intensity)
                
                if prob > 0.4:  # Threshold for detection (balanced for new aggregation)
                    sequence_result['artifacts'][artifact_name] = {
                        'probability': prob,
                        'intensity': intensity,
                        'impact': impact  # Add the actual impact value
                    }
            
            results.append(sequence_result)
            
            # Progress update
            progress = (start_idx + self.sequence_length) / len(frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')
        
        print("\nAnalysis complete!")
        
        # Generate summary
        summary = self._generate_summary(results, frames, output_dir)
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'detailed_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'video_path': video_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_frames': len(frames),
                'sequences_analyzed': len(results),
                'summary': summary,
                'detailed_results': results
            }, f, indent=2)
        
        return summary
    
    def _generate_summary(self, results: List[Dict], frames: List[np.ndarray], output_dir: str) -> Dict:
        """Generate analysis summary and visualizations"""
        
        # Calculate overall statistics
        total_sequences = len(results)
        avg_quality = np.mean([r['quality_score'] for r in results])
        
        # Count artifact occurrences
        artifact_counts = {}
        artifact_intensities = {}
        
        for result in results:
            for artifact_name, artifact_data in result['artifacts'].items():
                if artifact_name not in artifact_counts:
                    artifact_counts[artifact_name] = 0
                    artifact_intensities[artifact_name] = []
                
                artifact_counts[artifact_name] += 1
                artifact_intensities[artifact_name].append(artifact_data['intensity'])
        
        # Create visualizations
        self._create_quality_timeline(results, output_dir)
        self._create_artifact_summary_chart(artifact_counts, artifact_intensities, output_dir)
        self._create_sample_frames_with_annotations(results, frames, output_dir)
        
        # Generate summary
        summary = {
            'overall_quality_score': float(avg_quality),
            'total_sequences_analyzed': total_sequences,
            'artifacts_detected': len(artifact_counts),
            'artifact_summary': {}
        }
        
        for artifact_name, count in artifact_counts.items():
            avg_intensity = np.mean(artifact_intensities[artifact_name])
            summary['artifact_summary'][artifact_name] = {
                'occurrences': count,
                'percentage': (count / total_sequences) * 100,
                'average_intensity': float(avg_intensity)
            }
        
        return summary
    
    def _create_quality_timeline(self, results: List[Dict], output_dir: str):
        """Create quality score timeline"""
        plt.figure(figsize=(12, 6))
        
        frame_numbers = [r['start_frame'] for r in results]
        quality_scores = [r['quality_score'] for r in results]
        
        plt.plot(frame_numbers, quality_scores, 'b-', linewidth=2, label='Quality Score')
        plt.axhline(y=0.85, color='g', linestyle='--', alpha=0.7, label='Good Quality Threshold')
        plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Poor Quality Threshold')
        
        plt.xlabel('Frame Number')
        plt.ylabel('Quality Score')
        plt.title('Video Quality Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_timeline.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_artifact_summary_chart(self, artifact_counts: Dict, artifact_intensities: Dict, output_dir: str):
        """Create artifact summary chart"""
        if not artifact_counts:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Occurrence chart
        artifacts = list(artifact_counts.keys())
        counts = list(artifact_counts.values())
        
        bars1 = ax1.bar(artifacts, counts, color='skyblue', alpha=0.7)
        ax1.set_title('Artifact Occurrences')
        ax1.set_ylabel('Number of Sequences')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Intensity chart
        avg_intensities = [np.mean(artifact_intensities[artifact]) for artifact in artifacts]
        bars2 = ax2.bar(artifacts, avg_intensities, color='lightcoral', alpha=0.7)
        ax2.set_title('Average Artifact Intensity')
        ax2.set_ylabel('Intensity Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, intensity in zip(bars2, avg_intensities):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{intensity:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'artifact_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_sample_frames_with_annotations(self, results: List[Dict], frames: List[np.ndarray], output_dir: str):
        """Create sample frames with artifact annotations"""
        
        # Find interesting frames (with artifacts)
        interesting_results = [r for r in results if r['artifacts']]
        
        if not interesting_results:
            return
        
        # Select up to 6 sample frames
        sample_count = min(6, len(interesting_results))
        sample_indices = np.linspace(0, len(interesting_results)-1, sample_count, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            result = interesting_results[idx]
            frame_idx = result['start_frame'] + 3  # Middle frame of sequence
            frame = frames[frame_idx]
            
            ax = axes[i]
            ax.imshow(frame)
            ax.set_title(f'Frame {frame_idx}\nQuality: {result["quality_score"]:.2f}')
            ax.axis('off')
            
            # Add artifact annotations
            y_pos = 0.95
            for artifact_name, artifact_data in result['artifacts'].items():
                prob = artifact_data['probability']
                intensity = artifact_data['intensity']
                impact = artifact_data['impact']
                # FIXED: Show actual impact instead of just intensity
                text = f'{artifact_name}: {impact:.3f} (p={prob:.2f}, i={intensity:.2f})'
                ax.text(0.02, y_pos, text, transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                       fontsize=8, verticalalignment='top')
                y_pos -= 0.12
        
        # Hide unused subplots
        for i in range(sample_count, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_frames_annotated.png'), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Video Artifact Detection Demo')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--model_path', default='checkpoints/best_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--output_dir', default='video_analysis_results', 
                       help='Output directory for results')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found!")
        return
    
    try:
        # Create analyzer
        analyzer = VideoArtifactAnalyzer(args.model_path, args.device)
        
        # Analyze video
        summary = analyzer.analyze_video(args.video_path, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("VIDEO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Overall Quality Score: {summary['overall_quality_score']:.3f}")
        print(f"Sequences Analyzed: {summary['total_sequences_analyzed']}")
        print(f"Artifacts Detected: {summary['artifacts_detected']}")
        
        if summary['artifact_summary']:
            print("\nDetected Artifacts:")
            print("-" * 40)
            for artifact_name, data in summary['artifact_summary'].items():
                print(f"{artifact_name}:")
                print(f"  Occurrences: {data['occurrences']} ({data['percentage']:.1f}%)")
                print(f"  Avg Intensity: {data['average_intensity']:.3f}")
        else:
            print("\nNo significant artifacts detected!")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        print("Generated files:")
        print("- detailed_results.json (complete analysis data)")
        print("- quality_timeline.png (quality over time)")
        print("- artifact_summary.png (artifact statistics)")
        print("- sample_frames_annotated.png (example frames with detections)")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
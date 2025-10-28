"""
Real-Time AQS Video Quality Analyzer
===================================

Real-time video quality analysis using EfficientNet artifact detection
and the novel AQS (Artifact Quality Score) metric.

Optimizations for real-time performance:
- Sliding window processing
- GPU acceleration
- Batch processing
- Efficient memory management
"""

import torch
import cv2
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from efficientnet_artifact_detector import EfficientNetArtifactDetector, ArtifactType
from aqs_metric import AQSMetric, compute_aqs_simple
import torchvision.transforms as transforms

class RealTimeAQSAnalyzer:
    """
    Real-time video quality analyzer with AQS metric
    """
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'auto',
                 sequence_length: int = 7,
                 overlap: int = 3,
                 batch_size: int = 4,
                 buffer_size: int = 30):
        """
        Initialize real-time analyzer
        
        Args:
            model_path: Path to trained model
            device: Device for inference ('auto', 'cpu', 'cuda')
            sequence_length: Number of frames per sequence
            overlap: Overlap between sequences for smoothing
            batch_size: Batch size for processing multiple sequences
            buffer_size: Frame buffer size
        """
        self.device = self._get_device(device)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize AQS metric
        self.aqs_metric = AQSMetric(alpha=0.7, temporal_window=30)
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Frame buffer and processing queue
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue()
        
        # Processing state
        self.processing = False
        self.processing_thread = None
        
        # Performance metrics
        self.fps_counter = 0
        self.processing_times = []
        self.last_fps_time = time.time()
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> EfficientNetArtifactDetector:
        """Load the trained model"""
        model = EfficientNetArtifactDetector(
            num_artifact_types=len(ArtifactType.get_all_types()),
            sequence_length=self.sequence_length
        )
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: Using untrained model")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess single frame"""
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Apply transform
        frame_tensor = self.transform(frame_rgb)
        return frame_tensor
    
    def process_sequence_batch(self, sequences: List[torch.Tensor]) -> List[Dict]:
        """Process batch of sequences efficiently"""
        if not sequences:
            return []
        
        # Stack sequences into batch
        batch_tensor = torch.stack(sequences).to(self.device)
        
        start_time = time.perf_counter()
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        # Compute AQS for batch
        aqs_scores = compute_aqs_simple(
            outputs['artifact_logits'],
            outputs['intensity_scores'], 
            outputs['quality_score']
        )
        
        processing_time = time.perf_counter() - start_time
        self.processing_times.append(processing_time)
        
        # Process results
        results = []
        for i in range(len(sequences)):
            # Get individual results
            artifact_probs = torch.sigmoid(outputs['artifact_logits'][i]).cpu().numpy()
            intensity_scores = outputs['intensity_scores'][i].cpu().numpy()
            quality_score = outputs['quality_score'][i].cpu().numpy()[0]
            aqs_score = aqs_scores[i].cpu().numpy()
            
            # Detailed AQS analysis
            aqs_analysis = self.aqs_metric.compute_aqs_with_analysis(
                outputs['artifact_logits'][i:i+1],
                outputs['intensity_scores'][i:i+1],
                outputs['quality_score'][i:i+1]
            )
            
            # Create result dictionary
            result = {
                'timestamp': time.time(),
                'aqs_score': float(aqs_score),
                'quality_score': float(quality_score),
                'processing_time_ms': processing_time * 1000 / len(sequences),
                'artifacts': {},
                'aqs_analysis': {
                    'quality_component': float(aqs_analysis['quality_component'].item() if aqs_analysis['quality_component'].dim() == 0 else aqs_analysis['quality_component'][0]),
                    'artifact_component': float(aqs_analysis['artifact_component'].item() if aqs_analysis['artifact_component'].dim() == 0 else aqs_analysis['artifact_component'][0]),
                    'temporal_stats': aqs_analysis['temporal_stats']
                }
            }
            
            # Add artifact details with blur-specific detection logic
            detected_artifacts = {}
            
            for j, artifact_type in enumerate(ArtifactType.get_all_types()):
                artifact_name = ArtifactType.get_type_name(artifact_type)
                prob = float(artifact_probs[j])
                intensity = float(intensity_scores[j])
                
                if prob > 0.2:  # Reasonable threshold for artifact detection
                    detected_artifacts[artifact_name] = {
                        'probability': prob,
                        'intensity': intensity
                    }
            
            # Apply blur-specific detection logic with image processing validation
            # Use the middle frame of the sequence for blur validation
            middle_frame_idx = len(sequences[0]) // 2
            middle_frame = sequences[0][middle_frame_idx].permute(1, 2, 0).cpu().numpy()
            middle_frame = (middle_frame * 255).astype(np.uint8)
            
            blur_characteristics = self._detect_blur_characteristics(middle_frame)
            result['artifacts'] = self._refine_blur_detection(detected_artifacts, blur_characteristics)
            
            results.append(result)
        
        return results
    
    def _refine_blur_detection(self, detected_artifacts: Dict, blur_characteristics: Dict = None) -> Dict:
        """
        Refine artifact detection to prioritize blur over similar artifacts
        and reduce false positives using image processing validation
        """
        refined_artifacts = {}
        
        # Check if blur artifacts are detected
        blur_motion_prob = detected_artifacts.get('blur_motion', {}).get('probability', 0)
        blur_defocus_prob = detected_artifacts.get('blur_defocus', {}).get('probability', 0)
        compression_ringing_prob = detected_artifacts.get('compression_ringing', {}).get('probability', 0)
        compression_blocking_prob = detected_artifacts.get('compression_blocking', {}).get('probability', 0)
        
        # Use image processing validation if available
        is_actually_blurry = False
        blur_validation_score = 0.0
        
        if blur_characteristics:
            is_actually_blurry = blur_characteristics['is_blurry']
            blur_validation_score = blur_characteristics['blur_probability']
        
        # If both blur and compression artifacts are detected, use validation to decide
        max_blur_prob = max(blur_motion_prob, blur_defocus_prob)
        max_compression_prob = max(compression_ringing_prob, compression_blocking_prob)
        
        for artifact_name, artifact_data in detected_artifacts.items():
            prob = artifact_data['probability']
            intensity = artifact_data['intensity']
            
            # Apply blur-specific logic with image processing validation
            if artifact_name in ['blur_motion', 'blur_defocus']:
                # If image processing confirms blur, boost blur detection
                if is_actually_blurry and blur_validation_score > 0.6:
                    # Boost blur probability if validated by image processing
                    boosted_prob = min(prob * 1.3, 1.0)
                    refined_artifacts[artifact_name] = {
                        'probability': boosted_prob,
                        'intensity': intensity
                    }
                elif prob > 0.3:
                    # High confidence blur detection
                    refined_artifacts[artifact_name] = artifact_data
                elif max_blur_prob > max_compression_prob * 0.8:
                    # Blur is stronger than compression
                    refined_artifacts[artifact_name] = artifact_data
                    
            elif artifact_name in ['compression_ringing', 'compression_blocking']:
                # If image processing confirms blur, reduce compression false positives
                if is_actually_blurry and blur_validation_score > 0.6:
                    # Significantly reduce compression confidence when blur is validated
                    adjusted_prob = prob * 0.4
                    if adjusted_prob > 0.2:
                        refined_artifacts[artifact_name] = {
                            'probability': adjusted_prob,
                            'intensity': intensity * 0.5
                        }
                elif max_blur_prob > prob * 1.2:
                    # Reduce compression when blur is dominant
                    adjusted_prob = prob * 0.7
                    if adjusted_prob > 0.2:
                        refined_artifacts[artifact_name] = {
                            'probability': adjusted_prob,
                            'intensity': intensity * 0.8
                        }
                else:
                    # Keep compression artifact if it's clearly dominant
                    refined_artifacts[artifact_name] = artifact_data
                    
            else:
                # Keep other artifacts as-is
                refined_artifacts[artifact_name] = artifact_data
        
        return refined_artifacts
    
    def _detect_blur_characteristics(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Simple blur detection using image processing techniques
        to validate model predictions
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Laplacian variance for blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # Normalize scores (higher = sharper, lower = blurrier)
        laplacian_score = min(laplacian_var / 100.0, 1.0)  # Normalize to 0-1
        sobel_score = min(sobel_magnitude / 50.0, 1.0)     # Normalize to 0-1
        
        # Blur probability (inverse of sharpness)
        blur_probability = 1.0 - (laplacian_score * 0.6 + sobel_score * 0.4)
        
        return {
            'blur_probability': blur_probability,
            'laplacian_variance': laplacian_var,
            'sobel_magnitude': sobel_magnitude,
            'is_blurry': blur_probability > 0.6
        }
    
    def start_realtime_analysis(self, video_source=0, display_results=True):
        """
        Start real-time analysis from video source
        
        Args:
            video_source: Video source (0 for webcam, path for video file)
            display_results: Whether to display results in real-time
        """
        print(f"Starting real-time AQS analysis...")
        print(f"Device: {self.device}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Batch size: {self.batch_size}")
        
        # Open video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps} FPS")
        
        # Start processing thread
        self.processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Frame collection
        frame_count = 0
        frames_for_sequence = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):  # Video file ended
                        break
                    else:  # Webcam error
                        continue
                
                # Preprocess frame
                frame_tensor = self.preprocess_frame(frame)
                frames_for_sequence.append(frame_tensor)
                
                # When we have enough frames, create sequence
                if len(frames_for_sequence) >= self.sequence_length:
                    sequence = torch.stack(frames_for_sequence)
                    
                    # Add to processing queue (non-blocking)
                    try:
                        self.frame_buffer.put(sequence, block=False)
                    except queue.Full:
                        # Skip frame if buffer is full (maintain real-time)
                        pass
                    
                    # Slide window
                    frames_for_sequence = frames_for_sequence[self.sequence_length - self.overlap:]
                
                # Display results if available
                if display_results:
                    self._display_results(frame)
                
                frame_count += 1
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.last_fps_time
                    current_fps = 30 / elapsed
                    print(f"Processing FPS: {current_fps:.1f}")
                    self.last_fps_time = current_time
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Cleanup
            self.processing = False
            if self.processing_thread:
                self.processing_thread.join()
            cap.release()
            cv2.destroyAllWindows()
            
            print("Real-time analysis stopped")
            self._print_performance_stats()
    
    def _processing_loop(self):
        """Background processing loop"""
        sequences_batch = []
        
        while self.processing:
            try:
                # Get sequence from buffer
                sequence = self.frame_buffer.get(timeout=0.1)
                sequences_batch.append(sequence)
                
                # Process batch when full or timeout
                if len(sequences_batch) >= self.batch_size:
                    results = self.process_sequence_batch(sequences_batch)
                    
                    # Add results to output queue
                    for result in results:
                        try:
                            self.result_queue.put(result, block=False)
                        except queue.Full:
                            # Remove old result if queue is full
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put(result, block=False)
                            except queue.Empty:
                                pass
                    
                    sequences_batch = []
                    
            except queue.Empty:
                # Process partial batch if timeout
                if sequences_batch:
                    results = self.process_sequence_batch(sequences_batch)
                    for result in results:
                        try:
                            self.result_queue.put(result, block=False)
                        except queue.Full:
                            pass
                    sequences_batch = []
                continue
    
    def _display_results(self, frame: np.ndarray):
        """Display real-time results on frame"""
        try:
            # Get latest result
            result = self.result_queue.get_nowait()
            
            # Draw AQS score
            aqs_score = result['aqs_score']
            color = (0, 255, 0) if aqs_score > 0.7 else (0, 165, 255) if aqs_score > 0.5 else (0, 0, 255)
            
            cv2.putText(frame, f"AQS: {aqs_score:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw processing time
            proc_time = result['processing_time_ms']
            cv2.putText(frame, f"Process: {proc_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw detected artifacts
            y_offset = 110
            for artifact_name, artifact_data in result['artifacts'].items():
                prob = artifact_data['probability']
                intensity = artifact_data['intensity']
                impact = prob * intensity  # Calculate actual impact
                text = f"{artifact_name}: {impact:.3f} (p={prob:.2f})"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25
            
        except queue.Empty:
            # No new results available
            pass
        
        # Show frame
        cv2.imshow('Real-time AQS Analysis', frame)
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            min_time = np.min(self.processing_times) * 1000
            max_time = np.max(self.processing_times) * 1000
            
            print(f"\nPerformance Statistics:")
            print(f"Average processing time: {avg_time:.2f}ms")
            print(f"Min processing time: {min_time:.2f}ms")
            print(f"Max processing time: {max_time:.2f}ms")
            print(f"Theoretical max FPS: {1000/avg_time:.1f}")
        
        # AQS statistics
        stats = self.aqs_metric.get_summary_stats()
        if stats:
            print(f"\nAQS Statistics:")
            print(f"Mean AQS: {stats['mean_aqs']:.3f}")
            print(f"AQS Range: {stats['min_aqs']:.3f} - {stats['max_aqs']:.3f}")
            print(f"Good quality ratio: {stats['good_quality_ratio']:.1%}")
            print(f"Poor quality ratio: {stats['poor_quality_ratio']:.1%}")
    
    def analyze_video_file(self, video_path: str, output_dir: str = "aqs_analysis"):
        """Analyze complete video file and generate report"""
        print(f"Analyzing video file: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset AQS history
        self.aqs_metric.reset_history()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video: {total_frames} frames @ {fps} FPS")
        
        # Process video
        frame_count = 0
        frames_for_sequence = []
        all_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame_tensor = self.preprocess_frame(frame)
            frames_for_sequence.append(frame_tensor)
            
            # Process sequence when ready
            if len(frames_for_sequence) >= self.sequence_length:
                sequence = torch.stack(frames_for_sequence)
                results = self.process_sequence_batch([sequence])
                
                if results:
                    result = results[0]
                    result['frame_number'] = frame_count
                    all_results.append(result)
                
                # Slide window
                frames_for_sequence = frames_for_sequence[self.sequence_length - self.overlap:]
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        # Generate analysis report
        self._generate_analysis_report(all_results, output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        return all_results
    
    def _generate_analysis_report(self, results: List[Dict], output_dir: str):
        """Generate comprehensive analysis report"""
        # Save detailed results
        results_file = os.path.join(output_dir, 'aqs_analysis_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_sequences': len(results),
                'results': results,
                'summary_stats': self.aqs_metric.get_summary_stats()
            }, f, indent=2)
        
        # Create visualizations
        artifact_names = [ArtifactType.get_type_name(t) for t in ArtifactType.get_all_types()]
        
        self.aqs_metric.create_temporal_plot(
            os.path.join(output_dir, 'aqs_temporal.png')
        )
        
        self.aqs_metric.create_artifact_contribution_plot(
            artifact_names,
            os.path.join(output_dir, 'artifact_contributions.png')
        )
        
        # Create AQS distribution plot
        aqs_scores = [r['aqs_score'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(aqs_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(aqs_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(aqs_scores):.3f}')
        plt.axvline(0.7, color='green', linestyle='--', alpha=0.7, label='Good Quality')
        plt.axvline(0.5, color='orange', linestyle='--', alpha=0.7, label='Poor Quality')
        
        plt.xlabel('AQS Score')
        plt.ylabel('Frequency')
        plt.title('AQS Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aqs_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time AQS Video Analysis')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--video_source', type=str, default='0',
                       help='Video source (0 for webcam, path for file)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for processing')
    parser.add_argument('--output_dir', type=str, default='aqs_analysis',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    # Convert video source
    video_source = args.video_source
    if video_source.isdigit():
        video_source = int(video_source)
    
    # Create analyzer
    analyzer = RealTimeAQSAnalyzer(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run analysis
    if isinstance(video_source, str) and os.path.exists(video_source):
        # Analyze video file
        analyzer.analyze_video_file(video_source, args.output_dir)
    else:
        # Real-time analysis
        analyzer.start_realtime_analysis(video_source)

if __name__ == "__main__":
    main()
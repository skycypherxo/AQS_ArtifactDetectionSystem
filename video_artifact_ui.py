"""
Video Artifact Detection Web UI
===============================

A Streamlit web interface for video artifact detection that allows users to:
- Upload video files
- Analyze artifacts in real-time
- View detailed results and visualizations
- Download analysis reports
"""

import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Tuple
import base64
from io import BytesIO

from efficientnet_artifact_detector import EfficientNetArtifactDetector, ArtifactType
import torchvision.transforms as transforms

class VideoArtifactAnalyzerUI:
    """UI wrapper for video artifact analysis"""
    
    def __init__(self):
        self.sequence_length = 7
        self.overlap = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = self._get_transform()
        
    def _get_transform(self):
        """Get preprocessing transform"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            self.model = EfficientNetArtifactDetector(
                num_artifact_types=len(ArtifactType.get_all_types()),
                sequence_length=self.sequence_length
            )
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                st.success(f"✅ Model loaded successfully from {model_path}")
            else:
                st.warning(f"⚠️ Model file not found. Using untrained model.")
            
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def extract_frames(self, video_path: str, max_frames: int = 200) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            frame_count += 1
            progress = min(frame_count / min(max_frames or total_frames, total_frames), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Extracting frames: {frame_count}/{min(max_frames or total_frames, total_frames)}")
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        return frames, fps
    
    def preprocess_sequence(self, frames: List[np.ndarray], start_idx: int) -> torch.Tensor:
        """Preprocess sequence of frames"""
        sequence_frames = []
        
        for i in range(self.sequence_length):
            frame_idx = min(start_idx + i, len(frames) - 1)
            frame = frames[frame_idx]
            frame_tensor = self.transform(frame)
            sequence_frames.append(frame_tensor)
        
        return torch.stack(sequence_frames).unsqueeze(0)
    
    def analyze_video(self, video_path: str) -> Dict:
        """Analyze video for artifacts"""
        if self.model is None:
            st.error("❌ Model not loaded!")
            return None
        
        # Extract frames
        st.info("🎬 Extracting video frames...")
        frames, fps = self.extract_frames(video_path)
        
        if len(frames) < self.sequence_length:
            st.error(f"❌ Video too short. Need at least {self.sequence_length} frames.")
            return None
        
        st.success(f"✅ Extracted {len(frames)} frames (FPS: {fps:.1f})")
        
        # Analyze sequences
        st.info("🔍 Analyzing video sequences...")
        results = []
        step_size = self.sequence_length - self.overlap
        total_sequences = len(range(0, len(frames) - self.sequence_length + 1, step_size))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, start_idx in enumerate(range(0, len(frames) - self.sequence_length + 1, step_size)):
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
            for j, artifact_type in enumerate(ArtifactType.get_all_types()):
                artifact_name = ArtifactType.get_type_name(artifact_type)
                prob = float(artifact_probs[j])
                intensity = float(intensity_scores[j])
                
                if prob > 0.3:  # Detection threshold
                    sequence_result['artifacts'][artifact_name] = {
                        'probability': prob,
                        'intensity': intensity
                    }
            
            results.append(sequence_result)
            
            # Update progress
            progress = (i + 1) / total_sequences
            progress_bar.progress(progress)
            status_text.text(f"Analyzing sequence {i + 1}/{total_sequences}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Generate summary
        summary = self._generate_summary(results, frames, fps)
        
        return {
            'summary': summary,
            'detailed_results': results,
            'frames': frames,
            'fps': fps,
            'total_frames': len(frames)
        }
    
    def _generate_summary(self, results: List[Dict], frames: List[np.ndarray], fps: float) -> Dict:
        """Generate analysis summary"""
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
        
        # Generate summary
        summary = {
            'overall_quality_score': float(avg_quality),
            'total_sequences_analyzed': total_sequences,
            'artifacts_detected': len(artifact_counts),
            'video_duration': len(frames) / fps if fps > 0 else 0,
            'fps': fps,
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

def create_quality_timeline_plot(results: List[Dict]) -> go.Figure:
    """Create interactive quality timeline plot"""
    frame_numbers = [r['start_frame'] for r in results]
    quality_scores = [r['quality_score'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=quality_scores,
        mode='lines+markers',
        name='Quality Score',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                  annotation_text="Good Quality Threshold")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Poor Quality Threshold")
    
    fig.update_layout(
        title="Video Quality Timeline",
        xaxis_title="Frame Number",
        yaxis_title="Quality Score",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )
    
    return fig

def create_artifact_summary_plot(artifact_summary: Dict) -> go.Figure:
    """Create artifact summary plot"""
    if not artifact_summary:
        return go.Figure().add_annotation(text="No artifacts detected", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    artifacts = list(artifact_summary.keys())
    percentages = [data['percentage'] for data in artifact_summary.values()]
    intensities = [data['average_intensity'] for data in artifact_summary.values()]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Artifact Occurrence (%)', 'Average Intensity'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Occurrence chart
    fig.add_trace(
        go.Bar(x=artifacts, y=percentages, name="Occurrence %", 
               marker_color='lightblue'),
        row=1, col=1
    )
    
    # Intensity chart
    fig.add_trace(
        go.Bar(x=artifacts, y=intensities, name="Avg Intensity", 
               marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    return fig

def main():
    st.set_page_config(
        page_title="Video Artifact Detector",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Artifact Detection System")
    st.markdown("Upload a video to analyze for compression artifacts, blur, noise, and quality issues.")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = VideoArtifactAnalyzerUI()
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Configuration")
    
    model_options = {
        "Best F1 Score": "checkpoints/best_f1_checkpoint.pth",
        "Best Loss": "checkpoints/best_loss_checkpoint.pth", 
        "Latest": "checkpoints/latest_checkpoint.pth"
    }
    
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    model_path = model_options[selected_model]
    
    if st.sidebar.button("Load Model") or st.session_state.analyzer.model is None:
        with st.spinner("Loading model..."):
            st.session_state.analyzer.load_model(model_path)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for artifacts"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size / (1024*1024):.1f} MB")
            
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video..."):
                    results = st.session_state.analyzer.analyze_video(temp_video_path)
                    
                    if results:
                        st.session_state.analysis_results = results
                        st.success("✅ Analysis complete!")
                    
                # Clean up temp file
                os.unlink(temp_video_path)
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            summary = results['summary']
            
            # Summary metrics
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)
            
            with col2_1:
                st.metric("Overall Quality", f"{summary['overall_quality_score']:.3f}")
            with col2_2:
                st.metric("Duration", f"{summary['video_duration']:.1f}s")
            with col2_3:
                st.metric("FPS", f"{summary['fps']:.1f}")
            with col2_4:
                st.metric("Artifacts Found", summary['artifacts_detected'])
            
            # Quality timeline
            st.subheader("📈 Quality Timeline")
            quality_fig = create_quality_timeline_plot(results['detailed_results'])
            st.plotly_chart(quality_fig, use_container_width=True)
            
            # Artifact summary
            if summary['artifact_summary']:
                st.subheader("🎯 Detected Artifacts")
                artifact_fig = create_artifact_summary_plot(summary['artifact_summary'])
                st.plotly_chart(artifact_fig, use_container_width=True)
                
                # Detailed artifact table
                artifact_data = []
                for name, data in summary['artifact_summary'].items():
                    artifact_data.append({
                        'Artifact Type': name.replace('_', ' ').title(),
                        'Occurrences': data['occurrences'],
                        'Percentage': f"{data['percentage']:.1f}%",
                        'Avg Intensity': f"{data['average_intensity']:.3f}"
                    })
                
                df = pd.DataFrame(artifact_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.success("🎉 No significant artifacts detected!")
            
            # Download results
            st.subheader("💾 Download Results")
            
            # Prepare download data
            download_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'video_info': {
                    'total_frames': results['total_frames'],
                    'fps': results['fps'],
                    'duration': summary['video_duration']
                },
                'summary': summary,
                'detailed_results': results['detailed_results']
            }
            
            json_str = json.dumps(download_data, indent=2)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_str,
                file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("👆 Upload and analyze a video to see results here")

if __name__ == "__main__":
    main()
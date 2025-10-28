"""
AQS Web Dashboard
================

Interactive web dashboard for real-time AQS (Artifact Quality Score) analysis
with live video processing, charts, and detailed artifact breakdown.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import cv2
import torch
import time
import threading
import queue
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import tempfile

from realtime_aqs_analyzer import RealTimeAQSAnalyzer
from aqs_metric import AQSMetric
from efficientnet_artifact_detector import ArtifactType

# Page configuration
st.set_page_config(
    page_title="AQS Video Quality Dashboard",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .quality-good {
        color: #28a745;
        font-weight: bold;
    }
    .quality-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .quality-poor {
        color: #dc3545;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

class AQSWebDashboard:
    """Web dashboard for AQS analysis"""
    
    def __init__(self):
        self.analyzer = None
        self.analysis_results = []
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        # Initialize session state
        if 'aqs_history' not in st.session_state:
            st.session_state.aqs_history = []
        if 'artifact_history' not in st.session_state:
            st.session_state.artifact_history = []
        if 'processing_stats' not in st.session_state:
            st.session_state.processing_stats = {
                'total_frames': 0,
                'avg_processing_time': 0,
                'total_processing_time': 0
            }
        if 'sample_frames' not in st.session_state:
            st.session_state.sample_frames = []
        if 'video_info' not in st.session_state:
            st.session_state.video_info = {}
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = None
        if 'analyzer_initialized' not in st.session_state:
            st.session_state.analyzer_initialized = False
    
    def initialize_analyzer(self, model_path: str, device: str = 'auto'):
        """Initialize the AQS analyzer"""
        try:
            # Force CPU for web interface to avoid device issues
            if device == 'auto':
                device = 'cpu'  # Use CPU for web interface stability
            
            # Check if model file exists
            if not os.path.exists(model_path):
                st.warning(f"⚠️ Model file not found: {model_path}")
                st.info("🎯 You can still use demo data to explore the dashboard")
                return False
            
            analyzer = RealTimeAQSAnalyzer(
                model_path=model_path,
                device=device,
                batch_size=2  # Smaller batch for web interface
            )
            
            # Store in session state
            st.session_state.analyzer = analyzer
            st.session_state.analyzer_initialized = True
            
            st.info(f"🔧 Analyzer initialized on device: {analyzer.device}")
            return True
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {e}")
            st.error("💡 Try using demo data instead, or check if the model file exists")
            st.session_state.analyzer = None
            st.session_state.analyzer_initialized = False
            return False
    
    def analyze_uploaded_video(self, video_file, progress_bar, status_text, analyzer):
        """Analyze uploaded video file and capture sample frames"""
        st.info(f"📁 Processing file: {video_file.name} ({video_file.size / 1024 / 1024:.1f} MB)")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_path = tmp_file.name
        
        st.info(f"💾 Saved to temporary file: {tmp_path}")
        
        try:
            # Get video info first
            status_text.text("📹 Reading video properties...")
            cap = cv2.VideoCapture(tmp_path)
            
            if not cap.isOpened():
                st.error("❌ Cannot open video file. Please check the file format.")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            st.session_state.video_info = {
                'filename': video_file.name,
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            }
            cap.release()
            
            st.info(f"📊 Video info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                status_text.text("Analyzing video and capturing frames...")
                
                # Analyze video with frame capture
                results = self.analyze_video_with_frames(tmp_path, tmp_dir, progress_bar, status_text, analyzer)
                
                if not results:
                    return None
                
                # Update session state
                st.session_state.aqs_history = [r['aqs_score'] for r in results]
                
                # Process artifact history
                artifact_counts = {}
                artifact_intensities = {}
                
                for result in results:
                    for artifact_name, artifact_data in result['artifacts'].items():
                        if artifact_name not in artifact_counts:
                            artifact_counts[artifact_name] = 0
                            artifact_intensities[artifact_name] = []
                        
                        artifact_counts[artifact_name] += 1
                        # Use impact if available, otherwise calculate it
                        if 'impact' in artifact_data:
                            impact_value = artifact_data['impact']
                        else:
                            impact_value = artifact_data['probability'] * artifact_data['intensity']
                        artifact_intensities[artifact_name].append(impact_value)
                
                st.session_state.artifact_history = {
                    'counts': artifact_counts,
                    'intensities': artifact_intensities
                }
                
                # Update processing stats
                processing_times = [r['processing_time_ms'] for r in results]
                st.session_state.processing_stats = {
                    'total_frames': len(results),
                    'avg_processing_time': np.mean(processing_times),
                    'total_processing_time': sum(processing_times)
                }
                
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
                
                return results
                
        except Exception as e:
            st.error(f"Error analyzing video: {e}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def analyze_video_with_frames(self, video_path, output_dir, progress_bar, status_text, analyzer):
        """Analyze video and capture sample frames"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Cannot open video file")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process video and capture frames
            frame_count = 0
            frames_for_sequence = []
            all_results = []
            sample_frames = []
            
            # Capture frames at regular intervals for display
            frame_capture_interval = max(1, total_frames // 12)  # Capture ~12 frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame_tensor = analyzer.preprocess_frame(frame)
                frames_for_sequence.append(frame_tensor)
                
                # Capture sample frames for display
                if frame_count % frame_capture_interval == 0:
                    # Convert frame to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sample_frames.append({
                        'frame_number': frame_count,
                        'image': frame_rgb,
                        'aqs_score': None  # Will be filled later
                    })
                
                # Process sequence when ready
                if len(frames_for_sequence) >= analyzer.sequence_length:
                    sequence = torch.stack(frames_for_sequence)
                    results = analyzer.process_sequence_batch([sequence])
                    
                    if results:
                        result = results[0]
                        result['frame_number'] = frame_count
                        all_results.append(result)
                        
                        # Update AQS score for nearby sample frames
                        for sample_frame in sample_frames:
                            if (sample_frame['aqs_score'] is None and 
                                abs(sample_frame['frame_number'] - frame_count) <= analyzer.sequence_length):
                                sample_frame['aqs_score'] = result['aqs_score']
                                sample_frame['artifacts'] = result['artifacts']
                    
                    # Slide window
                    frames_for_sequence = frames_for_sequence[analyzer.sequence_length - analyzer.overlap:]
                
                frame_count += 1
                
                # Update progress
                if frame_count % 50 == 0:
                    progress = min(0.9, frame_count / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {frame_count}/{total_frames} frames...")
            
            cap.release()
            
            # Store sample frames in session state
            st.session_state.sample_frames = sample_frames[:12]  # Keep max 12 frames
            
            return all_results
            
        except Exception as e:
            st.error(f"Error in video analysis: {e}")
            return None
    
    def create_aqs_timeline_chart(self):
        """Create AQS timeline chart"""
        if not st.session_state.aqs_history:
            return None
        
        aqs_scores = st.session_state.aqs_history
        
        fig = go.Figure()
        
        # Add AQS line
        fig.add_trace(go.Scatter(
            x=list(range(len(aqs_scores))),
            y=aqs_scores,
            mode='lines+markers',
            name='AQS Score',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        # REALISTIC quality thresholds - blur should be "Poor" (0.3-0.5)
        fig.add_hline(y=0.85, line_dash="dash", line_color="green", 
                     annotation_text="Excellent Quality (0.85+)")
        fig.add_hline(y=0.70, line_dash="dash", line_color="lightgreen", 
                     annotation_text="Good Quality (0.70+)")
        fig.add_hline(y=0.50, line_dash="dash", line_color="orange", 
                     annotation_text="Fair Quality (0.50+)")
        fig.add_hline(y=0.30, line_dash="dash", line_color="red", 
                     annotation_text="Poor Quality (<0.30)")
        
        fig.update_layout(
            title="AQS Score Over Time",
            xaxis_title="Sequence Number",
            yaxis_title="AQS Score",
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_artifact_distribution_chart(self):
        """Create artifact distribution chart"""
        if not st.session_state.artifact_history:
            return None
        
        artifact_counts = st.session_state.artifact_history.get('counts', {})
        
        if not artifact_counts:
            return None
        
        # Calculate percentages
        total_sequences = st.session_state.processing_stats['total_frames']
        
        artifacts = list(artifact_counts.keys())
        percentages = [count / total_sequences * 100 for count in artifact_counts.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=artifacts,
                y=percentages,
                marker_color='lightcoral',
                text=[f'{p:.1f}%' for p in percentages],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Artifact Detection Frequency",
            xaxis_title="Artifact Type",
            yaxis_title="Detection Percentage (%)",
            height=400
        )
        
        return fig
    
    def create_quality_distribution_chart(self):
        """Create quality distribution histogram"""
        if not st.session_state.aqs_history:
            return None
        
        aqs_scores = st.session_state.aqs_history
        
        fig = go.Figure(data=[
            go.Histogram(
                x=aqs_scores,
                nbinsx=30,
                marker_color='skyblue',
                opacity=0.7
            )
        ])
        
        # Add vertical lines for quality thresholds
        fig.add_vline(x=np.mean(aqs_scores), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {np.mean(aqs_scores):.3f}")
        fig.add_vline(x=0.85, line_dash="dash", line_color="green",
                     annotation_text="Excellent Quality")
        fig.add_vline(x=0.70, line_dash="dash", line_color="lightgreen",
                     annotation_text="Good Quality")
        fig.add_vline(x=0.50, line_dash="dash", line_color="orange",
                     annotation_text="Fair Quality")
        fig.add_vline(x=0.30, line_dash="dash", line_color="red",
                     annotation_text="Poor Quality")
        
        fig.update_layout(
            title="AQS Score Distribution",
            xaxis_title="AQS Score",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    def create_artifact_intensity_chart(self):
        """Create artifact impact chart (probability × intensity)"""
        if not st.session_state.artifact_history:
            return None
        
        artifact_intensities = st.session_state.artifact_history.get('intensities', {})
        
        if not artifact_intensities:
            return None
        
        artifacts = list(artifact_intensities.keys())
        avg_intensities = [np.mean(intensities) for intensities in artifact_intensities.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=artifacts,
                y=avg_intensities,
                marker_color='lightgreen',
                text=[f'{i:.3f}' for i in avg_intensities],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Average Artifact Impact",
            xaxis_title="Artifact Type",
            yaxis_title="Average Impact (Probability × Intensity)",
            height=400
        )
        
        return fig
    
    def display_summary_metrics(self):
        """Display summary metrics"""
        if not st.session_state.aqs_history:
            st.info("Upload and analyze a video to see metrics")
            return
        
        aqs_scores = st.session_state.aqs_history
        stats = st.session_state.processing_stats
        
        # Calculate quality metrics
        mean_aqs = np.mean(aqs_scores)
        excellent_quality_ratio = sum(1 for score in aqs_scores if score >= 0.85) / len(aqs_scores)
        good_quality_ratio = sum(1 for score in aqs_scores if score >= 0.70) / len(aqs_scores)
        fair_quality_ratio = sum(1 for score in aqs_scores if 0.50 <= score < 0.70) / len(aqs_scores)
        poor_quality_ratio = sum(1 for score in aqs_scores if score < 0.30) / len(aqs_scores)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Mean AQS Score",
                value=f"{mean_aqs:.3f}",
                delta=f"Range: {min(aqs_scores):.3f} - {max(aqs_scores):.3f}"
            )
        
        with col2:
            st.metric(
                label="Good+ Quality",
                value=f"{good_quality_ratio:.1%}",
                delta="Frames ≥ 0.70 AQS"
            )
        
        with col3:
            st.metric(
                label="Processing Speed",
                value=f"{stats['avg_processing_time']:.1f}ms",
                delta=f"Total: {stats['total_frames']} frames"
            )
        
        with col4:
            # REALISTIC quality assessment - blur should be "Poor" (0.3-0.5)
            if mean_aqs >= 0.85:
                quality_status = "Excellent"
                bg_color = "#d1f2eb"  # Very light green
                text_color = "#0e6b47"  # Dark green
                border_color = "#a3e4d7"
            elif mean_aqs >= 0.70:
                quality_status = "Good"
                bg_color = "#d4edda"  # Light green
                text_color = "#155724"  # Dark green
                border_color = "#c3e6cb"
            elif mean_aqs >= 0.50:
                quality_status = "Fair"
                bg_color = "#fff3cd"  # Light yellow
                text_color = "#856404"  # Dark yellow
                border_color = "#ffeaa7"
            elif mean_aqs >= 0.30:
                quality_status = "Poor"
                bg_color = "#ffeaa7"  # Light orange
                text_color = "#b7791f"  # Dark orange
                border_color = "#ffdf7e"
            else:
                quality_status = "Very Poor"
                bg_color = "#f8d7da"  # Light red
                text_color = "#721c24"  # Dark red
                border_color = "#f5c6cb"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 1rem; border-radius: 0.5rem; 
                       border: 2px solid {border_color}; text-align: center;">
                <h3 style="color: {text_color}; margin: 0;">Overall Quality</h3>
                <p style="color: {text_color}; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                    {quality_status}
                </p>
                <small style="color: {text_color};">Based on AQS analysis</small>
            </div>
            """, unsafe_allow_html=True)
    
    def display_artifact_details(self):
        """Display detailed artifact information"""
        if not st.session_state.artifact_history:
            return
        
        artifact_counts = st.session_state.artifact_history.get('counts', {})
        artifact_intensities = st.session_state.artifact_history.get('intensities', {})
        
        if not artifact_counts:
            return
        
        st.subheader("Detected Artifacts Details")
        
        # Create DataFrame for artifact details
        artifact_data = []
        total_sequences = st.session_state.processing_stats['total_frames']
        
        for artifact_name in artifact_counts:
            count = artifact_counts[artifact_name]
            percentage = count / total_sequences * 100
            avg_intensity = np.mean(artifact_intensities[artifact_name])
            
            artifact_data.append({
                'Artifact Type': artifact_name.replace('_', ' ').title(),
                'Occurrences': count,
                'Percentage': f"{percentage:.1f}%",
                'Avg Intensity': f"{avg_intensity:.3f}",
                'Severity': 'Critical' if avg_intensity > 0.8 else 'High' if avg_intensity > 0.6 else 'Medium' if avg_intensity > 0.3 else 'Low'
            })
        
        df = pd.DataFrame(artifact_data)
        st.dataframe(df, use_container_width=True)
    
    def display_sample_frames(self):
        """Display sample frames with AQS scores"""
        if not st.session_state.sample_frames:
            return
        
        st.subheader("🖼️ Sample Frames with AQS Scores")
        
        # Display video info
        if st.session_state.video_info:
            info = st.session_state.video_info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Video File", info['filename'])
            with col2:
                st.metric("Resolution", f"{info['width']}x{info['height']}")
            with col3:
                st.metric("Duration", f"{info['duration']:.1f}s")
            with col4:
                st.metric("FPS", f"{info['fps']:.1f}")
        
        # Display frames in a grid
        frames = st.session_state.sample_frames
        
        # Create columns for frame grid (3 frames per row)
        frames_per_row = 3
        num_rows = (len(frames) + frames_per_row - 1) // frames_per_row
        
        for row in range(num_rows):
            cols = st.columns(frames_per_row)
            
            for col_idx in range(frames_per_row):
                frame_idx = row * frames_per_row + col_idx
                
                if frame_idx < len(frames):
                    frame_data = frames[frame_idx]
                    
                    with cols[col_idx]:
                        # Display frame image
                        st.image(
                            frame_data['image'], 
                            caption=f"Frame {frame_data['frame_number']}",
                            use_column_width=True
                        )
                        
                        # Display AQS score
                        if frame_data['aqs_score'] is not None:
                            aqs_score = frame_data['aqs_score']
                            
                            # REALISTIC quality thresholds - blur should be "Poor" (0.3-0.5)
                            if aqs_score >= 0.85:
                                quality_color = "🌟"
                                quality_text = "Excellent"
                                bg_color = "#d1f2eb"  # Very light green
                                text_color = "#0e6b47"  # Dark green
                            elif aqs_score >= 0.70:
                                quality_color = "🟢"
                                quality_text = "Good"
                                bg_color = "#d4edda"  # Light green
                                text_color = "#155724"  # Dark green
                            elif aqs_score >= 0.50:
                                quality_color = "🟡"
                                quality_text = "Fair"
                                bg_color = "#fff3cd"  # Light yellow
                                text_color = "#856404"  # Dark yellow
                            elif aqs_score >= 0.30:
                                quality_color = "🟠"
                                quality_text = "Poor"
                                bg_color = "#ffeaa7"  # Light orange
                                text_color = "#b7791f"  # Dark orange
                            else:
                                quality_color = "🔴"
                                quality_text = "Very Poor"
                                bg_color = "#f8d7da"  # Light red
                                text_color = "#721c24"  # Dark red
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 0.5rem; background-color: {bg_color}; 
                                       border-radius: 0.5rem; margin-top: 0.5rem; border: 1px solid #ddd;">
                                <strong style="color: {text_color}; font-size: 1.1rem;">AQS: {aqs_score:.3f}</strong><br>
                                <span style="color: {text_color}; font-weight: bold;">{quality_color} {quality_text}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show detected artifacts
                            if 'artifacts' in frame_data and frame_data['artifacts']:
                                artifacts_text = ", ".join([
                                    f"{name.replace('_', ' ').title()}: {data['probability']:.2f}"
                                    for name, data in frame_data['artifacts'].items()
                                ])
                                
                                st.markdown(f"""
                                <div style="font-size: 0.8rem; color: #333; text-align: center; margin-top: 0.25rem; 
                                           background-color: #f8f9fa; padding: 0.25rem; border-radius: 0.25rem; border: 1px solid #e9ecef;">
                                    <strong>Artifacts:</strong><br>{artifacts_text}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="text-align: center; padding: 0.5rem; background-color: #e9ecef; 
                                       border-radius: 0.5rem; margin-top: 0.5rem; border: 1px solid #ddd;">
                                <em style="color: #6c757d;">Processing...</em>
                            </div>
                            """, unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">🎬 AQS Video Quality Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Model selection
            model_path = st.selectbox(
                "Select Model",
                options=[
                    "checkpoints/best_loss_checkpoint.pth",
                    "checkpoints/best_f1_checkpoint.pth", 
                    "checkpoints/latest_checkpoint.pth"
                ],
                help="Choose the trained model checkpoint"
            )
            
            # Device selection
            device = st.selectbox(
                "Processing Device",
                options=["cpu", "cuda", "auto"],
                index=0,  # Default to CPU
                help="CPU recommended for web interface stability"
            )
            
            # Initialize analyzer button
            if st.button("🚀 Initialize Analyzer"):
                with st.spinner("Initializing analyzer..."):
                    try:
                        if self.initialize_analyzer(model_path, device):
                            st.success("✅ Analyzer initialized successfully!")
                            
                            # Test with dummy data
                            st.info("🧪 Testing analyzer with dummy data...")
                            dummy_input = torch.randn(1, 7, 3, 224, 224).to(st.session_state.analyzer.device)
                            with torch.no_grad():
                                test_output = st.session_state.analyzer.model(dummy_input)
                            st.success("✅ Analyzer test passed!")
                        else:
                            st.error("❌ Failed to initialize analyzer")
                    except Exception as e:
                        st.error(f"❌ Error initializing analyzer: {str(e)}")
                        st.exception(e)
            
            st.divider()
            
            # Analysis options
            st.header("Analysis Options")
            
            # Video upload
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for AQS analysis"
            )
            
            # Show analyzer status
            if st.session_state.analyzer_initialized and st.session_state.analyzer:
                st.success("✅ Analyzer ready")
            else:
                st.warning("⚠️ Please initialize analyzer first")
            
            if uploaded_file:
                if st.session_state.analyzer_initialized and st.session_state.analyzer:
                    if st.button("🎬 Analyze Video", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            with st.spinner("Processing video..."):
                                results = self.analyze_uploaded_video(
                                    uploaded_file, progress_bar, status_text, st.session_state.analyzer
                                )
                                
                                if results:
                                    st.success(f"✅ Analysis complete! Processed {len(results)} sequences.")
                                    st.balloons()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Analysis failed. Please check the video file.")
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.exception(e)
                else:
                    st.error("❌ Please initialize the analyzer first before uploading video")
            
            # Quick test with sample video
            if st.button("🎯 Quick Test (Sample Video)") and st.session_state.analyzer_initialized and st.session_state.analyzer:
                try:
                    # Use the existing sample video
                    sample_video_path = "5192-183786490_tiny.mp4"
                    if os.path.exists(sample_video_path):
                        st.info("🎬 Running quick test with sample video...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Analyze sample video (first 50 frames only for speed)
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            results = st.session_state.analyzer.analyze_video_file(sample_video_path, tmp_dir)
                            
                            # Take only first 20 results for quick demo
                            results = results[:20] if results else []
                            
                            if results:
                                # Update session state with sample results
                                st.session_state.aqs_history = [r['aqs_score'] for r in results]
                                
                                # Process artifacts
                                artifact_counts = {}
                                artifact_intensities = {}
                                
                                for result in results:
                                    for artifact_name, artifact_data in result['artifacts'].items():
                                        if artifact_name not in artifact_counts:
                                            artifact_counts[artifact_name] = 0
                                            artifact_intensities[artifact_name] = []
                                        
                                        artifact_counts[artifact_name] += 1
                                        artifact_intensities[artifact_name].append(artifact_data['intensity'])
                                
                                st.session_state.artifact_history = {
                                    'counts': artifact_counts,
                                    'intensities': artifact_intensities
                                }
                                
                                st.session_state.processing_stats = {
                                    'total_frames': len(results),
                                    'avg_processing_time': np.mean([r['processing_time_ms'] for r in results]),
                                    'total_processing_time': sum([r['processing_time_ms'] for r in results])
                                }
                                
                                st.session_state.video_info = {
                                    'filename': 'Sample Video (5192-183786490_tiny.mp4)',
                                    'total_frames': len(results),
                                    'fps': 30.0,
                                    'width': 1920,
                                    'height': 1080,
                                    'duration': len(results) / 30.0
                                }
                                
                                progress_bar.progress(1.0)
                                st.success(f"✅ Quick test complete! Analyzed {len(results)} sequences.")
                                st.rerun()
                            else:
                                st.error("❌ Quick test failed")
                    else:
                        st.error("❌ Sample video not found")
                except Exception as e:
                    st.error(f"❌ Quick test error: {str(e)}")
            
            st.divider()
            
            # Demo data button
            if st.button("📊 Load Demo Data"):
                # Generate realistic demo data - blur should be "Poor" (0.3-0.5)
                excellent_samples = np.random.normal(0.88, 0.02, 15)  # 15% excellent
                good_samples = np.random.normal(0.75, 0.03, 25)       # 25% good
                fair_samples = np.random.normal(0.60, 0.04, 30)       # 30% fair
                poor_samples = np.random.normal(0.40, 0.05, 20)       # 20% poor (blurry videos)
                very_poor_samples = np.random.normal(0.20, 0.04, 10)  # 10% very poor (severely degraded)
                
                demo_aqs = np.concatenate([excellent_samples, good_samples, fair_samples, 
                                         poor_samples, very_poor_samples])
                np.random.shuffle(demo_aqs)  # Randomize order
                demo_aqs = np.clip(demo_aqs, 0.1, 0.95)  # Realistic range
                
                st.session_state.aqs_history = demo_aqs.tolist()
                # Generate artifact data showing blur dominance and extreme penalties
                st.session_state.artifact_history = {
                    'counts': {
                        'blur_motion': 60,           # Dominant - blur everywhere
                        'blur_defocus': 45,         # Very common focus issues
                        'compression_blocking': 30,  # Common in streaming
                        'noise_gaussian': 25,       # Low light conditions
                        'compression_ringing': 20,  # High compression
                        'motion_ghosting': 15,      # Motion artifacts
                        'color_banding': 10,        # Gradient issues
                        'noise_impulse': 8          # Sensor defects
                    },
                    'intensities': {
                        'blur_motion': np.random.gamma(3, 0.3, 60).tolist(),         # Very high intensity blur
                        'blur_defocus': np.random.gamma(2.8, 0.28, 45).tolist(),     # High intensity blur
                        'compression_blocking': np.random.gamma(1.8, 0.18, 30).tolist(),
                        'noise_gaussian': np.random.gamma(1.5, 0.15, 25).tolist(),
                        'compression_ringing': np.random.gamma(2.0, 0.2, 20).tolist(),
                        'motion_ghosting': np.random.gamma(2.5, 0.25, 15).tolist(),
                        'color_banding': np.random.gamma(1.6, 0.16, 10).tolist(),
                        'noise_impulse': np.random.gamma(3.5, 0.35, 8).tolist()      # Extremely noticeable
                    }
                }
                st.session_state.processing_stats = {
                    'total_frames': 100,
                    'avg_processing_time': 45.2,
                    'total_processing_time': 4520
                }
                st.session_state.video_info = {
                    'filename': 'Demo Data',
                    'total_frames': 100,
                    'fps': 30.0,
                    'width': 1920,
                    'height': 1080,
                    'duration': 3.33
                }
                st.success("📊 Demo data loaded!")
                st.rerun()
        
        # Main content area
        if not st.session_state.aqs_history:
            st.info("👆 Upload a video file or load demo data to start analysis")
            
            # Show sample analysis explanation
            st.subheader("About AQS (Artifact Quality Score)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **AQS** is a novel no-reference video quality metric that:
                
                - 🔍 **Detects artifacts** in real-time
                - 📊 **Provides interpretable scores** (0-1 scale)
                - ⚡ **Works without reference video**
                - 🎯 **Identifies specific quality issues**
                
                **Formula:**
                ```
                AQS = α × quality_score + (1-α) × (1 - artifact_impact)
                ```
                """)
            
            with col2:
                st.markdown("""
                **Detected Artifact Types:**
                
                - Compression (blocking, ringing)
                - Motion (ghosting, judder)  
                - Noise (gaussian, impulse)
                - Blur (motion, defocus)
                - Color (banding, saturation)
                
                **REALISTIC Quality Thresholds:**
                - 🌟 Excellent: AQS ≥ 0.85 (Broadcast quality)
                - 🟢 Good: AQS ≥ 0.70 (High quality streaming)
                - 🟡 Fair: AQS ≥ 0.50 (Acceptable quality)
                - 🟠 Poor: AQS ≥ 0.30 (Blurry/noticeable artifacts)
                - 🔴 Very Poor: AQS < 0.30 (Severe degradation)
                """)
            
            return
        
        # Display summary metrics
        self.display_summary_metrics()
        
        st.divider()
        
        # Charts section
        st.subheader("📈 Analysis Charts")
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # AQS timeline
            timeline_fig = self.create_aqs_timeline_chart()
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Quality distribution
            dist_fig = self.create_quality_distribution_chart()
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
        
        with col2:
            # Artifact frequency
            freq_fig = self.create_artifact_distribution_chart()
            if freq_fig:
                st.plotly_chart(freq_fig, use_container_width=True)
            
            # Artifact intensity
            intensity_fig = self.create_artifact_intensity_chart()
            if intensity_fig:
                st.plotly_chart(intensity_fig, use_container_width=True)
        
        st.divider()
        
        # Sample frames section
        self.display_sample_frames()
        
        st.divider()
        
        # Detailed artifact information
        self.display_artifact_details()
        
        # Export options
        st.subheader("📥 Export Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Download AQS Data"):
                if st.session_state.aqs_history:
                    data = {
                        'sequence': list(range(len(st.session_state.aqs_history))),
                        'aqs_score': st.session_state.aqs_history
                    }
                    df = pd.DataFrame(data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"aqs_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("Generate Report"):
                if st.session_state.aqs_history:
                    report = self.generate_text_report()
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"aqs_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        with col3:
            if st.button("Clear Data"):
                st.session_state.aqs_history = []
                st.session_state.artifact_history = []
                st.session_state.sample_frames = []
                st.session_state.video_info = {}
                st.session_state.processing_stats = {
                    'total_frames': 0,
                    'avg_processing_time': 0,
                    'total_processing_time': 0
                }
                st.success("Data cleared!")
                st.rerun()
        
        with col4:
            if st.button("🔄 Reset Analyzer"):
                st.session_state.analyzer = None
                st.session_state.analyzer_initialized = False
                st.session_state.aqs_history = []
                st.session_state.artifact_history = []
                st.session_state.sample_frames = []
                st.session_state.video_info = {}
                st.session_state.processing_stats = {
                    'total_frames': 0,
                    'avg_processing_time': 0,
                    'total_processing_time': 0
                }
                st.success("Analyzer reset!")
                st.rerun()
    
    def generate_text_report(self):
        """Generate text report"""
        if not st.session_state.aqs_history:
            return ""
        
        aqs_scores = st.session_state.aqs_history
        stats = st.session_state.processing_stats
        
        mean_aqs = np.mean(aqs_scores)
        std_aqs = np.std(aqs_scores)
        excellent_quality_ratio = sum(1 for score in aqs_scores if score >= 0.85) / len(aqs_scores)
        good_quality_ratio = sum(1 for score in aqs_scores if score >= 0.70) / len(aqs_scores)
        fair_quality_ratio = sum(1 for score in aqs_scores if 0.50 <= score < 0.70) / len(aqs_scores)
        poor_quality_ratio = sum(1 for score in aqs_scores if score < 0.30) / len(aqs_scores)
        
        report = f"""# AQS Video Quality Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- **Total Sequences Analyzed:** {len(aqs_scores)}
- **Mean AQS Score:** {mean_aqs:.3f}
- **Standard Deviation:** {std_aqs:.3f}
- **AQS Range:** {min(aqs_scores):.3f} - {max(aqs_scores):.3f}

## Quality Distribution

- **Excellent Quality (≥0.85):** {excellent_quality_ratio:.1%}
- **Good Quality (≥0.70):** {good_quality_ratio:.1%}
- **Fair Quality (0.50-0.69):** {fair_quality_ratio:.1%}
- **Poor Quality (0.30-0.49):** {sum(1 for score in aqs_scores if 0.30 <= score < 0.50) / len(aqs_scores):.1%}
- **Very Poor Quality (<0.30):** {poor_quality_ratio:.1%}

## Performance Metrics

- **Average Processing Time:** {stats['avg_processing_time']:.1f}ms per sequence
- **Total Processing Time:** {stats['total_processing_time']:.1f}ms

## Detected Artifacts
"""
        
        if st.session_state.artifact_history:
            artifact_counts = st.session_state.artifact_history.get('counts', {})
            artifact_intensities = st.session_state.artifact_history.get('intensities', {})
            
            for artifact_name in artifact_counts:
                count = artifact_counts[artifact_name]
                percentage = count / len(aqs_scores) * 100
                avg_intensity = np.mean(artifact_intensities[artifact_name])
                
                report += f"\n- **{artifact_name.replace('_', ' ').title()}:** {count} occurrences ({percentage:.1f}%), avg intensity: {avg_intensity:.3f}"
        
        report += f"""

## Methodology

This analysis used the AQS (Artifact Quality Score) metric, which combines:
1. Artifact detection probabilities
2. Artifact intensity scores  
3. Overall quality prediction

Formula: AQS = α × quality_score + (1-α) × (1 - artifact_impact)

## Conclusions

The video shows {'excellent' if mean_aqs >= 0.85 else 'good' if mean_aqs >= 0.70 else 'fair' if mean_aqs >= 0.50 else 'poor' if mean_aqs >= 0.30 else 'very poor'} overall quality with a mean AQS of {mean_aqs:.3f}.
"""
        
        return report

def main():
    """Main function to run the dashboard"""
    dashboard = AQSWebDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
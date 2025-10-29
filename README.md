# AQS (Artifact Quality Score) - Video Quality Assessment System

A comprehensive no-reference video quality assessment system that combines deep learning-based artifact detection with perceptual quality scoring to provide interpretable video quality metrics in real-time.

## 🎯 Overview

The **AQS (Artifact Quality Score)** system provides a novel approach to video quality assessment by detecting specific artifact types and computing perceptual quality scores without requiring reference videos. Built with EfficientNet architecture and optimized for real-time processing.

**Key Innovation:** AQS delivers interpretable quality scores (0-1 scale) with detailed artifact breakdowns, making it ideal for streaming platforms, content creators, and quality monitoring applications.

## ✨ Key Features

- **🎯 No-Reference Quality Assessment**: Works without original video references
- **🔍 10 Artifact Types Detection**: Comprehensive artifact analysis including blur, noise, compression artifacts
- **⚡ Real-Time Processing**: Optimized for streaming applications (~45ms per sequence)
- **📊 Interactive Web Dashboard**: User-friendly Streamlit interface for video analysis
- **📈 Perceptual Quality Scoring**: 0-1 scale matching human perception
- **🔧 Blur-Specific Validation**: Enhanced blur detection with image processing techniques
- **📤 Export Capabilities**: JSON/CSV export for integration with other systems
- **🎮 Multiple Interfaces**: Web dashboard, command-line, and API options

## 📁 Project Structure

### Core Components
```
├── aqs_web_dashboard.py              # 🌐 Main web interface (Streamlit)
├── realtime_aqs_analyzer.py          # 🔧 Core analysis engine
├── aqs_metric.py                     # 📊 AQS computation logic
├── efficientnet_artifact_detector.py # 🧠 Deep learning model
├── train_efficientnet_model.py       # 🎯 Model training script
├── video_inference_demo.py           # 💻 Command-line interface
├── run_web_dashboard.py              # 🚀 Dashboard launcher
└── requirements.txt                  # 📦 Dependencies
```

### Analysis & Research Tools
```
├── aqs_research_framework.py         # 🔬 Research framework
├── efficiency_benchmark.py           # ⚡ Performance benchmarking
├── resource_profiler.py              # 📈 Resource monitoring
├── model_analyzer.py                 # 🔍 Model analysis tools
└── video_artifact_ui.py              # 🎨 Alternative UI
```

### Data & Models
```
├── checkpoints/                      # 💾 Trained model files
│   ├── best_f1_checkpoint.pth
│   ├── best_loss_checkpoint.pth
│   └── latest_checkpoint.pth
└── test_videos/                      # 🎬 Sample test videos
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/skycypherxo/AQS_ArtifactDetectionSystem.git
cd AQS_ArtifactDetectionSystem
```

2. **Set up virtual environment (recommended):**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
# or use the automated installer
python install_dependencies.py
```

## 💻 Usage

### 🌐 Web Dashboard (Recommended)

Launch the interactive web interface:

```bash
python run_web_dashboard.py
# or directly
streamlit run aqs_web_dashboard.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Upload videos for analysis
- Real-time quality scoring
- Interactive artifact visualization
- Export results (JSON/CSV)
- Batch processing support

### 💻 Command Line Interface

For quick analysis or automation:

```bash
# Analyze a single video
python video_inference_demo.py --video test_videos/sample.mp4

# Batch processing
python video_inference_demo.py --input_dir videos/ --output_dir results/

# With custom settings
python video_inference_demo.py --video sample.mp4 --threshold 0.2 --export json
```

### 🔬 Research & Analysis Tools

**Performance Benchmarking:**
```bash
python efficiency_benchmark.py
```

**Resource Profiling:**
```bash
python resource_profiler.py
```

**Research Framework:**
```bash
python aqs_research_framework.py
```

## 🧠 How AQS Works

### Core Algorithm

AQS combines two main components:

```
AQS = α × quality_component + (1-α) × (1 - artifact_component)
```

Where:
- **α = 0.6**: Balance between base quality and artifact penalties
- **quality_component**: Perceptual quality prediction (0-1)
- **artifact_component**: Aggregated artifact impact (0-1)

### 🔍 Artifact Detection (10 Types)

The system detects and quantifies the following artifact categories:

**🗜️ Compression Artifacts:**
- **Blocking** (penalty: 1.2x) - JPEG compression blocks and DCT artifacts
- **Ringing** (penalty: 1.2x) - Edge ringing from quantization and filtering

**🏃 Motion Artifacts:**
- **Ghosting** (penalty: 1.8x) - Double images from temporal blending
- **Judder** (penalty: 1.7x) - Frame rate inconsistencies and stuttering

**📡 Noise Artifacts:**
- **Gaussian Noise** (penalty: 1.3x) - Random pixel value variations
- **Impulse Noise** (penalty: 2.0x) - Salt-and-pepper noise spikes

**🌫️ Blur Artifacts:**
- **Motion Blur** (penalty: 1.3x) - Camera/object movement blur
- **Defocus Blur** (penalty: 1.4x) - Out-of-focus regions

**🎨 Color Artifacts:**
- **Color Banding** (penalty: 1.5x) - Quantization in color gradients
- **Color Saturation Issues** (penalty: 1.6x) - Over/under-saturated regions

Each artifact type is detected using specialized algorithms and contributes to the overall quality penalty based on its perceptual impact.

### 📊 Quality Scale

- **0.85-1.0**: 🟢 Excellent (broadcast/professional quality)
- **0.70-0.84**: 🔵 Good (high-quality streaming)
- **0.50-0.69**: 🟡 Fair (acceptable viewing experience)
- **0.30-0.49**: 🟠 Poor (noticeable quality issues)
- **0.00-0.29**: 🔴 Very Poor (severe degradation)

## 🎯 Applications

### 📺 Streaming Platforms
- Real-time quality monitoring
- Adaptive bitrate optimization
- Content quality assurance
- User experience enhancement

### 🎬 Content Creation
- Video production quality control
- Post-processing validation
- Compression optimization
- Quality benchmarking

### 🔬 Research & Development
- Video codec evaluation
- Quality metric comparison
- Perceptual studies
- Algorithm development

### 🏢 Enterprise Solutions
- Video conferencing quality monitoring
- Broadcast quality assurance
- Archive quality assessment
- Automated quality reporting

## 🔧 Technical Specifications

### Model Architecture
- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Input**: 7-frame sequences (224×224 resolution)
- **Processing**: Temporal feature aggregation with attention
- **Output**: AQS score + 10 artifact probabilities + intensity scores
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Performance Metrics
- **Inference Time**: ~45ms per sequence (GPU)
- **Memory Usage**: ~2GB GPU memory
- **Throughput**: ~22 FPS processing capability
- **Model Size**: ~61MB (checkpoint files)

## 🎓 Training Details

### Dataset & Training Process

**Training Data:**
- **Source**: Vimeo-90K septuplet dataset (high-quality video sequences)
- **Sequences**: 7-frame temporal sequences (224×224 resolution)
- **Augmentation**: Synthetic artifact generation with controlled intensities
- **Split**: 80% training, 20% validation

**Artifact Generation:**
- **Synthetic Artifacts**: Procedurally generated using computer vision techniques
- **Intensity Levels**: 0.2-0.8 range for realistic degradation simulation
- **Multi-Artifact**: Up to 3 simultaneous artifacts per sequence
- **Probability**: 80% of sequences contain artifacts during training

**Training Configuration:**
- **Architecture**: EfficientNet-B0 backbone + temporal processing layers
- **Transfer Learning**: Pre-trained ImageNet weights for feature extraction
- **Loss Function**: Multi-task loss combining classification, intensity, and quality prediction
- **Optimizer**: AdamW with differential learning rates (0.1x for backbone, 1x for new layers)
- **Scheduler**: Cosine annealing with warm restarts
- **Mixed Precision**: Automatic mixed precision for faster training
- **Regularization**: Weight decay (1e-4) and dropout for generalization

**Training Process:**
- **Epochs**: 50+ with early stopping (patience: 10)
- **Batch Size**: Adaptive based on GPU memory
- **Validation**: F1-score and loss-based model selection
- **Checkpoints**: Best F1, best loss, and latest model states
- **Monitoring**: TensorBoard logging for metrics and visualizations

## 📤 Output Examples

### JSON Output
```json
{
  "aqs_score": 0.73,
  "quality_level": "Good",
  "artifacts": {
    "blocking": 0.12,
    "blur": 0.31,
    "noise": 0.08,
    "ghosting": 0.05
  },
  "processing_time": 0.045,
  "frame_count": 7
}
```

### Web Dashboard Features
- 📊 Interactive quality charts
- 🎯 Artifact heatmaps
- 📈 Temporal quality analysis
- 💾 Export capabilities (JSON/CSV)
- 🔄 Batch processing results

## 💻 System Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (4+ cores)
- **RAM**: 8GB system memory
- **Storage**: 2GB free space
- **Python**: 3.8+ with pip

### Recommended Configuration
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster processing)
- **RAM**: 16GB+ system memory
- **CPU**: 8+ cores for batch processing
- **Storage**: SSD for faster I/O operations

### Supported Platforms
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 18.04+)
- ✅ macOS 10.15+
- ✅ Docker containers

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- 🔍 Additional artifact detection types
- ⚡ Performance optimizations
- 🎨 UI/UX enhancements
- 📱 Mobile/edge deployment
- 🧪 New quality metrics
- 📚 Documentation improvements

### Development Setup
```bash
git clone https://github.com/skycypherxo/AQS_ArtifactDetectionSystem.git
cd AQS_ArtifactDetectionSystem
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 📄 Documentation

- 📖 **[Complete System Documentation](AQS_SYSTEM_DOCUMENTATION.md)**
- 🏗️ **[Project Structure](PROJECT_STRUCTURE.md)**
- 📋 **[Usage Guide](USAGE.md)**

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **EfficientNet Architecture**: Google Research
- **PyTorch Framework**: Meta AI Research
- **Streamlit**: For the web interface
- **OpenCV**: For image processing utilities

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/skycypherxo/AQS_ArtifactDetectionSystem/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/skycypherxo/AQS_ArtifactDetectionSystem/discussions)
- 📧 **Contact**: Create an issue for support

---

**⭐ Star this repository if you find it useful!**
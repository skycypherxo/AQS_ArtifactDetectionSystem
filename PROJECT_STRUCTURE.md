# AQS System - Project Structure

## 📁 **Core Files**

### **Main Components:**
- `aqs_web_dashboard.py` - **Web interface** (Streamlit dashboard)
- `realtime_aqs_analyzer.py` - **Analysis engine** (main processing)
- `aqs_metric.py` - **Core metric** (AQS computation)
- `efficientnet_artifact_detector.py` - **Deep learning model**

### **Training & Utilities:**
- `train_efficientnet_model.py` - Model training script
- `video_inference_demo.py` - Command-line demo
- `video_artifact_ui.py` - Alternative UI
- `run_web_dashboard.py` - Dashboard launcher

### **Analysis Tools:**
- `aqs_research_framework.py` - Research framework
- `efficiency_benchmark.py` - Performance benchmarking
- `resource_profiler.py` - Resource monitoring
- `model_analyzer.py` - Model analysis tools

## 📁 **Directories**

### **Models & Data:**
- `checkpoints/` - Trained model files
  - `best_f1_checkpoint.pth`
  - `best_loss_checkpoint.pth`
  - `latest_checkpoint.pth`

### **Test Data:**
- `test_videos/` - Sample videos for testing
  - `5192-183786490_tiny.mp4`

### **Environment:**
- `venv/` - Python virtual environment
- `__pycache__/` - Python cache files
- `.gradio/` - Gradio cache (if used)

## 📄 **Documentation**

### **Main Documentation:**
- `AQS_SYSTEM_DOCUMENTATION.md` - **Complete system documentation**
- `README.md` - Project overview and setup
- `USAGE.md` - Usage instructions
- `requirements.txt` - Python dependencies

### **Project Files:**
- `PROJECT_STRUCTURE.md` - This file
- `install_dependencies.py` - Dependency installer

## 🚀 **Quick Start**

### **1. Setup Environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### **2. Launch Web Dashboard:**
```bash
python run_web_dashboard.py
# OR
streamlit run aqs_web_dashboard.py
```

### **3. Command Line Usage:**
```bash
python video_inference_demo.py --video test_videos/sample.mp4
```

## 🎯 **Key Features**

- ✅ **Real-time video quality analysis**
- ✅ **10 artifact types detection**
- ✅ **Interactive web dashboard**
- ✅ **Blur-specific validation**
- ✅ **Perceptual quality scoring**
- ✅ **Export capabilities**

## 📊 **Model Information**

- **Architecture**: EfficientNet-B0 + Temporal Processing
- **Input**: 7-frame sequences (224×224)
- **Output**: AQS scores (0-1) + artifact breakdown
- **Performance**: ~45ms per sequence (GPU)

## 🔧 **Configuration**

- **Quality Thresholds**: 0.85 (Excellent), 0.70 (Good), 0.50 (Fair), 0.30 (Poor)
- **Detection Threshold**: 0.2 (artifact probability)
- **Blur Validation**: Enhanced with image processing
- **Penalties**: Artifact-specific (1.2x - 2.2x)

---

**Total Files**: ~15 core files + documentation
**Total Size**: ~500MB (including models and venv)
**Status**: Production ready ✅
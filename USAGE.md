# Usage Guide - EfficientNet Computational Efficiency Analysis

This document provides detailed usage instructions for the EfficientNet computational efficiency analysis framework.

## Quick Start

### 1. Setup Verification
First, verify your setup is correct:

```bash
python test_setup.py
```

### 2. Complete Analysis (Recommended)
Run the full efficiency analysis:

```bash
python run_efficiency_analysis.py --analysis_type all
```

### 3. Quick Analysis
For faster results with fewer iterations:

```bash
python run_efficiency_analysis.py --quick
```

## Detailed Usage

### Individual Analysis Components

#### Performance Benchmarking Only
```bash
python run_efficiency_analysis.py --analysis_type benchmark --device cuda
```

#### Resource Profiling Only
```bash
python run_efficiency_analysis.py --analysis_type profiling --device auto
```

#### Architecture Analysis Only
```bash
python run_efficiency_analysis.py --analysis_type architecture
```

### Command Line Options

```bash
python run_efficiency_analysis.py [OPTIONS]

Options:
  --analysis_type {all,benchmark,profiling,architecture}
                        Type of analysis to run (default: all)
  --output_dir OUTPUT_DIR
                        Output directory for results (default: efficiency_analysis_results)
  --device {auto,cpu,cuda}
                        Device to use for analysis (default: auto)
  --quick               Run quick analysis with fewer iterations
```

### Direct Module Usage

#### Efficiency Benchmarking
```python
from efficiency_benchmark import EfficiencyBenchmark

benchmark = EfficiencyBenchmark(device='cuda', warmup_runs=5, benchmark_runs=50)
results = benchmark.benchmark_all_variants(sequence_length=7, batch_size=1)
benchmark.save_results('benchmark_output')
benchmark.create_visualizations('benchmark_output')
```

#### Resource Profiling
```python
from resource_profiler import AdvancedProfiler
from efficiency_benchmark import EfficientNetVariants

profiler = AdvancedProfiler(device='cuda')
model = EfficientNetVariants.create_model('efficientnet_b0')
input_tensor = torch.randn(1, 7, 3, 224, 224)

result = profiler.profile_model(model, input_tensor, 'efficientnet_b0', num_runs=20)
```

#### Architecture Analysis
```python
from model_analyzer import EfficientNetAnalyzer

analyzer = EfficientNetAnalyzer()
variants = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
comparison = analyzer.compare_architectures(variants)
analyzer.create_analysis_report(comparison, 'architecture_output')
```

## Configuration Options

### Benchmark Configuration
- `warmup_runs`: Number of warmup iterations (default: 10)
- `benchmark_runs`: Number of benchmark iterations (default: 100)
- `sequence_length`: Video sequence length (default: 7)
- `batch_size`: Batch size for analysis (default: 1)

### Profiling Configuration
- `sampling_interval`: Resource monitoring interval (default: 0.1s)
- `num_runs`: Number of profiling runs (default: 10)
- Device selection: 'auto', 'cpu', or 'cuda'

### Analysis Variants
The framework analyzes these EfficientNet variants by default:
- EfficientNet-B0 (224x224)
- EfficientNet-B1 (240x240)
- EfficientNet-B2 (260x260)
- EfficientNet-B3 (300x300)
- EfficientNet-B4 (380x380)
- EfficientNet-B5 (456x456)
- EfficientNet-B6 (528x528)
- EfficientNet-B7 (600x600)

## Output Structure

After running analysis, results are organized as:

```
efficiency_analysis_results_YYYYMMDD_HHMMSS/
├── benchmark_results/
│   ├── benchmark_results.json          # Raw benchmark data
│   ├── inference_vs_size.png          # Speed vs size trade-off
│   ├── throughput_comparison.png      # FPS comparison
│   ├── memory_analysis.png            # Memory usage analysis
│   ├── efficiency_frontier.png        # Efficiency frontier plot
│   └── comparison_table.png           # Comprehensive comparison
├── profiling_results/
│   ├── profiling_report.json          # Detailed profiling data
│   ├── execution_analysis.png         # CPU vs GPU execution
│   ├── memory_comparison.png          # Memory usage comparison
│   └── profiling_summary.txt          # Text summary
├── architecture_analysis/
│   ├── architecture_analysis.json     # Architecture data
│   ├── parameter_flop_comparison.png  # Parameters vs FLOPs
│   ├── scaling_analysis.png           # Compound scaling analysis
│   ├── efficiency_ranking.png         # Efficiency ranking
│   ├── comparison_table.png           # Architecture comparison
│   └── architecture_summary.txt       # Text summary
├── comprehensive_efficiency_report.md  # Complete research report
└── analysis_metadata.json             # Analysis metadata
```

## Research Applications

### For Academic Papers
The framework generates publication-ready materials:

1. **Performance Tables**: LaTeX-ready comparison tables
2. **Efficiency Plots**: High-resolution charts for papers
3. **Statistical Analysis**: Comprehensive metrics and rankings
4. **Methodology Documentation**: Detailed analysis procedures

### For Industry Applications
- **Model Selection**: Data-driven model choice for deployment
- **Resource Planning**: Hardware requirement estimation
- **Cost Analysis**: Computational cost vs performance trade-offs
- **Optimization Guidance**: Specific recommendations for improvement

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.7+

### Recommended for Complete Analysis
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **RAM**: 16GB+ system memory
- **CPU**: 8+ cores for parallel processing
- **Storage**: 5GB+ free space for comprehensive results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python run_efficiency_analysis.py --device cpu
   
   # Or run individual components
   python run_efficiency_analysis.py --analysis_type benchmark
   ```

2. **Missing Dependencies**
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   
   # Install optional packages
   pip install thop torchsummary pynvml
   ```

3. **Slow Analysis**
   ```bash
   # Use quick mode
   python run_efficiency_analysis.py --quick
   
   # Or run specific analysis only
   python run_efficiency_analysis.py --analysis_type architecture
   ```

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed for GPU acceleration
- **Memory Management**: Close other applications to free up RAM
- **Batch Processing**: Adjust batch size based on available memory
- **Parallel Processing**: Use multi-core CPU for faster analysis

## Advanced Usage

### Custom Model Analysis
```python
from efficiency_benchmark import EfficiencyBenchmark

# Create custom model
import torch.nn as nn
custom_model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# Benchmark custom model
benchmark = EfficiencyBenchmark()
result = benchmark.benchmark_model(
    custom_model, 
    (1, 3, 224, 224), 
    'custom_model'
)
```

### Integration with Other Frameworks
The analysis modules can be integrated with:
- **MLflow**: For experiment tracking
- **Weights & Biases**: For visualization
- **TensorBoard**: For monitoring
- **Optuna**: For hyperparameter optimization

### Extending the Framework
To add new analysis metrics:
1. Extend the `BenchmarkResult` dataclass
2. Modify the benchmarking logic
3. Update visualization functions
4. Add new analysis methods

## Best Practices

1. **Consistent Environment**: Use the same hardware/software setup for comparisons
2. **Multiple Runs**: Run analysis multiple times for statistical significance
3. **Resource Monitoring**: Monitor system resources during analysis
4. **Documentation**: Keep detailed records of analysis parameters
5. **Version Control**: Track analysis code and results versions
"""
Computational Efficiency Benchmark for EfficientNet Architectures
================================================================

This module provides comprehensive benchmarking tools for analyzing the computational
efficiency of different EfficientNet variants for video quality assessment.

Key metrics analyzed:
- Inference time per frame/sequence
- Memory usage (GPU/CPU)
- FLOPs (Floating Point Operations)
- Model size and parameters
- Throughput (frames per second)
- Energy consumption estimation
"""

import torch
import torch.nn as nn
import torchvision.models as models
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import gc

# Try to import additional profiling tools
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with 'pip install thop' for FLOP counting.")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU memory monitoring limited.")

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_name: str
    input_shape: Tuple[int, ...]
    inference_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    throughput_fps: float
    model_size_mb: float
    parameters: int
    flops: Optional[int]
    energy_estimate_mj: Optional[float]

class MemoryMonitor:
    """Monitor memory usage during inference"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self.monitoring = True
        self.peak_memory = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage in MB"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.peak_memory
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            if self.device == 'cuda' and torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                self.peak_memory = max(self.peak_memory, current_memory)
            else:
                # Monitor CPU memory
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024**2  # MB
                self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.01)  # Check every 10ms

class EfficientNetVariants:
    """Factory for creating different EfficientNet variants"""
    
    @staticmethod
    def get_available_variants() -> List[str]:
        """Get list of available EfficientNet variants"""
        return ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                'efficientnet_b6', 'efficientnet_b7']
    
    @staticmethod
    def create_model(variant: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
        """Create EfficientNet model variant"""
        if variant == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        elif variant == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=pretrained)
        elif variant == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=pretrained)
        elif variant == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
        elif variant == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=pretrained)
        elif variant == 'efficientnet_b5':
            model = models.efficientnet_b5(pretrained=pretrained)
        elif variant == 'efficientnet_b6':
            model = models.efficientnet_b6(pretrained=pretrained)
        elif variant == 'efficientnet_b7':
            model = models.efficientnet_b7(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Modify classifier for our task
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
        return model
    
    @staticmethod
    def get_input_size(variant: str) -> Tuple[int, int]:
        """Get recommended input size for variant"""
        input_sizes = {
            'efficientnet_b0': (224, 224),
            'efficientnet_b1': (240, 240),
            'efficientnet_b2': (260, 260),
            'efficientnet_b3': (300, 300),
            'efficientnet_b4': (380, 380),
            'efficientnet_b5': (456, 456),
            'efficientnet_b6': (528, 528),
            'efficientnet_b7': (600, 600)
        }
        return input_sizes.get(variant, (224, 224))

class EfficiencyBenchmark:
    """Main benchmarking class for EfficientNet variants"""
    
    def __init__(self, device: str = 'auto', warmup_runs: int = 10, benchmark_runs: int = 100):
        self.device = self._get_device(device)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       model_name: str) -> BenchmarkResult:
        """Benchmark a single model"""
        print(f"Benchmarking {model_name} with input shape {input_shape}")
        
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        print("  Warming up...")
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)
        
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Benchmark inference time
        print("  Measuring inference time...")
        memory_monitor = MemoryMonitor(self.device.type)
        
        memory_monitor.start_monitoring()
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                _ = model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        peak_memory = memory_monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_inference_time = (total_time / self.benchmark_runs) * 1000  # ms
        throughput = self.benchmark_runs / total_time  # fps
        
        # Model size and parameters
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        parameters = sum(p.numel() for p in model.parameters())
        
        # FLOPs calculation
        flops = None
        if THOP_AVAILABLE:
            try:
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            except Exception as e:
                print(f"    Warning: FLOP calculation failed: {e}")
        
        # GPU memory usage
        gpu_memory_mb = None
        if self.device.type == 'cuda' and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()
        
        # Energy estimation (rough approximation)
        energy_estimate = self._estimate_energy_consumption(
            avg_inference_time, parameters, self.device.type
        )
        
        result = BenchmarkResult(
            model_name=model_name,
            input_shape=input_shape,
            inference_time_ms=avg_inference_time,
            memory_usage_mb=peak_memory,
            gpu_memory_mb=gpu_memory_mb,
            throughput_fps=throughput,
            model_size_mb=model_size_mb,
            parameters=parameters,
            flops=flops,
            energy_estimate_mj=energy_estimate
        )
        
        print(f"  Results: {avg_inference_time:.2f}ms, {throughput:.1f}fps, {model_size_mb:.1f}MB")
        return result
    
    def _estimate_energy_consumption(self, inference_time_ms: float, 
                                   parameters: int, device_type: str) -> float:
        """Rough energy consumption estimate in millijoules"""
        # Very rough estimates based on typical hardware
        if device_type == 'cuda':
            # GPU: ~200W typical, efficiency varies with model size
            base_power_w = 200
            efficiency_factor = min(1.0, parameters / 1e6)  # Larger models less efficient
            power_w = base_power_w * (0.3 + 0.7 * efficiency_factor)
        else:
            # CPU: ~50W typical
            power_w = 50
        
        # Energy = Power * Time
        energy_j = power_w * (inference_time_ms / 1000)
        return energy_j * 1000  # Convert to millijoules
    
    def benchmark_all_variants(self, sequence_length: int = 7, 
                             batch_size: int = 1) -> List[BenchmarkResult]:
        """Benchmark all EfficientNet variants"""
        print("Starting comprehensive EfficientNet efficiency benchmark")
        print("=" * 60)
        
        variants = EfficientNetVariants.get_available_variants()
        results = []
        
        for variant in variants:
            try:
                # Create model
                model = EfficientNetVariants.create_model(variant)
                
                # Get input size
                h, w = EfficientNetVariants.get_input_size(variant)
                input_shape = (batch_size, sequence_length, 3, h, w)
                
                # Benchmark
                result = self.benchmark_model(model, input_shape, variant)
                results.append(result)
                
                # Clean up
                del model
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error benchmarking {variant}: {e}")
                continue
        
        self.results = results
        return results
    
    def save_results(self, output_dir: str = "benchmark_results"):
        """Save benchmark results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to JSON-serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                'model_name': result.model_name,
                'input_shape': result.input_shape,
                'inference_time_ms': result.inference_time_ms,
                'memory_usage_mb': result.memory_usage_mb,
                'gpu_memory_mb': result.gpu_memory_mb,
                'throughput_fps': result.throughput_fps,
                'model_size_mb': result.model_size_mb,
                'parameters': result.parameters,
                'flops': result.flops,
                'energy_estimate_mj': result.energy_estimate_mj
            }
            results_data.append(result_dict)
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'warmup_runs': self.warmup_runs,
                'benchmark_runs': self.benchmark_runs,
                'results': results_data
            }, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def create_visualizations(self, output_dir: str = "benchmark_results"):
        """Create comprehensive visualizations"""
        if not self.results:
            print("No results to visualize")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Inference Time vs Model Size
        self._plot_inference_vs_size(output_dir)
        
        # 2. Throughput comparison
        self._plot_throughput_comparison(output_dir)
        
        # 3. Memory usage analysis
        self._plot_memory_analysis(output_dir)
        
        # 4. Efficiency frontier
        self._plot_efficiency_frontier(output_dir)
        
        # 5. Comprehensive comparison table
        self._create_comparison_table(output_dir)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _plot_inference_vs_size(self, output_dir: str):
        """Plot inference time vs model size"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = [r.model_name for r in self.results]
        inference_times = [r.inference_time_ms for r in self.results]
        model_sizes = [r.model_size_mb for r in self.results]
        
        scatter = ax.scatter(model_sizes, inference_times, 
                           s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        
        # Add labels for each point
        for i, name in enumerate(model_names):
            ax.annotate(name.replace('efficientnet_', 'B'), 
                       (model_sizes[i], inference_times[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Time vs Model Size Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_vs_size.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_comparison(self, output_dir: str):
        """Plot throughput comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = [r.model_name.replace('efficientnet_', 'B') for r in self.results]
        throughputs = [r.throughput_fps for r in self.results]
        
        bars = ax.bar(model_names, throughputs, alpha=0.7, color='skyblue')
        
        # Add value labels on bars
        for bar, throughput in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   f'{throughput:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('EfficientNet Variant')
        ax.set_ylabel('Throughput (FPS)')
        ax.set_title('Video Processing Throughput Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_analysis(self, output_dir: str):
        """Plot memory usage analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = [r.model_name.replace('efficientnet_', 'B') for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]
        gpu_memory = [r.gpu_memory_mb or 0 for r in self.results]
        
        # CPU Memory
        bars1 = ax1.bar(model_names, memory_usage, alpha=0.7, color='lightcoral')
        ax1.set_xlabel('EfficientNet Variant')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('CPU Memory Usage')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, mem in zip(bars1, memory_usage):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{mem:.0f}', ha='center', va='bottom', fontsize=9)
        
        # GPU Memory (if available)
        if any(gpu_memory):
            bars2 = ax2.bar(model_names, gpu_memory, alpha=0.7, color='lightgreen')
            ax2.set_xlabel('EfficientNet Variant')
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.set_title('GPU Memory Usage')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, mem in zip(bars2, gpu_memory):
                if mem > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{mem:.0f}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'GPU Memory\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_frontier(self, output_dir: str):
        """Plot efficiency frontier (accuracy vs computational cost)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use inverse of inference time as efficiency proxy
        efficiency_scores = [1000 / r.inference_time_ms for r in self.results]
        model_sizes = [r.model_size_mb for r in self.results]
        model_names = [r.model_name for r in self.results]
        
        # Create bubble chart
        scatter = ax.scatter(efficiency_scores, model_sizes, 
                           s=[r.parameters/10000 for r in self.results],  # Bubble size = parameters
                           alpha=0.6, c=range(len(model_names)), cmap='plasma')
        
        # Add labels
        for i, name in enumerate(model_names):
            ax.annotate(name.replace('efficientnet_', 'B'), 
                       (efficiency_scores[i], model_sizes[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Efficiency Score (1000/inference_time_ms)')
        ax.set_ylabel('Model Size (MB)')
        ax.set_title('Efficiency Frontier: Speed vs Size\n(Bubble size = Parameters)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Model Variant')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_frontier.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_table(self, output_dir: str):
        """Create comprehensive comparison table"""
        import pandas as pd
        
        # Prepare data
        data = []
        for result in self.results:
            row = {
                'Model': result.model_name.replace('efficientnet_', 'EfficientNet-'),
                'Parameters (M)': f"{result.parameters / 1e6:.1f}",
                'Model Size (MB)': f"{result.model_size_mb:.1f}",
                'Inference Time (ms)': f"{result.inference_time_ms:.2f}",
                'Throughput (FPS)': f"{result.throughput_fps:.1f}",
                'Memory (MB)': f"{result.memory_usage_mb:.0f}",
                'FLOPs (G)': f"{result.flops / 1e9:.1f}" if result.flops else "N/A",
                'Energy (mJ)': f"{result.energy_estimate_mj:.1f}" if result.energy_estimate_mj else "N/A"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'efficiency_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Create formatted table visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('EfficientNet Variants: Comprehensive Efficiency Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison table saved as CSV: {csv_path}")

def main():
    """Main benchmarking function"""
    print("EfficientNet Computational Efficiency Analysis")
    print("=" * 50)
    
    # Create benchmark instance
    benchmark = EfficiencyBenchmark(
        device='auto',
        warmup_runs=5,
        benchmark_runs=50
    )
    
    # Run comprehensive benchmark
    results = benchmark.benchmark_all_variants(
        sequence_length=7,
        batch_size=1
    )
    
    # Save results and create visualizations
    benchmark.save_results()
    benchmark.create_visualizations()
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 30)
    for result in results:
        print(f"{result.model_name}: {result.inference_time_ms:.2f}ms, "
              f"{result.throughput_fps:.1f}fps, {result.model_size_mb:.1f}MB")

if __name__ == "__main__":
    main()
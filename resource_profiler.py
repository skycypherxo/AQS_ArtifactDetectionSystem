"""
Resource Profiler for EfficientNet Video Quality Assessment
==========================================================

Advanced profiling tools for analyzing resource consumption patterns,
bottlenecks, and optimization opportunities in EfficientNet-based
video quality assessment systems.
"""

import torch
import torch.nn as nn
import torch.profiler
import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import gc
import tracemalloc

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_temperature: Optional[float] = None

@dataclass
class ProfilingResult:
    """Complete profiling result for a model"""
    model_name: str
    total_time_ms: float
    cpu_time_ms: float
    gpu_time_ms: Optional[float]
    memory_peak_mb: float
    gpu_memory_peak_mb: Optional[float]
    cpu_utilization_avg: float
    gpu_utilization_avg: Optional[float]
    bottleneck_analysis: Dict[str, Any]
    layer_timings: Dict[str, float]
    optimization_suggestions: List[str]

class SystemResourceMonitor:
    """Monitors system resources during model execution"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.snapshots.clear()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> List[ResourceSnapshot]:
        """Stop monitoring and return snapshots"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.snapshots.copy()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a single resource snapshot"""
        timestamp = time.time()
        
        # CPU and system memory
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / 1024**2
        
        # GPU metrics
        gpu_memory_mb = None
        gpu_utilization = None
        gpu_temperature = None
        
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_mb = mem_info.used / 1024**2
                
                # GPU utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util_info.gpu
                
                # GPU temperature
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
            except Exception as e:
                pass  # Silently handle NVML errors
        
        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature
        )

class LayerProfiler:
    """Profiles individual layer execution times"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_times = {}
        self.hooks = []
        
    def start_profiling(self):
        """Start layer-wise profiling"""
        self.layer_times.clear()
        self._register_hooks()
        
    def stop_profiling(self):
        """Stop profiling and remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_layer_times(self) -> Dict[str, float]:
        """Get layer execution times in milliseconds"""
        return self.layer_times.copy()
    
    def _register_hooks(self):
        """Register forward hooks for timing"""
        def create_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                # This is called after forward pass, so we need to measure differently
                # We'll use a different approach with profiler
                pass
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)

class AdvancedProfiler:
    """Advanced profiler combining multiple profiling techniques"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.resource_monitor = SystemResourceMonitor()
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def profile_model(self, model: nn.Module, input_tensor: torch.Tensor, 
                     model_name: str, num_runs: int = 10) -> ProfilingResult:
        """Comprehensive model profiling"""
        print(f"Profiling {model_name}...")
        
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Memory tracking
        tracemalloc.start()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # Profiling with PyTorch profiler
        layer_timings = {}
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA if self.device.type == 'cuda' else None
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(input_tensor)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
        
        # Stop monitoring
        snapshots = self.resource_monitor.stop_monitoring()
        
        # Memory stats
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_memory_peak = None
        if self.device.type == 'cuda':
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        # Analyze profiler results
        cpu_time_ms, gpu_time_ms = self._analyze_profiler_results(prof)
        
        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        avg_cpu_util = np.mean([s.cpu_percent for s in snapshots]) if snapshots else 0
        avg_gpu_util = None
        if snapshots and snapshots[0].gpu_utilization is not None:
            gpu_utils = [s.gpu_utilization for s in snapshots if s.gpu_utilization is not None]
            avg_gpu_util = np.mean(gpu_utils) if gpu_utils else None
        
        # Bottleneck analysis
        bottleneck_analysis = self._analyze_bottlenecks(
            prof, snapshots, total_time_ms, cpu_time_ms, gpu_time_ms
        )
        
        # Optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            bottleneck_analysis, avg_cpu_util, avg_gpu_util, model_name
        )
        
        return ProfilingResult(
            model_name=model_name,
            total_time_ms=total_time_ms,
            cpu_time_ms=cpu_time_ms,
            gpu_time_ms=gpu_time_ms,
            memory_peak_mb=peak_memory / 1024**2,
            gpu_memory_peak_mb=gpu_memory_peak,
            cpu_utilization_avg=avg_cpu_util,
            gpu_utilization_avg=avg_gpu_util,
            bottleneck_analysis=bottleneck_analysis,
            layer_timings=layer_timings,
            optimization_suggestions=optimization_suggestions
        )
    
    def _analyze_profiler_results(self, prof) -> Tuple[float, Optional[float]]:
        """Analyze PyTorch profiler results"""
        cpu_time_ms = 0
        gpu_time_ms = 0
        
        # Get key averages
        key_averages = prof.key_averages()
        
        for event in key_averages:
            if event.device_type == torch.profiler.DeviceType.CPU:
                cpu_time_ms += event.cpu_time_total / 1000  # Convert to ms
            elif event.device_type == torch.profiler.DeviceType.CUDA:
                gpu_time_ms += event.cuda_time_total / 1000  # Convert to ms
        
        return cpu_time_ms, gpu_time_ms if gpu_time_ms > 0 else None
    
    def _analyze_bottlenecks(self, prof, snapshots: List[ResourceSnapshot], 
                           total_time_ms: float, cpu_time_ms: float, 
                           gpu_time_ms: Optional[float]) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        analysis = {
            'primary_bottleneck': 'unknown',
            'cpu_bound': False,
            'memory_bound': False,
            'gpu_bound': False,
            'io_bound': False,
            'details': {}
        }
        
        # CPU vs GPU time analysis
        if gpu_time_ms is not None:
            cpu_ratio = cpu_time_ms / total_time_ms
            gpu_ratio = gpu_time_ms / total_time_ms
            
            if cpu_ratio > 0.7:
                analysis['cpu_bound'] = True
                analysis['primary_bottleneck'] = 'cpu'
            elif gpu_ratio > 0.7:
                analysis['gpu_bound'] = True
                analysis['primary_bottleneck'] = 'gpu'
        else:
            analysis['cpu_bound'] = True
            analysis['primary_bottleneck'] = 'cpu'
        
        # Memory analysis
        if snapshots:
            memory_usage = [s.memory_mb for s in snapshots]
            if max(memory_usage) > 8000:  # > 8GB
                analysis['memory_bound'] = True
                if analysis['primary_bottleneck'] == 'unknown':
                    analysis['primary_bottleneck'] = 'memory'
        
        # Detailed analysis from profiler
        key_averages = prof.key_averages()
        top_operations = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:5]
        
        analysis['details']['top_cpu_operations'] = [
            {
                'name': op.key,
                'cpu_time_ms': op.cpu_time_total / 1000,
                'calls': op.count
            }
            for op in top_operations
        ]
        
        return analysis
    
    def _generate_optimization_suggestions(self, bottleneck_analysis: Dict[str, Any],
                                         avg_cpu_util: float, avg_gpu_util: Optional[float],
                                         model_name: str) -> List[str]:
        """Generate optimization suggestions based on profiling results"""
        suggestions = []
        
        # Based on bottleneck analysis
        if bottleneck_analysis['cpu_bound']:
            suggestions.append("Consider using GPU acceleration or optimizing CPU operations")
            suggestions.append("Enable mixed precision training to reduce CPU overhead")
            
        if bottleneck_analysis['memory_bound']:
            suggestions.append("Reduce batch size or use gradient checkpointing")
            suggestions.append("Consider model pruning or quantization")
            
        if bottleneck_analysis['gpu_bound']:
            suggestions.append("Optimize GPU kernel usage or reduce model complexity")
            suggestions.append("Consider using TensorRT or other GPU optimization frameworks")
        
        # Based on utilization
        if avg_cpu_util < 50:
            suggestions.append("CPU underutilized - consider increasing batch size")
            
        if avg_gpu_util is not None and avg_gpu_util < 70:
            suggestions.append("GPU underutilized - consider optimizing data loading or increasing batch size")
        
        # Model-specific suggestions
        if 'b7' in model_name.lower():
            suggestions.append("EfficientNet-B7 is very large - consider using smaller variants for better efficiency")
        elif 'b0' in model_name.lower():
            suggestions.append("EfficientNet-B0 is efficient but may benefit from batch processing")
        
        return suggestions
    
    def create_profiling_report(self, results: List[ProfilingResult], 
                              output_dir: str = "profiling_results"):
        """Create comprehensive profiling report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_data = [asdict(result) for result in results]
        
        report_file = os.path.join(output_dir, 'profiling_report.json')
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'results': results_data
            }, f, indent=2)
        
        # Create visualizations
        self._create_profiling_visualizations(results, output_dir)
        
        # Generate text report
        self._generate_text_report(results, output_dir)
        
        print(f"Profiling report saved to {output_dir}")
    
    def _create_profiling_visualizations(self, results: List[ProfilingResult], output_dir: str):
        """Create profiling visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Execution time breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = [r.model_name.replace('efficientnet_', 'B') for r in results]
        cpu_times = [r.cpu_time_ms for r in results]
        gpu_times = [r.gpu_time_ms or 0 for r in results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, cpu_times, width, label='CPU Time', alpha=0.7)
        ax1.bar(x + width/2, gpu_times, width, label='GPU Time', alpha=0.7)
        
        ax1.set_xlabel('Model Variant')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('CPU vs GPU Execution Time')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Resource utilization
        cpu_utils = [r.cpu_utilization_avg for r in results]
        gpu_utils = [r.gpu_utilization_avg or 0 for r in results]
        
        ax2.bar(x - width/2, cpu_utils, width, label='CPU Utilization', alpha=0.7)
        ax2.bar(x + width/2, gpu_utils, width, label='GPU Utilization', alpha=0.7)
        
        ax2.set_xlabel('Model Variant')
        ax2.set_ylabel('Utilization (%)')
        ax2.set_title('Resource Utilization')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'execution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Memory usage comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        memory_peaks = [r.memory_peak_mb for r in results]
        gpu_memory_peaks = [r.gpu_memory_peak_mb or 0 for r in results]
        
        x = np.arange(len(model_names))
        
        bars1 = ax.bar(x - width/2, memory_peaks, width, label='CPU Memory', alpha=0.7, color='lightblue')
        bars2 = ax.bar(x + width/2, gpu_memory_peaks, width, label='GPU Memory', alpha=0.7, color='lightgreen')
        
        # Add value labels
        for bar, value in zip(bars1, memory_peaks):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        for bar, value in zip(bars2, gpu_memory_peaks):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Peak Memory Usage (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results: List[ProfilingResult], output_dir: str):
        """Generate detailed text report"""
        report_path = os.path.join(output_dir, 'profiling_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("EfficientNet Computational Profiling Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            for result in results:
                f.write(f"Model: {result.model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Execution Time: {result.total_time_ms:.2f} ms\n")
                f.write(f"CPU Time: {result.cpu_time_ms:.2f} ms\n")
                if result.gpu_time_ms:
                    f.write(f"GPU Time: {result.gpu_time_ms:.2f} ms\n")
                f.write(f"Peak Memory: {result.memory_peak_mb:.1f} MB\n")
                if result.gpu_memory_peak_mb:
                    f.write(f"Peak GPU Memory: {result.gpu_memory_peak_mb:.1f} MB\n")
                f.write(f"CPU Utilization: {result.cpu_utilization_avg:.1f}%\n")
                if result.gpu_utilization_avg:
                    f.write(f"GPU Utilization: {result.gpu_utilization_avg:.1f}%\n")
                
                f.write(f"\nBottleneck Analysis:\n")
                f.write(f"  Primary Bottleneck: {result.bottleneck_analysis['primary_bottleneck']}\n")
                f.write(f"  CPU Bound: {result.bottleneck_analysis['cpu_bound']}\n")
                f.write(f"  Memory Bound: {result.bottleneck_analysis['memory_bound']}\n")
                f.write(f"  GPU Bound: {result.bottleneck_analysis['gpu_bound']}\n")
                
                f.write(f"\nOptimization Suggestions:\n")
                for suggestion in result.optimization_suggestions:
                    f.write(f"  - {suggestion}\n")
                
                f.write("\n" + "="*50 + "\n\n")

def main():
    """Main profiling function"""
    from efficiency_benchmark import EfficientNetVariants
    
    print("Advanced EfficientNet Resource Profiling")
    print("=" * 40)
    
    profiler = AdvancedProfiler()
    results = []
    
    # Profile a subset of variants for detailed analysis
    variants_to_profile = ['efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4']
    
    for variant in variants_to_profile:
        try:
            # Create model
            model = EfficientNetVariants.create_model(variant)
            
            # Create input
            h, w = EfficientNetVariants.get_input_size(variant)
            input_tensor = torch.randn(1, 7, 3, h, w)  # Batch=1, Sequence=7
            
            # Profile
            result = profiler.profile_model(model, input_tensor, variant, num_runs=20)
            results.append(result)
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"Error profiling {variant}: {e}")
    
    # Generate report
    profiler.create_profiling_report(results)
    
    print("Profiling completed. Check 'profiling_results' directory for detailed analysis.")

if __name__ == "__main__":
    main()
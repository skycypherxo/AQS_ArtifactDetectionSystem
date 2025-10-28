"""
Model Architecture Analyzer for EfficientNet Variants
====================================================

Analyzes and compares the architectural characteristics of different
EfficientNet variants to understand their computational trade-offs.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from dataclasses import dataclass, asdict
import pandas as pd

try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False
    print("Warning: torchsummary not available. Install with 'pip install torchsummary'")

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

@dataclass
class LayerInfo:
    """Information about a single layer"""
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: int
    flops: Optional[int] = None
    memory_mb: Optional[float] = None

@dataclass
class ModelArchitecture:
    """Complete model architecture analysis"""
    model_name: str
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    total_flops: Optional[int]
    layers: List[LayerInfo]
    depth: int
    width_multiplier: float
    resolution: Tuple[int, int]
    compound_scaling: Dict[str, float]
    efficiency_metrics: Dict[str, float]

class EfficientNetAnalyzer:
    """Analyzes EfficientNet model architectures"""
    
    def __init__(self):
        self.scaling_coefficients = {
            'efficientnet_b0': {'depth': 1.0, 'width': 1.0, 'resolution': 224},
            'efficientnet_b1': {'depth': 1.1, 'width': 1.0, 'resolution': 240},
            'efficientnet_b2': {'depth': 1.2, 'width': 1.1, 'resolution': 260},
            'efficientnet_b3': {'depth': 1.4, 'width': 1.2, 'resolution': 300},
            'efficientnet_b4': {'depth': 1.8, 'width': 1.4, 'resolution': 380},
            'efficientnet_b5': {'depth': 2.2, 'width': 1.6, 'resolution': 456},
            'efficientnet_b6': {'depth': 2.6, 'width': 1.8, 'resolution': 528},
            'efficientnet_b7': {'depth': 3.1, 'width': 2.0, 'resolution': 600}
        }
    
    def analyze_model(self, model_name: str, input_shape: Tuple[int, ...] = None) -> ModelArchitecture:
        """Analyze a single EfficientNet model"""
        print(f"Analyzing {model_name}...")
        
        # Create model
        model = self._create_model(model_name)
        
        # Get input shape
        if input_shape is None:
            resolution = self.scaling_coefficients[model_name]['resolution']
            input_shape = (1, 3, resolution, resolution)
        
        # Basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        # Layer analysis
        layers = self._analyze_layers(model, input_shape)
        
        # FLOP analysis
        total_flops = None
        if THOP_AVAILABLE:
            try:
                dummy_input = torch.randn(input_shape)
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
                total_flops = flops
            except Exception as e:
                print(f"  Warning: FLOP calculation failed: {e}")
        
        # Scaling analysis
        scaling_info = self.scaling_coefficients.get(model_name, {})
        depth_mult = scaling_info.get('depth', 1.0)
        width_mult = scaling_info.get('width', 1.0)
        resolution = scaling_info.get('resolution', 224)
        
        # Efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            total_params, total_flops, model_size_mb, resolution
        )
        
        return ModelArchitecture(
            model_name=model_name,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            total_flops=total_flops,
            layers=layers,
            depth=len([l for l in layers if 'conv' in l.layer_type.lower()]),
            width_multiplier=width_mult,
            resolution=(resolution, resolution),
            compound_scaling={
                'depth': depth_mult,
                'width': width_mult,
                'resolution': resolution
            },
            efficiency_metrics=efficiency_metrics
        )
    
    def _create_model(self, model_name: str) -> nn.Module:
        """Create EfficientNet model"""
        if model_name == 'efficientnet_b0':
            return models.efficientnet_b0(pretrained=False)
        elif model_name == 'efficientnet_b1':
            return models.efficientnet_b1(pretrained=False)
        elif model_name == 'efficientnet_b2':
            return models.efficientnet_b2(pretrained=False)
        elif model_name == 'efficientnet_b3':
            return models.efficientnet_b3(pretrained=False)
        elif model_name == 'efficientnet_b4':
            return models.efficientnet_b4(pretrained=False)
        elif model_name == 'efficientnet_b5':
            return models.efficientnet_b5(pretrained=False)
        elif model_name == 'efficientnet_b6':
            return models.efficientnet_b6(pretrained=False)
        elif model_name == 'efficientnet_b7':
            return models.efficientnet_b7(pretrained=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _analyze_layers(self, model: nn.Module, input_shape: Tuple[int, ...]) -> List[LayerInfo]:
        """Analyze individual layers"""
        layers = []
        
        # Hook to capture layer information
        layer_info = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    input_shape = input[0].shape if input else None
                else:
                    input_shape = input.shape if input is not None else None
                
                if isinstance(output, tuple):
                    output_shape = output[0].shape if output else None
                else:
                    output_shape = output.shape if output is not None else None
                
                params = sum(p.numel() for p in module.parameters())
                
                layer_info[name] = {
                    'input_shape': tuple(input_shape) if input_shape is not None else None,
                    'output_shape': tuple(output_shape) if output_shape is not None else None,
                    'parameters': params,
                    'layer_type': type(module).__name__
                }
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to collect info
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to LayerInfo objects
        for name, info in layer_info.items():
            if info['parameters'] > 0 or 'conv' in info['layer_type'].lower():
                layer = LayerInfo(
                    name=name,
                    layer_type=info['layer_type'],
                    input_shape=info['input_shape'] or (),
                    output_shape=info['output_shape'] or (),
                    parameters=info['parameters']
                )
                layers.append(layer)
        
        return layers
    
    def _calculate_efficiency_metrics(self, params: int, flops: Optional[int], 
                                    size_mb: float, resolution: int) -> Dict[str, float]:
        """Calculate various efficiency metrics"""
        metrics = {
            'params_per_mb': params / size_mb if size_mb > 0 else 0,
            'params_per_pixel': params / (resolution * resolution) if resolution > 0 else 0,
        }
        
        if flops is not None:
            metrics.update({
                'flops_per_param': flops / params if params > 0 else 0,
                'flops_per_mb': flops / size_mb if size_mb > 0 else 0,
                'flops_per_pixel': flops / (resolution * resolution) if resolution > 0 else 0
            })
        
        return metrics
    
    def compare_architectures(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple EfficientNet architectures"""
        print("Comparing EfficientNet architectures...")
        
        architectures = []
        for model_name in model_names:
            try:
                arch = self.analyze_model(model_name)
                architectures.append(arch)
            except Exception as e:
                print(f"Error analyzing {model_name}: {e}")
        
        # Generate comparison analysis
        comparison = {
            'models': architectures,
            'scaling_analysis': self._analyze_scaling_patterns(architectures),
            'efficiency_ranking': self._rank_by_efficiency(architectures),
            'layer_distribution': self._analyze_layer_distribution(architectures),
            'parameter_growth': self._analyze_parameter_growth(architectures)
        }
        
        return comparison
    
    def _analyze_scaling_patterns(self, architectures: List[ModelArchitecture]) -> Dict[str, Any]:
        """Analyze compound scaling patterns"""
        scaling_data = []
        
        for arch in architectures:
            scaling_data.append({
                'model': arch.model_name,
                'depth_mult': arch.compound_scaling['depth'],
                'width_mult': arch.compound_scaling['width'],
                'resolution': arch.compound_scaling['resolution'],
                'parameters': arch.total_parameters,
                'flops': arch.total_flops
            })
        
        # Calculate scaling relationships
        base_model = next((s for s in scaling_data if 'b0' in s['model']), scaling_data[0])
        
        relationships = []
        for data in scaling_data:
            if data['model'] != base_model['model']:
                param_ratio = data['parameters'] / base_model['parameters']
                flop_ratio = (data['flops'] / base_model['flops']) if (data['flops'] and base_model['flops']) else None
                
                relationships.append({
                    'model': data['model'],
                    'parameter_ratio': param_ratio,
                    'flop_ratio': flop_ratio,
                    'theoretical_ratio': (data['depth_mult'] * data['width_mult']**2 * 
                                        (data['resolution'] / base_model['resolution'])**2)
                })
        
        return {
            'scaling_data': scaling_data,
            'scaling_relationships': relationships
        }
    
    def _rank_by_efficiency(self, architectures: List[ModelArchitecture]) -> List[Dict[str, Any]]:
        """Rank models by various efficiency metrics"""
        rankings = []
        
        for arch in architectures:
            efficiency_score = 0
            
            # Parameter efficiency (lower is better)
            param_score = 1 / (arch.total_parameters / 1e6)  # Inverse of millions of parameters
            
            # FLOP efficiency (lower is better)
            flop_score = 1 / (arch.total_flops / 1e9) if arch.total_flops else 0  # Inverse of GFLOPs
            
            # Size efficiency (lower is better)
            size_score = 1 / arch.model_size_mb
            
            # Combined efficiency score
            efficiency_score = (param_score + flop_score + size_score) / 3
            
            rankings.append({
                'model': arch.model_name,
                'efficiency_score': efficiency_score,
                'param_score': param_score,
                'flop_score': flop_score,
                'size_score': size_score,
                'parameters_m': arch.total_parameters / 1e6,
                'flops_g': arch.total_flops / 1e9 if arch.total_flops else 0,
                'size_mb': arch.model_size_mb
            })
        
        # Sort by efficiency score
        rankings.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return rankings
    
    def _analyze_layer_distribution(self, architectures: List[ModelArchitecture]) -> Dict[str, Any]:
        """Analyze layer type distribution across models"""
        layer_stats = {}
        
        for arch in architectures:
            model_stats = {}
            
            for layer in arch.layers:
                layer_type = layer.layer_type
                if layer_type not in model_stats:
                    model_stats[layer_type] = {
                        'count': 0,
                        'total_params': 0
                    }
                
                model_stats[layer_type]['count'] += 1
                model_stats[layer_type]['total_params'] += layer.parameters
            
            layer_stats[arch.model_name] = model_stats
        
        return layer_stats
    
    def _analyze_parameter_growth(self, architectures: List[ModelArchitecture]) -> Dict[str, Any]:
        """Analyze how parameters grow across variants"""
        # Sort by model variant
        sorted_archs = sorted(architectures, key=lambda x: x.model_name)
        
        growth_data = []
        base_params = None
        
        for arch in sorted_archs:
            if base_params is None:
                base_params = arch.total_parameters
                growth_factor = 1.0
            else:
                growth_factor = arch.total_parameters / base_params
            
            growth_data.append({
                'model': arch.model_name,
                'parameters': arch.total_parameters,
                'growth_factor': growth_factor,
                'depth_mult': arch.compound_scaling['depth'],
                'width_mult': arch.compound_scaling['width'],
                'resolution': arch.compound_scaling['resolution']
            })
        
        return {
            'growth_data': growth_data,
            'base_parameters': base_params
        }
    
    def create_analysis_report(self, comparison: Dict[str, Any], output_dir: str = "architecture_analysis"):
        """Create comprehensive architecture analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'comparison': comparison
        }
        
        # Convert ModelArchitecture objects to dicts for JSON serialization
        serializable_comparison = comparison.copy()
        serializable_comparison['models'] = [asdict(arch) for arch in comparison['models']]
        
        report_file = os.path.join(output_dir, 'architecture_analysis.json')
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'comparison': serializable_comparison
            }, f, indent=2, default=str)
        
        # Create visualizations
        self._create_architecture_visualizations(comparison, output_dir)
        
        # Generate summary report
        self._generate_summary_report(comparison, output_dir)
        
        print(f"Architecture analysis saved to {output_dir}")
    
    def _create_architecture_visualizations(self, comparison: Dict[str, Any], output_dir: str):
        """Create architecture comparison visualizations"""
        plt.style.use('seaborn-v0_8')
        
        architectures = comparison['models']
        
        # 1. Parameter and FLOP comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = [arch.model_name.replace('efficientnet_', 'B') for arch in architectures]
        parameters = [arch.total_parameters / 1e6 for arch in architectures]  # Millions
        flops = [arch.total_flops / 1e9 if arch.total_flops else 0 for arch in architectures]  # GFLOPs
        
        # Parameters
        bars1 = ax1.bar(model_names, parameters, alpha=0.7, color='skyblue')
        ax1.set_xlabel('EfficientNet Variant')
        ax1.set_ylabel('Parameters (Millions)')
        ax1.set_title('Model Parameters Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, param in zip(bars1, parameters):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{param:.1f}M', ha='center', va='bottom', fontsize=9)
        
        # FLOPs
        if any(flops):
            bars2 = ax2.bar(model_names, flops, alpha=0.7, color='lightcoral')
            ax2.set_xlabel('EfficientNet Variant')
            ax2.set_ylabel('FLOPs (GFLOPs)')
            ax2.set_title('Computational Complexity (FLOPs)')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, flop in zip(bars2, flops):
                if flop > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                            f'{flop:.1f}G', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'FLOP Data\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_flop_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scaling analysis
        scaling_data = comparison['scaling_analysis']['scaling_data']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models = [d['model'].replace('efficientnet_', 'B') for d in scaling_data]
        depth_mults = [d['depth_mult'] for d in scaling_data]
        width_mults = [d['width_mult'] for d in scaling_data]
        resolutions = [d['resolution'] for d in scaling_data]
        
        # Normalize resolution for bubble size
        min_res = min(resolutions)
        bubble_sizes = [(r / min_res) * 100 for r in resolutions]
        
        scatter = ax.scatter(depth_mults, width_mults, s=bubble_sizes, 
                           alpha=0.6, c=range(len(models)), cmap='viridis')
        
        # Add labels
        for i, model in enumerate(models):
            ax.annotate(f'{model}\n({resolutions[i]}px)', 
                       (depth_mults[i], width_mults[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Depth Multiplier')
        ax.set_ylabel('Width Multiplier')
        ax.set_title('EfficientNet Compound Scaling\n(Bubble size = Resolution)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scaling_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Efficiency ranking
        rankings = comparison['efficiency_ranking']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = [r['model'].replace('efficientnet_', 'B') for r in rankings]
        efficiency_scores = [r['efficiency_score'] for r in rankings]
        
        bars = ax.barh(models, efficiency_scores, alpha=0.7, color='lightgreen')
        
        ax.set_xlabel('Efficiency Score (Higher is Better)')
        ax.set_ylabel('EfficientNet Variant')
        ax.set_title('Model Efficiency Ranking')
        
        # Add value labels
        for bar, score in zip(bars, efficiency_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{score:.3f}', ha='left', va='center', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, comparison: Dict[str, Any], output_dir: str):
        """Generate text summary report"""
        report_path = os.path.join(output_dir, 'architecture_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("EfficientNet Architecture Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model overview
            f.write("Model Overview:\n")
            f.write("-" * 20 + "\n")
            for arch in comparison['models']:
                f.write(f"{arch.model_name}:\n")
                f.write(f"  Parameters: {arch.total_parameters:,} ({arch.total_parameters/1e6:.1f}M)\n")
                f.write(f"  Model Size: {arch.model_size_mb:.1f} MB\n")
                if arch.total_flops:
                    f.write(f"  FLOPs: {arch.total_flops:,} ({arch.total_flops/1e9:.1f}G)\n")
                f.write(f"  Resolution: {arch.resolution[0]}x{arch.resolution[1]}\n")
                f.write(f"  Depth Multiplier: {arch.compound_scaling['depth']}\n")
                f.write(f"  Width Multiplier: {arch.compound_scaling['width']}\n\n")
            
            # Efficiency ranking
            f.write("Efficiency Ranking:\n")
            f.write("-" * 20 + "\n")
            for i, ranking in enumerate(comparison['efficiency_ranking'], 1):
                f.write(f"{i}. {ranking['model']}: {ranking['efficiency_score']:.3f}\n")
                f.write(f"   Parameters: {ranking['parameters_m']:.1f}M\n")
                f.write(f"   Size: {ranking['size_mb']:.1f}MB\n")
                if ranking['flops_g'] > 0:
                    f.write(f"   FLOPs: {ranking['flops_g']:.1f}G\n")
                f.write("\n")
            
            # Scaling insights
            f.write("Scaling Insights:\n")
            f.write("-" * 20 + "\n")
            growth_data = comparison['parameter_growth']['growth_data']
            for data in growth_data:
                if data['growth_factor'] > 1:
                    f.write(f"{data['model']}: {data['growth_factor']:.1f}x parameter growth\n")
            
            f.write(f"\nBase model ({growth_data[0]['model']}) parameters: {comparison['parameter_growth']['base_parameters']:,}\n")

def main():
    """Main analysis function"""
    print("EfficientNet Architecture Analysis")
    print("=" * 35)
    
    analyzer = EfficientNetAnalyzer()
    
    # Analyze all variants
    variants = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                'efficientnet_b6', 'efficientnet_b7']
    
    comparison = analyzer.compare_architectures(variants)
    
    # Create analysis report
    analyzer.create_analysis_report(comparison)
    
    print("Architecture analysis completed. Check 'architecture_analysis' directory for results.")

if __name__ == "__main__":
    main()
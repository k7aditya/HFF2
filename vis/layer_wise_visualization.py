# layer_wise_visualization.py
# COMPREHENSIVE LAYER-WISE FEATURE MAP VISUALIZATION FOR HFF-NET
# ================================================================
#
# Captures intermediate activations from all major modules:
# - Frequency Domain Decomposition (FDD)
# - Frequency Domain Cross Attention (FDCA)
# - Slice Attention Layer (SAL)
# - Adaptive Layer Convolution (ALC)
# - Encoder/Decoder layers
#
# Uses PyTorch forward hooks to capture intermediate outputs without modifying model code
# Generates high-quality visualizations for interpretability and debugging

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime
import os
from tqdm import tqdm

# ============================================================================
# SECTION 1: LAYER-WISE ACTIVATION CAPTURE WITH HOOKS
# ============================================================================

class HFFNetActivationCapture:
    """
    Captures intermediate layer activations from HFF-Net using PyTorch forward hooks.
    
    Key advantages:
    - Non-invasive: No modification to model code required
    - Single forward pass: Captures all layers in one pass
    - Flexible: Can enable/disable specific layers dynamically
    - Memory efficient: Activations stored after detachment
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.activations = {}
        self.handles = []
        self.layer_names_to_capture = []
        
        # Define layers to capture (organized by module type)
        self.layers_to_capture = {
            # Input and Frequency Domain Processing
            'input_ed': 'Input Edge Detection (HighFreqConv)',
            
            # Encoder outputs (Low Freq Branch)
            'mobilenet_encoder_b1.stem': 'LF Stem (4→16 channels)',
            'mobilenet_encoder_b1.enc1': 'LF Encoder Block 1 (16 channels)',
            'mobilenet_encoder_b1.enc2': 'LF Encoder Block 2 (16→32 channels)',
            'mobilenet_encoder_b1.enc3': 'LF Encoder Block 3 (32→64 channels)',
            'mobilenet_encoder_b1.enc4': 'LF Encoder Block 4 (64→128 channels)',
            'mobilenet_encoder_b1.enc5': 'LF Encoder Block 5 (128→256 channels)',
            
            # Encoder outputs (High Freq Branch)
            'mobilenet_encoder_b2.stem': 'HF Stem (16→16 channels)',
            'mobilenet_encoder_b2.enc1': 'HF Encoder Block 1 (16 channels)',
            'mobilenet_encoder_b2.enc2': 'HF Encoder Block 2 (16→32 channels)',
            'mobilenet_encoder_b2.enc3': 'HF Encoder Block 3 (32→64 channels)',
            'mobilenet_encoder_b2.enc4': 'HF Encoder Block 4 (64→128 channels)',
            'mobilenet_encoder_b2.enc5': 'HF Encoder Block 5 (128→256 channels)',
            
            # Frequency Cross Attention (FDD + FDCA)
            'LF_l4_FDCA': 'LF Frequency Domain Cross Attention L4',
            'LF_l5_FDCA': 'LF Frequency Domain Cross Attention L5',
            'HF_l4_FDCA': 'HF Frequency Domain Cross Attention L4',
            'HF_l5_FDCA': 'HF Frequency Domain Cross Attention L5',
            
            # Fusion layers (Adaptive Layer Convolution)
            'l4_b1_t': 'LF L4 Transition Conv (Fusion)',
            'l4_b2_t': 'HF L4 Transition Conv (Fusion)',
            'l5_b1_t': 'LF L5 Transition Conv (Fusion)',
            'l5_b2_t': 'HF L5 Transition Conv (Fusion)',
            
            # Decoder outputs (LF Branch)
            'l3_b1_2': 'LF Decoder L3 Block',
            'l3_b1_u': 'LF Decoder L3 Upsample',
            'l2_b1_2': 'LF Decoder L2 Block',
            'l2_b1_u': 'LF Decoder L2 Upsample',
            'l1_b1_2': 'LF Decoder L1 Block',
            'l1_b1_f': 'LF Final Output (4 classes)',
            
            # Decoder outputs (HF Branch)
            'l3_b2_2': 'HF Decoder L3 Block',
            'l3_b2_u': 'HF Decoder L3 Upsample',
            'l2_b2_2': 'HF Decoder L2 Block',
            'l2_b2_u': 'HF Decoder L2 Upsample',
            'l1_b2_2': 'HF Decoder L1 Block',
            'l1_b2_f': 'HF Final Output (4 classes)',
        }
    
    def get_activation_hook(self, layer_name):
        """Create a forward hook to capture activation"""
        def hook(module, input, output):
            # Handle both tensor and tuple outputs
            if isinstance(output, tuple):
                output = output[0]
            
            # Detach and move to CPU to save memory
            self.activations[layer_name] = output.detach().cpu()
        
        return hook
    
    def register_hooks(self, layer_names=None):
        """Register forward hooks for specified layers"""
        if layer_names is None:
            layer_names = list(self.layers_to_capture.keys())
        
        self.layer_names_to_capture = layer_names
        
        # Register hooks for each layer
        for layer_name in layer_names:
            try:
                # Navigate to layer using dotted path
                module = self.model
                for part in layer_name.split('.'):
                    module = getattr(module, part)
                
                # Register the hook
                handle = module.register_forward_hook(self.get_activation_hook(layer_name))
                self.handles.append(handle)
                print(f"✓ Hook registered for: {layer_name}")
            
            except AttributeError as e:
                print(f"⚠ Warning: Could not register hook for {layer_name}: {e}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("✓ All hooks removed")
    
    def capture_activations(self, low_freq_input, high_freq_input):
        """
        Perform forward pass and capture all activations
        
        Args:
            low_freq_input: Low frequency input (Batch, Channels, D, H, W)
            high_freq_input: High frequency input (Batch, Channels, D, H, W)
        
        Returns:
            dict: Captured activations {layer_name: tensor}
        """
        self.activations = {}
        
        with torch.no_grad():
            self.model.eval()
            low_freq_input = low_freq_input.to(self.device)
            high_freq_input = high_freq_input.to(self.device)
            
            # Forward pass triggers all hooks
            output = self.model(low_freq_input, high_freq_input)
        
        return self.activations
    
    def get_activation_info(self):
        """Get summary information about captured activations"""
        info = {}
        for layer_name, activation in self.activations.items():
            info[layer_name] = {
                'shape': tuple(activation.shape),
                'dtype': str(activation.dtype),
                'min': float(activation.min()),
                'max': float(activation.max()),
                'mean': float(activation.mean()),
                'std': float(activation.std()),
            }
        return info

# ============================================================================
# SECTION 2: VISUALIZATION UTILITIES FOR 3D MEDICAL IMAGING
# ============================================================================

class ActivationVisualizer:
    """Visualizes 3D activations from HFF-Net in multiple ways"""
    
    def __init__(self, output_dir='./viz_output', dpi=100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    @staticmethod
    def extract_slices_3d(activation_3d, num_slices=5):
        """
        Extract representative slices from 3D activation
        
        Args:
            activation_3d: Shape (D, H, W) or (D, H, W, C) - channels averaged
            num_slices: Number of slices to extract
        
        Returns:
            dict: {slice_index: 2D slice}
        """
        if len(activation_3d.shape) == 4:
            activation_3d = activation_3d.mean(axis=-1)  # Average channels
        
        d_dim = activation_3d.shape[0]
        slice_indices = np.linspace(0, d_dim - 1, num_slices, dtype=int)
        
        slices = {}
        for idx in slice_indices:
            slices[f'slice_{idx}'] = activation_3d[idx]
        
        return slices
    
    @staticmethod
    def extract_channel_slices(activation, num_channels=8):
        """
        Extract slices from specified number of channels
        
        Args:
            activation: Shape (D, H, W, C) or (D, H, W)
        
        Returns:
            dict: {channel_id: 2D channel map}
        """
        if len(activation.shape) == 3:
            activation = activation[..., np.newaxis]  # Add channel dimension
        
        c_dim = activation.shape[-1]
        channel_indices = np.linspace(0, c_dim - 1, min(num_channels, c_dim), dtype=int)
        
        channels = {}
        middle_slice = activation.shape[0] // 2  # Middle depth slice
        
        for ch_idx in channel_indices:
            channels[f'ch_{ch_idx}'] = activation[middle_slice, :, :, ch_idx]
        
        return channels
    
    @staticmethod
    def normalize_for_visualization(array):
        """Normalize array to [0, 1] range for visualization"""
        arr_min = np.min(array)
        arr_max = np.max(array)
        
        if arr_max - arr_min < 1e-6:
            return np.ones_like(array) * 0.5  # Uniform activation
        
        return (array - arr_min) / (arr_max - arr_min)
    
    def visualize_layer_activation(self, layer_name, activation, layer_description, output_subdir='layers'):
        """
        Visualize a single layer's activation with multiple views
        
        Args:
            layer_name: Name of the layer
            activation: Activation tensor (B, C, D, H, W)
            layer_description: Human readable description
            output_subdir: Subdirectory for output
        
        Returns:
            Path to saved figure
        """
        output_dir = self.output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove batch dimension if present
        if activation.shape[0] == 1:
            activation = activation.squeeze(0)
        elif len(activation.shape) == 5:
            activation = activation[0]  # Take first in batch
        
        # Convert to numpy
        if isinstance(activation, torch.Tensor):
            activation = activation.cpu().numpy()
        
        # Permute to (D, H, W, C) if needed
        if len(activation.shape) == 4 and activation.shape[0] < min(activation.shape[1:]):
            activation = np.transpose(activation, (1, 2, 3, 0))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{layer_name}\n{layer_description}\nShape: {activation.shape}', 
                     fontsize=14, fontweight='bold')
        
        # 1. Depth slices
        slices = self.extract_slices_3d(activation, num_slices=3)
        for idx, (slice_name, slice_data) in enumerate(slices.items()):
            ax = axes[0, idx]
            norm_slice = self.normalize_for_visualization(slice_data)
            im = ax.imshow(norm_slice, cmap='turbo', interpolation='bilinear')
            ax.set_title(f'{slice_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 2. Channel slices
        channels = self.extract_channel_slices(activation, num_channels=3)
        for idx, (ch_name, ch_data) in enumerate(channels.items()):
            ax = axes[1, idx]
            norm_ch = self.normalize_for_visualization(ch_data)
            im = ax.imshow(norm_ch, cmap='hot', interpolation='bilinear')
            ax.set_title(f'{ch_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Save figure
        output_path = output_dir / f'{layer_name.replace(".", "_")}_viz.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_all_activations(self, activations_dict, layer_descriptions):
        """
        Visualize all captured activations
        
        Args:
            activations_dict: {layer_name: activation_tensor}
            layer_descriptions: {layer_name: description}
        
        Returns:
            list: Paths to all saved figures
        """
        saved_paths = []
        
        for layer_name, activation in tqdm(activations_dict.items(), desc='Visualizing layers'):
            try:
                description = layer_descriptions.get(layer_name, 'Unknown layer')
                path = self.visualize_layer_activation(layer_name, activation, description)
                saved_paths.append(path)
                print(f"✓ Visualized: {layer_name}")
            
            except Exception as e:
                print(f"✗ Error visualizing {layer_name}: {e}")
        
        return saved_paths

# ============================================================================
# SECTION 3: COMPREHENSIVE PIPELINE
# ============================================================================

class ComprehensiveLayerVisualizationPipeline:
    """End-to-end layer-wise visualization pipeline"""
    
    def __init__(self, model, device='cuda', output_base_dir='./layer_viz_output'):
        self.model = model
        self.device = device
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.output_base_dir / f'session_{timestamp}'
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.capture = HFFNetActivationCapture(model, device)
        self.visualizer = ActivationVisualizer(self.session_dir, dpi=600)
    
    def analyze_sample(self, low_freq_input, high_freq_input, sample_name='sample_0'):
        """
        Comprehensive analysis of a single sample
        
        Args:
            low_freq_input: Low frequency input tensor
            high_freq_input: High frequency input tensor
            sample_name: Name for output files
        
        Returns:
            dict: Analysis results
        """
        print(f"\n{'='*80}")
        print(f"LAYER-WISE VISUALIZATION: {sample_name}")
        print(f"{'='*80}")
        
        # Register hooks
        print(f"\n[1] Registering capture hooks...")
        self.capture.register_hooks()
        
        # Capture activations
        print(f"[2] Capturing layer activations...")
        activations = self.capture.capture_activations(low_freq_input, high_freq_input)
        
        # Get statistics
        print(f"[3] Generating activation statistics...")
        activation_info = self.capture.get_activation_info()
        
        # Print statistics
        self._print_activation_stats(activation_info)
        
        # Visualize all layers
        print(f"\n[4] Creating visualizations...")
        saved_paths = self.visualizer.visualize_all_activations(
            activations, 
            self.capture.layers_to_capture
        )
        
        # Create summary report
        print(f"\n[5] Generating summary report...")
        report_path = self._generate_report(sample_name, activation_info, saved_paths)
        
        print(f"\n✓ Visualization complete!")
        print(f"Output saved to: {self.session_dir}")
        
        return {
            'session_dir': str(self.session_dir),
            'activations': activations,
            'activation_info': activation_info,
            'saved_paths': saved_paths,
            'report_path': str(report_path),
        }
    
    def _print_activation_stats(self, activation_info):
        """Print statistics about captured activations"""
        print(f"\n{'Layer Name':<40} {'Shape':<20} {'Min':<10} {'Max':<10} {'Mean':<10}")
        print(f"{'-'*90}")
        
        for layer_name, stats in activation_info.items():
            shape_str = str(stats['shape'])
            print(f"{layer_name:<40} {shape_str:<20} {stats['min']:<10.4f} {stats['max']:<10.4f} {stats['mean']:<10.4f}")
    
    def _generate_report(self, sample_name, activation_info, saved_paths):
        """Generate a text report of visualization"""
        report_path = self.session_dir / f'{sample_name}_report.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"HFF-NET LAYER-WISE VISUALIZATION REPORT\n")
            f.write(f"{'='*80}\n")
            f.write(f"Sample: {sample_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Output Directory: {self.session_dir}\n\n")
            
            f.write(f"ACTIVATION STATISTICS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Layer':<40} {'Shape':<20} {'Min':<15} {'Max':<15} {'Mean':<15}\n")
            f.write(f"{'-'*80}\n")
            
            for layer_name, stats in activation_info.items():
                f.write(f"{layer_name:<40} {str(stats['shape']):<20} ")
                f.write(f"{stats['min']:<15.6f} {stats['max']:<15.6f} {stats['mean']:<15.6f}\n")
            
            f.write(f"\n\nVISUALIZATION FILES\n")
            f.write(f"{'-'*80}\n")
            for i, path in enumerate(saved_paths, 1):
                f.write(f"{i}. {path}\n")
        
        return report_path

# ============================================================================
# SECTION 4: MAIN EXECUTION EXAMPLE
# ============================================================================

def main():
    """Example usage"""
    from model.HFF_MobileNetV3_fixed import HFFNet
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HFFNet(in_chs1=4, in_chs2=16, num_classes=4).to(device)
    model.eval()
    
    # Load checkpoint (optional)
    # model = load_checkpoint_to_model(model, checkpoint_path, device)
    
    # Create dummy inputs (same as real usage)
    batch_size = 1
    low_freq_input = torch.randn(batch_size, 4, 128, 128, 128)
    high_freq_input = torch.randn(batch_size, 16, 128, 128, 128)
    
    # Initialize pipeline
    pipeline = ComprehensiveLayerVisualizationPipeline(model, device=device)
    
    # Run analysis
    results = pipeline.analyze_sample(low_freq_input, high_freq_input, 'sample_000')
    
    print(f"\n\nVisualization complete!")
    print(f"Results saved to: {results['session_dir']}")

if __name__ == '__main__':
    main()

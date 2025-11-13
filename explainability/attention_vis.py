# attention_vis_CORRECTED.py
# FIXES FOR 3D VOLUME HANDLING AND GRADIENT TRACKING

"""
CORRECTED ATTENTION VISUALIZATION - Handles 3D BraTS Data Properly
Key fixes:
1. Extract 2D slices from 3D volumes before visualization
2. Proper gradient tracking for Grad-CAM
3. Robust tensor to numpy conversion
4. Type checking and casting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


class EnhancedFDCAAttentionVisualizer:
    """Enhanced FDCA attention extraction with 3D volume handling"""
    
    def __init__(self, device='cuda', save_dir='results/figures/xai/attention', dpi=600):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.attention_maps = {}
        self.hooks = []
        self.activations_cache = {}
    
    def register_hooks(self, model):
        """Register hooks for FDCA layers with activation caching"""
        self.hooks = []
        self.activations_cache.clear()
        
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.attention_maps[name] = output.detach().cpu()
                    self.activations_cache[name] = output.detach().cpu().numpy()
                elif isinstance(output, tuple) and len(output) > 0:
                    self.attention_maps[name] = output[0].detach().cpu()
                    self.activations_cache[name] = output[0].detach().cpu().numpy()
            return hook
        
        for name, module in model.named_modules():
            if 'fdca' in name.lower() or 'fusion' in name.lower():
                handle = module.register_forward_hook(create_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Clean up all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_attention_maps(self, batch: torch.Tensor, model, layer_name: Optional[str] = None,
                              lf_channels: int = 4) -> Dict:
        """Extract attention maps with enhanced error handling"""
        self.attention_maps.clear()
        self.activations_cache.clear()
        
        # Split concatenated channels
        low_freq_input = batch[:, :lf_channels, ...]
        high_freq_input = batch[:, lf_channels:, ...]
        
        try:
            with torch.no_grad():
                _ = model(low_freq_input.to(self.device), high_freq_input.to(self.device))
        except RuntimeError as e:
            print(f"Warning: Model forward pass error: {e}")
            return {}
        
        attention_data = {}
        for name, attn in self.attention_maps.items():
            if layer_name is None or layer_name in name:
                attention_data[name] = attn
        
        return attention_data
    
    def aggregate_attention_maps(self, attention_maps: Dict) -> torch.Tensor:
        """Aggregate multiple attention maps with normalization"""
        aggregated = None
        target_size = None
        count = 0
        
        for _, attn in attention_maps.items():
            if attn.ndim == 5:  # (B, C, D, H, W)
                attn_spatial = attn.mean(dim=(1, 2))  # Average over C and D, get (B, H, W)
            elif attn.ndim == 4:  # (B, C, H, W)
                attn_spatial = attn.mean(dim=1)  # (B, H, W)
            else:
                continue
            
            attn_spatial = attn_spatial.float()
            
            if target_size is None:
                target_size = attn_spatial.shape[-2:]
            
            # Resize to target size
            if attn_spatial.shape[-2:] != target_size:
                attn_spatial = F.interpolate(
                    attn_spatial.unsqueeze(1),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            # Normalize per-sample to [0, 1]
            minv = attn_spatial.amin(dim=(-2, -1), keepdim=True)
            maxv = attn_spatial.amax(dim=(-2, -1), keepdim=True)
            attn_spatial = (attn_spatial - minv) / (maxv - minv + 1e-8)
            
            aggregated = attn_spatial if aggregated is None else (aggregated + attn_spatial)
            count += 1
        
        if aggregated is not None and count > 0:
            aggregated = aggregated / float(count)
        
        return aggregated
    
    def visualize_attention_enhanced(self, input_img: np.ndarray, attention_map: torch.Tensor,
                                     uncertainty_map: Optional[np.ndarray] = None,
                                     output_path: Path = None, cmap='hot', alpha=0.5, dpi=600):
        """Enhanced attention visualization with 3D volume handling"""
        
        # ===== FIX 1: Handle 3D volumes =====
        if input_img.ndim == 3:
            input_img = input_img[input_img.shape[0] // 2]  # Extract middle slice
        
        # Handle single channel
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        # Normalize
        if input_img.max() > 1:
            input_img = input_img / (input_img.max() + 1e-8)
        
        # ===== FIX 2: Convert attention to numpy properly =====
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.cpu().numpy()  # Move to CPU first
        
        # ===== FIX 3: Handle 3D attention maps =====
        if attention_map.ndim == 3:
            attention_map = attention_map[attention_map.shape[0] // 2]
        
        # ===== FIX 4: Resize with proper type casting =====
        if attention_map.shape != input_img.shape[:2]:
            attention_map = cv2.resize(attention_map.astype(np.float32), 
                                      (input_img.shape[1], input_img.shape[0]))
        
        # Create figure with high DPI
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=dpi//100)
        
        # Original image
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Input MRI', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Attention map
        im1 = axes[0, 1].imshow(attention_map, cmap=cmap)
        axes[0, 1].set_title('FDCA Attention Map', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Attention Weight')
        
        # Overlay
        heatmap = cm.get_cmap(cmap)(attention_map)[:, :, :3]
        overlay = (1 - alpha) * input_img + alpha * heatmap
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Attention Overlay', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Uncertainty overlay (if provided)
        if uncertainty_map is not None:
            # ===== FIX 5: Handle 3D uncertainty maps =====
            if uncertainty_map.ndim == 3:
                uncertainty_map = uncertainty_map[uncertainty_map.shape[0] // 2]
            
            if uncertainty_map.shape != input_img.shape[:2]:
                uncertainty_map = cv2.resize(uncertainty_map.astype(np.float32),
                                            (input_img.shape[1], input_img.shape[0]))
            
            unc_colored = cm.get_cmap('hot')(uncertainty_map)[:, :, :3]
            unc_overlay = 0.7 * input_img + 0.3 * unc_colored
            axes[1, 1].imshow(unc_overlay)
            axes[1, 1].set_title('Uncertainty Map', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', quality=95)
            print(f"✓ Saved: {output_path}")
        
        plt.close()


class EnhancedSegmentationGradCAM:
    """Enhanced Grad-CAM with 3D support and proper gradient tracking"""
    
    def __init__(self, model, target_layers: List[str], device='cuda',
                 save_dir='results/figures/xai/gradcam', dpi=600):
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks with proper gradient tracking"""
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    self.activations[name] = output[0].detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output, (tuple, list)) and len(grad_output) > 0:
                    self.gradients[name] = grad_output[0].detach()
                else:
                    self.gradients[name] = grad_output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            for target in self.target_layers:
                if target in name:
                    h1 = module.register_forward_hook(get_activation(name))
                    h2 = module.register_full_backward_hook(get_gradient(name))
                    self.hooks.append(h1)
                    self.hooks.append(h2)
                    break
    
    def remove_hooks(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_tensor: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     target_class: int = 1, eigen_smooth: bool = True,
                     lf_channels: int = 4) -> np.ndarray:
        """Generate Grad-CAM with proper gradient handling"""
        
        self.activations.clear()
        self.gradients.clear()
        
        # ===== FIX 6: Proper gradient tracking setup =====
        if isinstance(input_tensor, (tuple, list)) and len(input_tensor) == 2:
            low = input_tensor.to(self.device).float()
            high = input_tensor.to(self.device).float()
            
            # ✅ Set requires_grad=True EXPLICITLY
            low.requires_grad = True
            high.requires_grad = True
            
            # ✅ Ensure gradients are computed
            if low.grad is not None:
                low.grad.zero_()
            if high.grad is not None:
                high.grad.zero_()
            
            # ✅ NO with torch.no_grad() here!
            output = self.model(low, high)  # ← Gradients ARE computed here
        else:
            x = input_tensor.to(self.device).float()
            x.requires_grad = True
            if x.grad is not None:
                x.grad.zero_()
            
            low = x[:, :lf_channels, ...]
            high = x[:, lf_channels:, ...]
            output = self.model(low, high)   
                 
        # Handle tuple outputs
        if isinstance(output, tuple):
            output_main = output[0]
        else:
            output_main = output
        
        # Select target class
        if output_main.ndim == 5:  # 3D (B, C, D, H, W)
            target_output = output_main[:, target_class, ...]
        elif output_main.ndim == 4:  # 2D (B, C, H, W)
            target_output = output_main[:, target_class, ...]
        else:
            raise ValueError(f"Unexpected output shape: {output_main.shape}")
        
        # Backward pass
        loss = target_output.sum()
        loss.backward()
        
        # Compute Grad-CAM with improved weighting
        cam = None
        
        for layer_name in list(self.activations.keys()):
            if layer_name not in self.gradients:
                continue
            
            activations = self.activations[layer_name]
            gradients = self.gradients[layer_name]
            
            # Improved weighting: second-order gradients
            if gradients.ndim == 5:  # 3D
                weights = (gradients ** 2).mean(dim=(2, 3, 4), keepdim=True)
                weighted_act = (weights * activations).sum(dim=1, keepdim=True)
                layer_cam = weighted_act.mean(dim=2).squeeze(1)
            elif gradients.ndim == 4:  # 2D
                weights = (gradients ** 2).mean(dim=(2, 3), keepdim=True)
                layer_cam = (weights * activations).sum(dim=1)
            else:
                continue
            
            layer_cam = F.relu(layer_cam)
            cam = layer_cam if cam is None else (cam + layer_cam)
        
        if cam is None:
            return None
        
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        if eigen_smooth:
            for i in range(cam.shape[0]):
                if cam[i].ndim == 3:
                    cam[i] = gaussian_filter(cam[i][cam[i].shape[0]//2], sigma=1.5)  # Extract 2D
                else:
                    cam[i] = gaussian_filter(cam[i], sigma=1.5)
        
        return cam
    
    def generate_multi_class_cam(self, input_tensor: Union[torch.Tensor, Tuple],
                                num_classes: int = 4, lf_channels: int = 4) -> Dict[int, np.ndarray]:
        """Generate CAM for all classes"""
        cams = {}
        for class_idx in range(num_classes):
            try:
                cams[class_idx] = self.generate_cam(input_tensor, class_idx, lf_channels=lf_channels)
            except Exception as e:
                print(f"Warning: Failed to generate CAM for class {class_idx}: {e}")
        return cams
    
    def visualize_gradcam_enhanced(self, input_img: np.ndarray, cam: np.ndarray,
                                   seg_mask: np.ndarray, uncertainty_map: Optional[np.ndarray] = None,
                                   output_path: Path = None, dpi=600):
        """Enhanced Grad-CAM visualization with 3D handling"""
        
        if cam is None:
            return
        
        if cam.ndim == 3:
            cam = cam[0]
        
        # ===== FIX 8: Handle 3D volumes =====
        if input_img.ndim == 3:
            input_img = input_img[input_img.shape[0] // 2]
        if seg_mask.ndim == 3:
            seg_mask = seg_mask[seg_mask.shape[0] // 2]
        
        # Handle single channel
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        if input_img.max() > 1:
            input_img = input_img / (input_img.max() + 1e-8)
        
        # Resize CAM
        if cam.shape != input_img.shape[:2]:
            cam = cv2.resize(cam.astype(np.float32), (input_img.shape[1], input_img.shape[0]))
        
        if seg_mask.shape != input_img.shape[:2]:
            seg_mask = cv2.resize(seg_mask.astype(np.float32), 
                                 (input_img.shape[1], input_img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.7 * input_img + 0.3 * heatmap
        
        # Create figure
        if uncertainty_map is not None:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=dpi//100)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi//100)
        
        # Row 1
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Input MRI', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(cam, cmap='hot')
        axes[0, 1].set_title('Grad-CAM Attention', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Attention')
        
        axes[1, 0].imshow(seg_mask, cmap='Paired')
        axes[1, 0].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Add uncertainty if provided
        if uncertainty_map is not None:
            if uncertainty_map.ndim == 3:
                uncertainty_map = uncertainty_map[uncertainty_map.shape[0] // 2]
            
            if uncertainty_map.shape != input_img.shape[:2]:
                uncertainty_map = cv2.resize(uncertainty_map.astype(np.float32),
                                            (input_img.shape[1], input_img.shape[0]))
            
            im2 = axes[0, 2].imshow(uncertainty_map, cmap='hot')
            axes[0, 2].set_title('Uncertainty Map', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], label='Uncertainty')
            
            axes[1, 2].imshow(input_img, cmap='gray', alpha=0.6)
            axes[1, 2].imshow(uncertainty_map, cmap='hot', alpha=0.4)
            axes[1, 2].set_title('Uncertainty Overlay', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', quality=95)
            print(f"✓ Saved: {output_path}")
        
        plt.close()


class EnhancedFrequencyComponentAnalyzer:
    """Analyze LF/HF contributions with 3D volume handling"""
    
    def __init__(self, device='cuda', save_dir='results/figures/xai/freq', dpi=600):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def generate_lf_only_prediction(self, model, full_input: torch.Tensor,
                                    lf_channels: int = 4) -> torch.Tensor:
        """LF-only prediction"""
        low = full_input[:, :lf_channels, ...].to(self.device)
        high = torch.zeros_like(full_input[:, lf_channels:, ...], device=self.device)
        
        with torch.no_grad():
            out = model(low, high)
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]
        
        return torch.softmax(out, dim=1)
    
    def generate_hf_only_prediction(self, model, full_input: torch.Tensor,
                                    lf_channels: int = 4) -> torch.Tensor:
        """HF-only prediction"""
        low = torch.zeros_like(full_input[:, :lf_channels, ...], device=self.device)
        high = full_input[:, lf_channels:, ...].to(self.device)
        
        with torch.no_grad():
            out = model(low, high)
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]
        
        return torch.softmax(out, dim=1)
    
    def visualize_frequency_contributions_enhanced(self, input_img: np.ndarray,
                                                   lf_pred: np.ndarray,
                                                   hf_pred: np.ndarray,
                                                   full_pred: np.ndarray,
                                                   ground_truth: np.ndarray,
                                                   output_path: Path = None,
                                                   dpi=600):
        """Enhanced visualization with 3D volume handling"""
        
        # ===== FIX 9: Handle 3D volumes =====
        if input_img.ndim == 3:
            input_img = input_img[input_img.shape[0] // 2]
        if lf_pred.ndim == 3:
            lf_pred = lf_pred[lf_pred.shape[0] // 2]
        if hf_pred.ndim == 3:
            hf_pred = hf_pred[hf_pred.shape[0] // 2]
        if full_pred.ndim == 3:
            full_pred = full_pred[full_pred.shape[0] // 2]
        if ground_truth.ndim == 3:
            ground_truth = ground_truth[ground_truth.shape[0] // 2]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=dpi//100)
        
        # Row 1
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Input MRI', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(lf_pred, cmap='Paired')
        axes[0, 1].set_title('LF-Only Prediction\n(Global Shape)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(hf_pred, cmap='Paired')
        axes[0, 2].set_title('HF-Only Prediction\n(Sharp Boundaries)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2
        axes[1, 0].imshow(full_pred, cmap='Paired')
        axes[1, 0].set_title('Full Model Prediction\n(LF+HF Fusion)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ground_truth, cmap='Paired')
        axes[1, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Disagreement map
        diff = np.abs(lf_pred.astype(np.float32) - hf_pred.astype(np.float32))
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        im = axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('LF vs HF Disagreement\n(Uncertainty)', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], label='Disagreement')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', quality=95)
            print(f"✓ Saved: {output_path}")
        
        plt.close()


# Export classes
__all__ = [
    'EnhancedFDCAAttentionVisualizer',
    'EnhancedSegmentationGradCAM',
    'EnhancedFrequencyComponentAnalyzer'
]

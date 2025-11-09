"""
Explainability Module: FDCA Attention Visualization and Grad-CAM for HFF-Net
Handles attention map extraction, Grad-CAM generation, and frequency analysis visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from scipy.ndimage import gaussian_filter


class FDCAAttentionVisualizer:
    """Extract and visualize FDCA attention maps from HFF-Net"""
    
    def __init__(self, device='cuda', save_dir='results/figures/xai/attention'):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.attention_maps = {}
        
    def register_hooks(self, model):
        """Register forward hooks to capture FDCA attention module outputs"""
        self.hooks = []
        
        # Hook into FDCA modules in the model
        def create_hook(name):
            def hook(module, input, output):
                # Output from FDCA includes attention-weighted features
                if isinstance(output, torch.Tensor):
                    self.attention_maps[name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    self.attention_maps[name] = output[0].detach().cpu()
            return hook
        
        # Register hooks for FDCA layers (adjust names based on actual model architecture)
        for name, module in model.named_modules():
            if 'fdca' in name.lower():
                handle = module.register_forward_hook(create_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def extract_attention_maps(self, batch: torch.Tensor, model, layer_name: Optional[str] = None) -> Dict:
        self.attention_maps.clear()

        # split channels according to LF and HF
        num_low_freq_channels = 4  # replace with actual number of LF channels used
        low_freq_input = batch[:, :num_low_freq_channels, ...]
        high_freq_input = batch[:, num_low_freq_channels:, ...]

        with torch.no_grad():
            _ = model(low_freq_input.to(self.device), high_freq_input.to(self.device))

        attention_data = {}
        for name, attn in self.attention_maps.items():
            if layer_name is None or layer_name in name:
                attention_data[name] = attn
        return attention_data

    
    def aggregate_attention_maps(self, attention_maps: Dict) -> torch.Tensor:
        """
        Aggregate multiple attention maps into a single spatial attention map
        
        Args:
            attention_maps: Dictionary of attention tensors
            
        Returns:
            Aggregated attention map (spatial dimensions)
        """
        aggregated = None
        
        for attn in attention_maps.values():
            if len(attn.shape) == 5:  # (B, C, D, H, W)
                # Average across channels and depth
                attn_spatial = attn.mean(dim=(1, 2))  # (B, H, W)
            elif len(attn.shape) == 4:  # (B, C, H, W)
                # Average across channels
                attn_spatial = attn.mean(dim=1)  # (B, H, W)
            else:
                continue
            
            # Normalize to [0, 1]
            attn_spatial = attn_spatial - attn_spatial.min()
            attn_spatial = attn_spatial / (attn_spatial.max() + 1e-8)
            
            if aggregated is None:
                aggregated = attn_spatial
            else:
                aggregated = aggregated + attn_spatial
        
        if aggregated is not None:
            aggregated = aggregated / len(attention_maps)
        
        return aggregated
    
    def visualize_attention(self, input_img: np.ndarray, attention_map: torch.Tensor,
                           output_path: Path, cmap='jet', alpha=0.5):
        """
        Overlay attention map on input image
        
        Args:
            input_img: Input MRI image (H, W) or (H, W, 3)
            attention_map: Attention heatmap (H, W)
            output_path: Path to save visualization
            cmap: Colormap name
            alpha: Transparency of overlay
        """
        # Handle single channel image
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        # Normalize input image to [0, 1]
        if input_img.max() > 1:
            input_img = input_img / input_img.max()
        
        # Convert attention map to numpy
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.numpy()
        
        # Resize attention map to match input image
        if attention_map.shape != input_img.shape[:2]:
            attention_map = cv2.resize(attention_map, 
                                       (input_img.shape[1], input_img.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8),
            getattr(cv2, f'COLORMAP_{cmap.upper()}', cv2.COLORMAP_JET)
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay
        overlay = (1 - alpha) * input_img + alpha * heatmap
        
        # Save
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(input_img, cmap='gray')
        plt.title('Input MRI')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(attention_map, cmap=cmap)
        plt.colorbar(label='Attention Weight')
        plt.title('FDCA Attention Map')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class SegmentationGradCAM:
    """Grad-CAM implementation for semantic segmentation"""
    
    def __init__(self, model, target_layers: List[str], device='cuda',
                 save_dir='results/figures/xai/gradcam'):
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations and gradients"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            for target in self.target_layers:
                if target in name:
                    h1 = module.register_forward_hook(get_activation(name))
                    h2 = module.register_backward_hook(get_gradient(name))
                    self.hooks.append(h1)
                    self.hooks.append(h2)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = 1,
                    eigen_smooth: bool = False) -> np.ndarray:
        """
        Generate Grad-CAM for semantic segmentation
        
        Args:
            input_tensor: Input image tensor (B, C, H, W) or (B, C, D, H, W)
            target_class: Target segmentation class
            eigen_smooth: Apply eigen smoothing
            
        Returns:
            CAM heatmap
        """
        self.activations.clear()
        self.gradients.clear()
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle different output shapes
        if len(output.shape) == 5:  # 3D segmentation
            target_output = output[:, target_class, :, :, :]
        elif len(output.shape) == 4:  # 2D segmentation
            target_output = output[:, target_class, :, :]
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")
        
        # Backward pass
        loss = target_output.sum()
        loss.backward()
        
        # Compute Grad-CAM for each layer
        cam = None
        for layer_name in self.activations.keys():
            if layer_name in self.gradients:
                activations = self.activations[layer_name]
                gradients = self.gradients[layer_name]
                
                # Compute weights
                if len(gradients.shape) == 5:  # 3D
                    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
                    weighted_act = (weights * activations).sum(dim=1, keepdim=True)
                    layer_cam = weighted_act.mean(dim=2).squeeze()
                elif len(gradients.shape) == 4:  # 2D
                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    weighted_act = (weights * activations).sum(dim=1, keepdim=True)
                    layer_cam = layer_cam.squeeze()
                
                layer_cam = F.relu(layer_cam)
                
                if cam is None:
                    cam = layer_cam
                else:
                    cam = cam + layer_cam
        
        if cam is not None:
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            if eigen_smooth:
                cam = gaussian_filter(cam, sigma=1.0)
        
        return cam
    
    def generate_multi_class_cam(self, input_tensor: torch.Tensor,
                                num_classes: int = 4) -> Dict[int, np.ndarray]:
        """
        Generate Grad-CAM for multiple segmentation classes
        
        Returns:
            Dictionary mapping class indices to CAM heatmaps
        """
        cams = {}
        for class_idx in range(num_classes):
            cams[class_idx] = self.generate_cam(input_tensor, class_idx)
        return cams
    
    def visualize_gradcam(self, input_img: np.ndarray, cam: np.ndarray,
                         seg_mask: np.ndarray, output_path: Path):
        """
        Visualize Grad-CAM with input image and segmentation mask
        
        Args:
            input_img: Input image (H, W) or (H, W, 3)
            cam: Grad-CAM heatmap (H, W)
            seg_mask: Segmentation mask (H, W)
            output_path: Path to save figure
        """
        # Handle single channel image
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        if input_img.max() > 1:
            input_img = input_img / input_img.max()
        
        # Resize CAM to match image
        if cam.shape != input_img.shape[:2]:
            cam = cv2.resize(cam, (input_img.shape[1], input_img.shape[0]))
        
        if seg_mask.shape != input_img.shape[:2]:
            seg_mask = cv2.resize(seg_mask, (input_img.shape[1], input_img.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        overlay = 0.7 * input_img + 0.3 * heatmap
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title('Input MRI')
        axes[0].axis('off')
        
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        axes[2].imshow(seg_mask, cmap='Paired')
        axes[2].set_title('Segmentation Mask')
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class FrequencyComponentAnalyzer:
    """Analyze contributions of LF and HF components"""
    
    def __init__(self, device='cuda', save_dir='results/figures/xai/freq'):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_lf_only_prediction(self, model, input_tensor: torch.Tensor,
                                    lf_indices: List[int] = None) -> torch.Tensor:
        """
        Generate prediction using only low-frequency components
        
        Args:
            model: HFF-Net model
            input_tensor: Input with both LF and HF components
            lf_indices: Indices of LF modalities (e.g., [0, 1, 2, 3] for 4 MRI modalities)
            
        Returns:
            Segmentation output using only LF
        """
        if lf_indices is None:
            lf_indices = list(range(4))  # Default: first 4 channels (LF components)
        
        # Create masked input with only LF
        masked_input = torch.zeros_like(input_tensor)
        masked_input[:, lf_indices, ...] = input_tensor[:, lf_indices, ...]
        
        with torch.no_grad():
            output = model(masked_input.to(self.device))
        
        return output
    
    def generate_hf_only_prediction(self, model, input_tensor: torch.Tensor,
                                   hf_indices: List[int] = None) -> torch.Tensor:
        """
        Generate prediction using only high-frequency components
        
        Args:
            model: HFF-Net model
            input_tensor: Input tensor
            hf_indices: Indices of HF modalities (e.g., [4, 5, ..., 19])
            
        Returns:
            Segmentation output using only HF
        """
        if hf_indices is None:
            hf_indices = list(range(4, 20))  # Default: channels 4-19 (HF components)
        
        masked_input = torch.zeros_like(input_tensor)
        for idx in hf_indices:
            if idx < input_tensor.shape[1]:
                masked_input[:, idx, ...] = input_tensor[:, idx, ...]
        
        with torch.no_grad():
            output = model(masked_input.to(self.device))
        
        return output
    
    def visualize_frequency_contributions(self, input_img: np.ndarray,
                                         lf_pred: np.ndarray,
                                         hf_pred: np.ndarray,
                                         full_pred: np.ndarray,
                                         ground_truth: np.ndarray,
                                         output_path: Path):
        """
        Visualize contributions of LF and HF components
        
        Args:
            input_img: Input MRI image
            lf_pred: Prediction from LF only
            hf_pred: Prediction from HF only
            full_pred: Prediction from full model
            ground_truth: Ground truth mask
            output_path: Save path
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Input and predictions
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Input MRI')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(lf_pred, cmap='Paired')
        axes[0, 1].set_title('LF-Only Prediction\n(Global Shape)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(hf_pred, cmap='Paired')
        axes[0, 2].set_title('HF-Only Prediction\n(Sharp Boundaries)')
        axes[0, 2].axis('off')
        
        # Row 2: Full prediction, GT, and difference
        axes[1, 0].imshow(full_pred, cmap='Paired')
        axes[1, 0].set_title('Full Model Prediction\n(LF + HF Fusion)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(ground_truth, cmap='Paired')
        axes[1, 1].set_title('Ground Truth')
        axes[1, 1].axis('off')
        
        # Show difference between LF and HF
        diff = np.abs(lf_pred - hf_pred).astype(np.float32)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        im = axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('LF vs HF Disagreement')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_frequency_stats(self, lf_pred: np.ndarray,
                               hf_pred: np.ndarray,
                               full_pred: np.ndarray,
                               ground_truth: np.ndarray) -> Dict:
        """
        Compute statistics on frequency component contributions
        
        Returns:
            Dictionary with statistics
        """
        from sklearn.metrics import dice_score, jaccard_score
        
        # Convert to binary
        lf_binary = (lf_pred > 0.5).astype(np.float32)
        hf_binary = (hf_pred > 0.5).astype(np.float32)
        full_binary = (full_pred > 0.5).astype(np.float32)
        gt_binary = (ground_truth > 0.5).astype(np.float32)
        
        stats = {
            'lf_dice': dice_score(gt_binary, lf_binary, average='weighted'),
            'hf_dice': dice_score(gt_binary, hf_binary, average='weighted'),
            'full_dice': dice_score(gt_binary, full_binary, average='weighted'),
            'lf_iou': jaccard_score(gt_binary, lf_binary, average='weighted'),
            'hf_iou': jaccard_score(gt_binary, hf_binary, average='weighted'),
            'full_iou': jaccard_score(gt_binary, full_binary, average='weighted'),
        }
        
        return stats


# Export classes
__all__ = [
    'FDCAAttentionVisualizer',
    'SegmentationGradCAM',
    'FrequencyComponentAnalyzer'
]

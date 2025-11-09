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
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from scipy.ndimage import gaussian_filter


class FDCAAttentionVisualizer:
    """Extract and visualize FDCA attention maps from HFF-Net"""
    def __init__(self, device='cuda', save_dir='results/figures/xai/attention'):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.attention_maps = {}
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks to capture FDCA attention module outputs"""
        self.hooks = []

        def create_hook(name):
            def hook(module, input, output):
                # Output from FDCA includes attention-weighted features
                if isinstance(output, torch.Tensor):
                    self.attention_maps[name] = output.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 0:
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
        self.hooks = []

    def extract_attention_maps(self, batch: torch.Tensor, model, layer_name: Optional[str] = None, lf_channels: int = 4) -> Dict:
        """
        Extract attention maps from FDCA layers

        Args:
            batch: Concatenated input tensor (B, C, D, H, W) or (B, C, H, W) with LF first, then HF
            model: HFF-Net model (expects two inputs: (LF, HF))
            layer_name: Optional filter for specific FDCA layer substring
            lf_channels: Number of LF channels at the front of 'batch'
        """
        self.attention_maps.clear()

        # Split concatenated channels into LF/HF branches
        low_freq_input = batch[:, :lf_channels, ...]
        high_freq_input = batch[:, lf_channels:, ...]

        with torch.no_grad():
            _ = model(low_freq_input.to(self.device), high_freq_input.to(self.device))

        attention_data = {}
        for name, attn in self.attention_maps.items():
            if layer_name is None or layer_name in name:
                attention_data[name] = attn
        return attention_data

    def aggregate_attention_maps(self, attention_maps: Dict) -> torch.Tensor:
        """
        Aggregate multiple attention maps into a single spatial attention map.

        - Reduces (B, C, D, H, W) to (B, H, W) by mean over C and D
        - Reduces (B, C, H, W) to (B, H, W) by mean over C
        - Resizes all maps to a common (H, W) via bilinear interpolation
        """
        aggregated = None
        target_size = None
        count = 0

        for _, attn in attention_maps.items():
            if attn.ndim == 5:  # (B, C, D, H, W)
                attn_spatial = attn.mean(dim=(1, 2))  # (B, H, W)
            elif attn.ndim == 4:  # (B, C, H, W)
                attn_spatial = attn.mean(dim=1)  # (B, H, W)
            else:
                continue

            attn_spatial = attn_spatial.float()

            # Establish target size using the first valid map
            if target_size is None:
                target_size = attn_spatial.shape[-2:]

            # Resize to match target size if needed (bilinear for 2D spatial)
            if attn_spatial.shape[-2:] != target_size:
                attn_spatial = torch.nn.functional.interpolate(
                    attn_spatial.unsqueeze(1),  # (B,1,H,W)
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # (B,H,W)

            # Normalize per-sample to [0,1]
            minv = attn_spatial.amin(dim=(-2, -1), keepdim=True)
            maxv = attn_spatial.amax(dim=(-2, -1), keepdim=True)
            attn_spatial = (attn_spatial - minv) / (maxv - minv + 1e-8)

            aggregated = attn_spatial if aggregated is None else (aggregated + attn_spatial)
            count += 1

        if aggregated is not None and count > 0:
            aggregated = aggregated / float(count)

        return aggregated

    def visualize_attention(self, input_img: np.ndarray, attention_map: torch.Tensor,
                            output_path: Path, cmap='jet', alpha=0.5):
        """
        Overlay attention map on input image

        Args:
            input_img: Input MRI image (H, W) or (H, W, 3)
            attention_map: Attention heatmap (H, W) or tensor
            output_path: Path to save visualization
            cmap: Colormap name
            alpha: Transparency of overlay
        """
        # Handle single channel image
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)

        # Normalize input image to [0, 1]
        if input_img.max() > 1:
            input_img = input_img / (input_img.max() + 1e-8)

        # Convert attention map to numpy
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.numpy()

        # Resize attention map to match input image
        if attention_map.shape != input_img.shape[:2]:
            attention_map = cv2.resize(attention_map, (input_img.shape[1], input_img.shape[0]))

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
    """Grad-CAM implementation for semantic segmentation (supports HFF-Net two-input forward)"""
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

        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            for target in self.target_layers:
                if target in name:
                    h1 = module.register_forward_hook(get_activation(name))
                    # backward_hook is deprecated; use register_full_backward_hook if needed
                    h2 = module.register_backward_hook(get_gradient(name))
                    self.hooks.append(h1)
                    self.hooks.append(h2)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _forward_two_inputs(self, input_data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], lf_channels: int = 4):
        """
        Forward wrapper to support either:
          - concatenated tensor (B, C, ...) with LF first (split inside), or
          - tuple/list of (low, high)
        """
        if isinstance(input_data, (tuple, list)) and len(input_data) == 2:
            low, high = input_data
        else:
            x: torch.Tensor = input_data
            low = x[:, :lf_channels, ...]
            high = x[:, lf_channels:, ...]
        return self.model(low.to(self.device), high.to(self.device))

    def generate_cam(self, input_tensor: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     target_class: int = 1, eigen_smooth: bool = False, lf_channels: int = 4) -> np.ndarray:
        """
        Generate Grad-CAM for semantic segmentation

        Args:
            input_tensor: Either concatenated input (B, C, ...) or a tuple (low, high)
            target_class: Target segmentation class index
            eigen_smooth: Apply gaussian smoothing to CAM
            lf_channels: Number of LF channels if input is concatenated
        """
        self.activations.clear()
        self.gradients.clear()

        # Forward pass with two-input support
        if isinstance(input_tensor, (tuple, list)) and len(input_tensor) == 2:
            low = input_tensor[0].to(self.device)
            high = input_tensor[1].to(self.device)
            low.requires_grad_(True)
            high.requires_grad_(True)
            output = self.model(low, high)
        else:
            x = input_tensor.to(self.device)
            x.requires_grad_(True)
            low = x[:, :lf_channels, ...]
            high = x[:, lf_channels:, ...]
            output = self.model(low, high)

        # Select target logits channel
        if isinstance(output, tuple):
            output_main = output[0]
        else:
            output_main = output

        if output_main.ndim == 5:  # 3D segmentation (B, C, D, H, W)
            target_output = output_main[:, target_class, ...]
        elif output_main.ndim == 4:  # 2D segmentation (B, C, H, W)
            target_output = output_main[:, target_class, ...]
        else:
            raise ValueError(f"Unexpected output shape: {output_main.shape}")

        # Backward pass to get gradients
        loss = target_output.sum()
        loss.backward()

        # Compute Grad-CAM
        cam = None
        for layer_name in list(self.activations.keys()):
            if layer_name not in self.gradients:
                continue

            activations = self.activations[layer_name]  # (B, C, D, H, W) or (B, C, H, W)
            gradients = self.gradients[layer_name]      # same ndim as activations

            if gradients.ndim == 5:  # 3D
                weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
                weighted_act = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, D, H, W)
                layer_cam = weighted_act.mean(dim=2).squeeze(1)  # average over D -> (B, H, W)
            elif gradients.ndim == 4:  # 2D
                weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
                layer_cam = (weights * activations).sum(dim=1)  # (B, H, W)
            else:
                continue

            layer_cam = F.relu(layer_cam)

            cam = layer_cam if cam is None else (cam + layer_cam)

        if cam is None:
            return None

        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        if eigen_smooth:
            cam = gaussian_filter(cam, sigma=1.0)

        # Return per-batch CAM (B, H, W)
        return cam

    def generate_multi_class_cam(self, input_tensor: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                                 num_classes: int = 4, lf_channels: int = 4) -> Dict[int, np.ndarray]:
        """
        Generate Grad-CAM for multiple segmentation classes
        Returns a dict: class_idx -> CAM (B, H, W)
        """
        cams = {}
        for class_idx in range(num_classes):
            cams[class_idx] = self.generate_cam(input_tensor, class_idx, lf_channels=lf_channels)
        return cams

    def visualize_gradcam(self, input_img: np.ndarray, cam: np.ndarray,
                          seg_mask: np.ndarray, output_path: Path):
        """
        Visualize Grad-CAM with input image and segmentation mask

        Args:
            input_img: Input image (H, W) or (H, W, 3)
            cam: Grad-CAM heatmap (H, W) or (B, H, W) -> use cam[0] if batched
            seg_mask: Segmentation mask (H, W)
            output_path: Path to save figure
        """
        if cam is None:
            return

        if cam.ndim == 3:
            cam = cam[0]

        # Handle single channel image
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)

        if input_img.max() > 1:
            input_img = input_img / (input_img.max() + 1e-8)

        # Resize CAM to match image
        if cam.shape != input_img.shape[:2]:
            cam = cv2.resize(cam, (input_img.shape[1], input_img.shape[0]))

        if seg_mask.shape != input_img.shape[:2]:
            seg_mask = cv2.resize(seg_mask, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_NEAREST)

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
    """Analyze contributions of LF and HF components for HFF-Net (two-input forward)"""
    def __init__(self, device='cuda', save_dir='results/figures/xai/freq'):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_lf_only_prediction(self, model, full_input: torch.Tensor, lf_channels: int = 4) -> torch.Tensor:
        """
        Generate prediction using only low-frequency components.
        Sends LF to low branch and zeros to high branch.
        """
        low = full_input[:, :lf_channels, ...].to(self.device)
        high = torch.zeros_like(full_input[:, lf_channels:, ...], device=self.device)

        with torch.no_grad():
            out = model(low, high)
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]
        return out

    def generate_hf_only_prediction(self, model, full_input: torch.Tensor, lf_channels: int = 4) -> torch.Tensor:
        """
        Generate prediction using only high-frequency components.
        Sends zeros to low branch and HF to high branch.
        """
        low = torch.zeros_like(full_input[:, :lf_channels, ...], device=self.device)
        high = full_input[:, lf_channels:, ...].to(self.device)

        with torch.no_grad():
            out = model(low, high)
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]
        return out

    def visualize_frequency_contributions(self, input_img: np.ndarray,
                                          lf_pred: np.ndarray,
                                          hf_pred: np.ndarray,
                                          full_pred: np.ndarray,
                                          ground_truth: np.ndarray,
                                          output_path: Path):
        """
        Visualize contributions of LF and HF components
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
        """
        # Note: sklearn.metrics.dice_score does not exist in standard sklearn.
        # Keep as placeholder or replace with a proper Dice implementation if needed.
        from sklearn.metrics import jaccard_score

        # Convert to binary
        lf_binary = (lf_pred > 0.5).astype(np.int32).ravel()
        hf_binary = (hf_pred > 0.5).astype(np.int32).ravel()
        full_binary = (full_pred > 0.5).astype(np.int32).ravel()
        gt_binary = (ground_truth > 0.5).astype(np.int32).ravel()

        def dice_coef(a, b):
            inter = np.logical_and(a == 1, b == 1).sum()
            s = (a == 1).sum() + (b == 1).sum()
            return 1.0 if s == 0 else (2.0 * inter) / (s + 1e-8)

        stats = {
            'lf_dice': float(dice_coef(gt_binary, lf_binary)),
            'hf_dice': float(dice_coef(gt_binary, hf_binary)),
            'full_dice': float(dice_coef(gt_binary, full_binary)),
            'lf_iou': float(jaccard_score(gt_binary, lf_binary, zero_division=1)),
            'hf_iou': float(jaccard_score(gt_binary, hf_binary, zero_division=1)),
            'full_iou': float(jaccard_score(gt_binary, full_binary, zero_division=1)),
        }

        return stats


# Export classes
__all__ = [
    'FDCAAttentionVisualizer',
    'SegmentationGradCAM',
    'FrequencyComponentAnalyzer'
]

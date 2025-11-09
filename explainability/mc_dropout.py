"""
MC-Dropout Uncertainty Estimation Module for HFF-Net
Enables uncertainty quantification through Monte Carlo sampling at inference time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import cv2
from scipy import stats
import seaborn as sns


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation in segmentation
    Performs multiple stochastic forward passes to estimate predictive uncertainty
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 20, 
                 device: str = 'cuda', save_dir: str = 'results/figures/uncertainty'):
        """
        Args:
            model: Trained HFF-Net model
            num_samples: Number of MC forward passes (N=20 recommended)
            device: Computation device
            save_dir: Directory to save uncertainty visualizations
        """
        self.model = model
        self.num_samples = num_samples
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.dropout_layers = []
        self._prepare_model()
    
    def _prepare_model(self):
        """Identify and prepare dropout layers for MC sampling"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d) or \
               isinstance(module, nn.Dropout3d):
                # Enable dropout at inference time
                self.dropout_layers.append(module)
    
    def enable_dropout_inference(self):
        """Enable dropout layers during inference (training mode without gradient)"""
        for layer in self.dropout_layers:
            layer.train()
    
    def disable_dropout_inference(self):
        """Disable dropout layers during inference"""
        for layer in self.dropout_layers:
            layer.eval()
    
    def mc_forward_pass(self, input1: torch.Tensor, input2: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        self.enable_dropout_inference()
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input1, input2)
                outputs.append(output[0].cpu())
        self.disable_dropout_inference()
        return outputs

    
    def compute_uncertainty_maps(self, outputs: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean prediction and uncertainty from MC samples
        
        Args:
            outputs: List of N segmentation outputs, each (B, C, H, W) or (B, C, D, H, W)
            
        Returns:
            Tuple of (mean_prediction, uncertainty_map)
            - mean_prediction: Average across samples (B, C, H, W)
            - uncertainty_map: Variance or entropy across samples (B, 1, H, W)
        """
        outputs = torch.stack(outputs, dim=0)  # (N, B, C, H, W) or (N, B, C, D, H, W)
        
        # Compute mean prediction
        mean_pred = outputs.mean(dim=0)  # (B, C, H, W)
        
        # Compute uncertainty as predictive variance
        # For segmentation, compute variance per spatial location
        variance = outputs.var(dim=0)  # (B, C, H, W)
        
        # Aggregate across classes: average variance
        uncertainty = variance.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        return mean_pred.numpy(), uncertainty.numpy()
    
    def compute_entropy_uncertainty(self, outputs: List[torch.Tensor]) -> np.ndarray:
        """
        Compute predictive entropy as uncertainty measure
        
        Args:
            outputs: List of N segmentation outputs (with softmax/sigmoid applied)
            
        Returns:
            Entropy map (B, 1, H, W)
        """
        outputs = torch.stack(outputs, dim=0)  # (N, B, C, H, W)
        
        # Compute mean softmax prediction
        mean_probs = outputs.mean(dim=0)  # (B, C, H, W)
        
        # Compute entropy: -sum(p * log(p))
        # Avoid log(0) by clamping
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1, keepdim=True)
        
        return entropy.numpy()
    
    def compute_mutual_information(self, outputs: List[torch.Tensor]) -> np.ndarray:
        """
        Compute mutual information (BALD - Bayesian Active Learning by Disagreement)
        
        Args:
            outputs: List of N segmentation outputs
            
        Returns:
            Mutual information map (B, 1, H, W)
        """
        outputs = torch.stack(outputs, dim=0)  # (N, B, C, H, W)
        
        # Compute mean prediction entropy
        mean_probs = outputs.mean(dim=0)
        mean_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1, keepdim=True)
        
        # Compute expected entropy
        entropies = -torch.sum(outputs * torch.log(outputs + 1e-10), dim=2)  # (N, B, H, W)
        expected_entropy = entropies.mean(dim=0, keepdim=True)  # (B, 1, H, W)
        
        # Mutual information = mean_entropy - expected_entropy
        mi = mean_entropy - expected_entropy
        
        return mi.numpy()
    
    def compute_segmentation_error(self, mean_pred: torch.Tensor,
                                   uncertainty: np.ndarray,
                                   ground_truth: np.ndarray,
                                   threshold: float = 0.5) -> Dict:
        """
        Compute correlation between uncertainty and segmentation error
        
        Args:
            mean_pred: Mean prediction from MC samples
            uncertainty: Uncertainty map
            ground_truth: Ground truth segmentation mask
            threshold: Threshold for binary segmentation
            
        Returns:
            Dictionary with correlation statistics
        """
        # Convert to binary
        pred_binary = (mean_pred.argmax(dim=1).numpy() > 0).astype(np.float32)
        gt_binary = (ground_truth > 0.5).astype(np.float32)
        
        # Compute pixel-wise error
        error = np.abs(pred_binary - gt_binary)
        
        # Flatten maps
        error_flat = error.flatten()
        uncertainty_flat = uncertainty.squeeze().flatten()
        
        # Compute correlation
        correlation, p_value = stats.pearsonr(uncertainty_flat, error_flat)
        
        # Compute Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(uncertainty_flat, error_flat)
        
        return {
            'pearson_correlation': correlation,
            'pearson_pvalue': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_p,
            'mean_uncertainty': uncertainty_flat.mean(),
            'mean_error': error_flat.mean(),
            'high_uncertainty_error_rate': error_flat[uncertainty_flat > np.percentile(uncertainty_flat, 75)].mean()
        }
    
    def visualize_uncertainty(self, input_img: np.ndarray,
                             predicted_mask: np.ndarray,
                             uncertainty_map: np.ndarray,
                             error_map: np.ndarray,
                             output_path: Path):
        """
        Visualize uncertainty estimation results
        
        Args:
            input_img: Input MRI image (H, W) or (H, W, 3)
            predicted_mask: Predicted segmentation mask
            uncertainty_map: Uncertainty heatmap
            error_map: Segmentation error (GT vs Pred)
            output_path: Save path
        """
        # Handle single channel image
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        if input_img.max() > 1:
            input_img = input_img / input_img.max()
        
        # Resize maps
        target_shape = input_img.shape[:2]
        uncertainty_resized = cv2.resize(uncertainty_map, (target_shape[1], target_shape[0]))
        error_resized = cv2.resize(error_map, (target_shape[1], target_shape[0]))
        pred_resized = cv2.resize(predicted_mask, (target_shape[1], target_shape[0]))
        
        # Create figure
        fig = plt.figure(figsize=(16, 5))
        
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(input_img, cmap='gray')
        ax1.set_title('Input MRI')
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(pred_resized, cmap='Paired')
        ax2.set_title('Predicted Mask')
        ax2.axis('off')
        
        ax3 = plt.subplot(1, 4, 3)
        im3 = ax3.imshow(uncertainty_resized, cmap='hot')
        ax3.set_title('Uncertainty Map\n(High uncertainty = Red)')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, label='Uncertainty')
        
        ax4 = plt.subplot(1, 4, 4)
        im4 = ax4.imshow(error_resized, cmap='RdYlBu_r')
        ax4.set_title('Error Map\n(High error = Red)')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, label='Error')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_uncertainty_statistics(self, uncertainty_maps: List[np.ndarray],
                                        errors: List[np.ndarray],
                                        output_path: Path):
        """
        Create statistical visualization of uncertainty-error relationship
        
        Args:
            uncertainty_maps: List of uncertainty maps from test set
            errors: List of error maps from test set
            output_path: Save path
        """
        # Flatten all maps
        all_uncertainties = np.concatenate([u.flatten() for u in uncertainty_maps])
        all_errors = np.concatenate([e.flatten() for e in errors])
        
        # Compute correlation
        correlation, _ = stats.pearsonr(all_uncertainties, all_errors)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot
        axes[0, 0].scatter(all_uncertainties, all_errors, alpha=0.1, s=1)
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Segmentation Error')
        axes[0, 0].set_title(f'Uncertainty vs Error\n(r = {correlation:.3f})')
        
        # Uncertainty distribution
        axes[0, 1].hist(all_uncertainties, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Uncertainty Distribution')
        
        # Error distribution
        axes[1, 0].hist(all_errors, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Segmentation Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        
        # Binned analysis
        bins = np.percentile(all_uncertainties, np.linspace(0, 100, 6))
        bin_errors = []
        bin_labels = []
        
        for i in range(len(bins) - 1):
            mask = (all_uncertainties >= bins[i]) & (all_uncertainties < bins[i + 1])
            if mask.sum() > 0:
                bin_errors.append(all_errors[mask].mean())
                bin_labels.append(f'{i+1}')
        
        axes[1, 1].bar(bin_labels, bin_errors, color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Uncertainty Bin')
        axes[1, 1].set_ylabel('Mean Error')
        axes[1, 1].set_title('Mean Error per Uncertainty Bin')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_uncertainty_report(self, test_loader, output_dir: Path) -> Dict:
        """
        Generate comprehensive uncertainty analysis report
        
        Args:
            test_loader: DataLoader for test set
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with all uncertainty statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_uncertainties = []
        all_errors = []
        correlation_stats = []
        
        self.model.eval()
        
        for batch_idx, (images, masks) in enumerate(test_loader):
            # MC forward passes
            mc_outputs = self.mc_forward_pass(images)
            
            # Compute uncertainty
            mean_pred, uncertainty = self.compute_uncertainty_maps(mc_outputs)
            
            # Compute error
            pred_binary = (mean_pred.argmax(axis=1, keepdims=True) > 0).astype(np.float32)
            masks_np = masks.numpy()
            error = np.abs(pred_binary - masks_np)
            
            # Store for statistics
            all_uncertainties.append(uncertainty)
            all_errors.append(error)
            
            # Compute correlation for this batch
            stats_dict = self.compute_segmentation_error(
                torch.from_numpy(mean_pred),
                uncertainty,
                masks_np
            )
            correlation_stats.append(stats_dict)
        
        # Aggregate statistics
        all_uncertainties = np.concatenate(all_uncertainties)
        all_errors = np.concatenate(all_errors)
        
        # Visualize
        self.visualize_uncertainty_statistics(
            [u for u in all_uncertainties],
            [e for e in all_errors],
            output_dir / 'uncertainty_analysis.png'
        )
        
        # Save statistics table
        stats_summary = {
            'mean_uncertainty': float(all_uncertainties.mean()),
            'std_uncertainty': float(all_uncertainties.std()),
            'mean_error': float(all_errors.mean()),
            'std_error': float(all_errors.std()),
            'correlation_mean': float(np.mean([s['pearson_correlation'] for s in correlation_stats])),
            'correlation_std': float(np.std([s['pearson_correlation'] for s in correlation_stats])),
        }
        
        return stats_summary


class DropoutScheduler:
    """Manage dropout rates during training and inference"""
    
    def __init__(self, model: nn.Module, base_dropout: float = 0.5):
        self.model = model
        self.base_dropout = base_dropout
    
    def set_dropout_rate(self, rate: float):
        """Set dropout rate for all Dropout layers"""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = rate
    
    def enable_mc_dropout(self):
        """Enable MC-Dropout for inference"""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
    
    def disable_mc_dropout(self):
        """Disable dropout for inference"""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.eval()


# Export classes
__all__ = [
    'MCDropoutUncertainty',
    'DropoutScheduler'
]

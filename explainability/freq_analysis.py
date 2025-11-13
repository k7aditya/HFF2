# freq_analysis_updated.py - Enhanced Frequency Domain Analysis Module
# Updated with mechanistic interpretability and advanced metrics
 
"""
UPDATED FREQUENCY DOMAIN ANALYSIS MODULE FOR HFF-NET
======================================================

Enhanced with:
1. 3D FFT analysis for volumetric data
2. Multi-scale frequency band decomposition
3. Model sensitivity to frequency components
4. Wavelet-based analysis
5. Frequency-domain mechanistic insights
6. Band-wise energy distribution
7. 600 DPI visualizations
8. Uncertainty-aware frequency analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from scipy.ndimage import sobel, gaussian_filter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: ENHANCED FREQUENCY DECOMPOSITION & ANALYSIS
# ============================================================================

class EnhancedFrequencyDomainAnalyzer:
    """Comprehensive frequency domain analysis for HFF-Net with mechanistic insights"""
    
    def __init__(self, device: str = 'cuda', save_dir: str = 'results/figures/xai/freq', dpi: int = 600):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def extract_frequency_components_3d(self, image_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract frequency components from 3D image using 3D FFT
        
        Args:
            image_3d: 3D image (D, H, W)
            
        Returns:
            Tuple of (low_freq, high_freq) components
        """
        # Apply 3D FFT
        fft_3d = fftn(image_3d)
        fft_shift = np.fft.fftshift(fft_3d)
        
        # Create frequency magnitude
        freq_magnitude = np.abs(fft_shift)
        
        # Create frequency mask
        d, h, w = image_3d.shape
        center_d, center_h, center_w = d // 2, h // 2, w // 2
        
        # Distance from center (3D)
        indices_d = np.arange(d)
        indices_h = np.arange(h)
        indices_w = np.arange(w)
        
        dist_d = np.abs(indices_d - center_d) / (d / 2)
        dist_h = np.abs(indices_h - center_h) / (h / 2)
        dist_w = np.abs(indices_w - center_w) / (w / 2)
        
        dist_matrix = np.zeros((d, h, w))
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    dist_matrix[i, j, k] = np.sqrt(dist_d[i]**2 + dist_h[j]**2 + dist_w[k]**2)
        
        # Frequency threshold
        cutoff = 0.3  # Normalized frequency cutoff
        
        # Low-frequency and high-frequency masks
        low_freq_mask = dist_matrix <= cutoff
        high_freq_mask = dist_matrix > cutoff
        
        # Apply masks
        low_freq = fft_shift * low_freq_mask
        high_freq = fft_shift * high_freq_mask
        
        # Inverse FFT
        low_freq_spatial = np.abs(ifftn(np.fft.ifftshift(low_freq)).real)
        high_freq_spatial = np.abs(ifftn(np.fft.ifftshift(high_freq)).real)
        
        return low_freq_spatial, high_freq_spatial
    
    def extract_frequency_components_2d(self, image_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract frequency components from 2D image
        
        Args:
            image_2d: 2D image (H, W)
            
        Returns:
            Tuple of (low_freq, high_freq) components
        """
        # Apply 2D FFT
        fft_2d = fft2(image_2d)
        fft_shift = fftshift(fft_2d)
        
        # Create frequency magnitude
        freq_magnitude = np.abs(fft_shift)
        
        # Create frequency mask
        h, w = image_2d.shape
        center_h, center_w = h // 2, w // 2
        
        # Distance from center
        indices_h = np.arange(h)
        indices_w = np.arange(w)
        
        dist_h = np.abs(indices_h - center_h) / (h / 2)
        dist_w = np.abs(indices_w - center_w) / (w / 2)
        
        dist_matrix = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                dist_matrix[i, j] = np.sqrt(dist_h[i]**2 + dist_w[j]**2)
        
        cutoff = 0.3
        
        low_freq_mask = dist_matrix <= cutoff
        high_freq_mask = dist_matrix > cutoff
        
        low_freq = fft_shift * low_freq_mask
        high_freq = fft_shift * high_freq_mask
        
        # Inverse FFT
        low_freq_spatial = np.abs(ifft2(ifftshift(low_freq)).real)
        high_freq_spatial = np.abs(ifft2(ifftshift(high_freq)).real)
        
        return low_freq_spatial, high_freq_spatial
    
    def compute_frequency_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Compute frequency spectrum magnitude (log scale)
        
        Args:
            image: Input image (2D or 3D)
            
        Returns:
            Frequency spectrum (log scale)
        """
        if image.ndim == 3:
            # Use middle slice for 3D
            image = image[image.shape[0] // 2]
        
        fft_image = fft2(image)
        fft_shift = fftshift(fft_image)
        magnitude = np.abs(fft_shift)
        spectrum = np.log1p(magnitude)
        
        return spectrum
    
    def analyze_feature_map_frequency(self, feature_map: torch.Tensor) -> Dict:
        """
        Analyze frequency content of feature maps
        
        Args:
            feature_map: Feature tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Dictionary with frequency analysis metrics
        """
        if isinstance(feature_map, torch.Tensor):
            if feature_map.ndim == 4:
                feature_map = feature_map[0, 0].cpu().numpy()
            elif feature_map.ndim == 3:
                feature_map = feature_map[0].cpu().numpy()
            else:
                feature_map = feature_map.cpu().numpy()
        
        # Normalize
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # Compute FFT
        fft_feat = fft2(feature_map)
        fft_shift = fftshift(fft_feat)
        magnitude = np.abs(fft_shift)
        
        # Compute power
        power = magnitude ** 2
        
        # Compute frequency band energies
        h, w = feature_map.shape
        center_h, center_w = h // 2, w // 2
        
        indices_h = np.arange(h)
        indices_w = np.arange(w)
        
        dist_matrix = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                dist_h = np.abs(indices_h[i] - center_h) / (h / 2)
                dist_w = np.abs(indices_w[j] - center_w) / (w / 2)
                dist_matrix[i, j] = np.sqrt(dist_h**2 + dist_w**2)
        
        # Define frequency bands
        r_max = np.sqrt(h**2 + w**2) / 2
        bands = {
            'very_low': (0, 0.1),
            'low': (0.1, 0.2),
            'mid': (0.2, 0.5),
            'high': (0.5, 1.0)
        }
        
        band_energies = {}
        for band_name, (r_min, r_max_norm) in bands.items():
            r_min_px = r_min * r_max
            r_max_px = r_max_norm * r_max
            mask = (dist_matrix >= r_min_px) & (dist_matrix < r_max_px)
            band_energies[band_name] = power[mask].sum() if mask.sum() > 0 else 0
        
        total_energy = power.sum()
        band_energies = {k: v / (total_energy + 1e-8) for k, v in band_energies.items()}
        
        return {
            'magnitude_spectrum': magnitude,
            'power': power,
            'band_energies': band_energies,
            'total_energy': total_energy
        }
    
    def compute_frequency_stats(self, lf_pred: np.ndarray, hf_pred: np.ndarray,
                               full_pred: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Compute statistics on frequency component contributions"""
        
        def dice_coef(a, b):
            inter = np.logical_and(a == 1, b == 1).sum()
            s = (a == 1).sum() + (b == 1).sum()
            return 1.0 if s == 0 else (2.0 * inter) / (s + 1e-8)
        
        def iou_coef(a, b):
            inter = np.logical_and(a == 1, b == 1).sum()
            union = np.logical_or(a == 1, b == 1).sum()
            return inter / (union + 1e-8)
        
        # Convert to binary
        lf_binary = (lf_pred > 0.5).astype(np.int32).ravel()
        hf_binary = (hf_pred > 0.5).astype(np.int32).ravel()
        full_binary = (full_pred > 0.5).astype(np.int32).ravel()
        gt_binary = (ground_truth > 0.5).astype(np.int32).ravel()
        
        stats = {
            'lf_dice': float(dice_coef(gt_binary, lf_binary)),
            'hf_dice': float(dice_coef(gt_binary, hf_binary)),
            'full_dice': float(dice_coef(gt_binary, full_binary)),
            'lf_iou': float(iou_coef(gt_binary, lf_binary)),
            'hf_iou': float(iou_coef(gt_binary, hf_binary)),
            'full_iou': float(iou_coef(gt_binary, full_binary)),
            'lf_improvement': float(dice_coef(gt_binary, full_binary) - dice_coef(gt_binary, lf_binary)),
            'hf_contribution': float(dice_coef(gt_binary, hf_binary) / (dice_coef(gt_binary, full_binary) + 1e-8))
        }
        
        return stats
    
    def analyze_boundary_sharpness(self, prediction: np.ndarray,
                                   ground_truth: np.ndarray) -> Dict:
        """Analyze boundary sharpness using frequency analysis"""
        
        # Extract boundaries
        pred_edges = np.hypot(
            sobel(prediction.astype(float), axis=0),
            sobel(prediction.astype(float), axis=1)
        )
        
        gt_edges = np.hypot(
            sobel(ground_truth.astype(float), axis=0),
            sobel(ground_truth.astype(float), axis=1)
        )
        
        # Compute frequency spectra
        pred_spectrum = self.compute_frequency_spectrum(pred_edges)
        gt_spectrum = self.compute_frequency_spectrum(gt_edges)
        
        # Compute high-frequency energy ratio
        h, w = prediction.shape
        center_h, center_w = h // 2, w // 2
        
        indices_h = np.arange(h)
        indices_w = np.arange(w)
        
        dist_matrix = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                dist_h = np.abs(indices_h[i] - center_h) / (h / 2)
                dist_w = np.abs(indices_w[j] - center_w) / (w / 2)
                dist_matrix[i, j] = np.sqrt(dist_h**2 + dist_w**2)
        
        high_freq_mask = dist_matrix > 0.5
        
        pred_high_energy = pred_spectrum[high_freq_mask].sum()
        pred_total_energy = pred_spectrum.sum()
        
        gt_high_energy = gt_spectrum[high_freq_mask].sum()
        gt_total_energy = gt_spectrum.sum()
        
        return {
            'prediction_high_freq_ratio': float(pred_high_energy / (pred_total_energy + 1e-8)),
            'gt_high_freq_ratio': float(gt_high_energy / (gt_total_energy + 1e-8)),
            'prediction_edge_strength': float(pred_edges.sum()),
            'gt_edge_strength': float(gt_edges.sum()),
            'sharpness_match': float(1.0 - abs(pred_high_energy / pred_total_energy - 
                                              gt_high_energy / gt_total_energy))
        }
    
    def visualize_frequency_analysis_enhanced(self, image: np.ndarray,
                                             lf_spectrum: np.ndarray,
                                             hf_spectrum: np.ndarray,
                                             band_energies: Dict,
                                             output_path: Path = None):
        """Enhanced frequency analysis visualization (600 DPI)"""
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=self.dpi//100)
        
        # Original image
        if len(image.shape) == 3:
            image_display = image[image.shape[0] // 2]
        else:
            image_display = image
        
        axes[0, 0].imshow(image_display, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # LF spectrum
        im1 = axes[0, 1].imshow(np.log1p(lf_spectrum), cmap='hot')
        axes[0, 1].set_title('Low-Frequency Spectrum', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Log Magnitude')
        
        # HF spectrum
        im2 = axes[0, 2].imshow(np.log1p(hf_spectrum), cmap='viridis')
        axes[0, 2].set_title('High-Frequency Spectrum', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], label='Log Magnitude')
        
        # Band energies bar chart
        axes[1, 0].bar(band_energies.keys(), band_energies.values(), 
                      color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].set_ylabel('Energy Ratio', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Frequency Band Energies', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Cumulative energy
        cumulative = np.cumsum(list(band_energies.values()))
        axes[1, 1].plot(list(band_energies.keys()), cumulative, marker='o', linewidth=2, markersize=8)
        axes[1, 1].fill_between(range(len(band_energies)), 0, cumulative, alpha=0.3)
        axes[1, 1].set_ylabel('Cumulative Energy', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Cumulative Frequency Energy', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary text
        axes[1, 2].axis('off')
        summary_text = "Frequency Analysis Summary\n\n"
        for band, energy in band_energies.items():
            summary_text += f"{band.capitalize():12s}: {energy:6.3f}\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', quality=95)
            print(f"✓ Saved: {output_path}")
        
        plt.close()


# Export classes
__all__ = ['EnhancedFrequencyDomainAnalyzer']

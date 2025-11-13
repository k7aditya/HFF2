# freq_analysis_CORRECTED.py
# FIXES FOR 3D VOLUME HANDLING IN FREQUENCY ANALYSIS

"""
CORRECTED FREQUENCY ANALYSIS - Handles 3D BraTS Data
Key fixes:
1. Extract 2D slices from 3D volumes before visualization
2. Proper type casting for cv2 operations
3. Matplotlib compatibility for 2D images only
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


class EnhancedFrequencyDomainAnalyzer:
    """Comprehensive frequency domain analysis with 3D volume handling"""
    
    def __init__(self, device: str = 'cuda', save_dir: str = 'results/figures/xai/freq', dpi: int = 600):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def extract_frequency_components_3d(self, image_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract frequency components from 3D image using 3D FFT"""
        
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
        cutoff = 0.3
        
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
        """Extract frequency components from 2D image"""
        
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
        """Compute frequency spectrum magnitude (log scale)"""
        
        if image.ndim == 3:
            # Use middle slice for 3D
            image = image[image.shape[0] // 2]
        
        fft_image = fft2(image)
        fft_shift = fftshift(fft_image)
        magnitude = np.abs(fft_shift)
        spectrum = np.log1p(magnitude)
        
        return spectrum
    
    def visualize_frequency_analysis_enhanced(self, image: np.ndarray,
                                             lf_spectrum: np.ndarray,
                                             hf_spectrum: np.ndarray,
                                             band_energies: Dict,
                                             output_path: Path = None):
        """Enhanced frequency analysis visualization (600 DPI)"""
        
        # ===== FIX 10: Handle 3D volumes =====
        if image.ndim == 3:
            image_display = image[image.shape[0] // 2]
        else:
            image_display = image
        
        # ===== FIX 11: Handle 3D spectra =====
        if lf_spectrum.ndim == 3:
            lf_spectrum = lf_spectrum[lf_spectrum.shape[0] // 2]
        if hf_spectrum.ndim == 3:
            hf_spectrum = hf_spectrum[hf_spectrum.shape[0] // 2]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=self.dpi//100)
        
        # Original image
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

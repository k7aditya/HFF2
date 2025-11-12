"""
MC-Dropout Uncertainty Estimation Module for HFF-Net (Fixed for 3D BraTS)
Ensures dropout stays active and model is in proper train mode during inference
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import cv2


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation in 3D segmentation
    Performs multiple stochastic forward passes with dropout ACTIVE
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 20, 
                 device: str = 'cuda', save_dir: str = 'results/figures/uncertainty'):
        self.model = model
        self.num_samples = num_samples
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.dropout_layers = []
        self._prepare_model()
    
    def _prepare_model(self):
        """Identify all dropout layers for MC sampling"""
        self.dropout_layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                self.dropout_layers.append(module)
        
        if len(self.dropout_layers) == 0:
            print("WARNING: No dropout layers found in model!")
        else:
            print(f"Found {len(self.dropout_layers)} dropout layers for MC sampling")
    
    def enable_dropout_inference(self):
        """
        CRITICAL: Enable dropout during inference
        - Sets dropout layers to train mode
        - Ensures dropout probability > 0
        """
        for layer in self.dropout_layers:
            layer.train()  # Enable dropout
            # Verify dropout rate is not zero
            if hasattr(layer, 'p') and layer.p < 0.05:
                print(f"WARNING: Dropout rate very low: {layer.p}")
    
    def disable_dropout_inference(self):
        """Disable dropout after MC sampling"""
        for layer in self.dropout_layers:
            layer.eval()
    
    def mc_forward_pass(self, input1: torch.Tensor, input2: torch.Tensor) -> List[torch.Tensor]:
        """
        Perform N stochastic forward passes with dropout ACTIVE
        
        CRITICAL FIXES:
        1. Set ENTIRE model to train() mode first
        2. Then enable dropout layers explicitly
        3. Use torch.no_grad() to prevent gradient accumulation
        4. Handle tuple output from model
        
        Args:
            input1: Low-frequency input (B, C_lf, D, H, W)
            input2: High-frequency input (B, C_hf, D, H, W)
            
        Returns:
            List of N prediction tensors
        """
        outputs = []
        
        # CRITICAL: Set model to train mode to enable dropout AND batchnorm stochasticity
        was_training = self.model.training
        self.model.train()
        
        # Double-ensure dropout layers are active
        self.enable_dropout_inference()
        
        # Verify dropout is actually active
        active_dropout_count = sum(1 for layer in self.dropout_layers if layer.training)
        if active_dropout_count == 0:
            print("ERROR: No dropout layers are in training mode!")
        
        with torch.no_grad():
            for sample_idx in range(self.num_samples):
                # Forward pass with dropout active
                output = self.model(input1, input2)
                
                # Handle tuple output (HFF-Net returns tuple)
                if isinstance(output, tuple):
                    output = output[0]  # Take first output (main prediction)
                
                outputs.append(output.cpu())
                
                # Debug: Check variance every 5 samples
                if sample_idx > 0 and sample_idx % 5 == 0:
                    stacked = torch.stack(outputs[:sample_idx+1])
                    var = stacked.var(dim=0).mean().item()
                    print(f"  MC sample {sample_idx+1}/{self.num_samples}, variance so far: {var:.6f}")
        
        # Restore original model state
        self.disable_dropout_inference()
        if not was_training:
            self.model.eval()
        
        # Final variance check
        if len(outputs) > 1:
            final_stack = torch.stack(outputs)
            final_var = final_stack.var(dim=0).mean().item()
            final_std = final_stack.std(dim=0).mean().item()
            print(f"Final MC-Dropout: mean variance={final_var:.6f}, mean std={final_std:.6f}")
            
            if final_var < 1e-8:
                print("ERROR: MC-Dropout variance is near zero! Dropout may not be active.")
        
        return outputs
    
    def compute_uncertainty_maps(self, outputs: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean prediction and uncertainty from MC samples
        
        Args:
            outputs: List of N segmentation outputs (B, C, D, H, W)
            
        Returns:
            Tuple of (mean_prediction, uncertainty_map)
        """
        if len(outputs) == 0:
            raise ValueError("No outputs provided")
        
        outputs = torch.stack(outputs, dim=0)  # (N, B, C, D, H, W)
        
        # Compute mean prediction
        mean_pred = outputs.mean(dim=0)  # (B, C, D, H, W)
        
        # Compute uncertainty as predictive variance
        variance = outputs.var(dim=0)  # (B, C, D, H, W)
        
        # Aggregate across classes: average variance
        uncertainty = variance.mean(dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        return mean_pred.numpy(), uncertainty.numpy()


class DropoutScheduler:
    """
    Manage dropout rates during training
    
    CRITICAL FIX: Maintain minimum dropout rate for MC-Dropout uncertainty
    """
    
    def __init__(self, model: nn.Module, base_dropout: float = 0.5, min_dropout: float = 0.2):
        self.model = model
        self.base_dropout = base_dropout
        self.min_dropout = min_dropout  # NEVER drop below this
        self.dropout_layers = []
        
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                self.dropout_layers.append(module)
    
    def set_dropout_rate(self, rate: float):
        """
        Set dropout rate for all Dropout layers
        
        CRITICAL: Enforce minimum dropout rate for uncertainty estimation
        """
        rate = max(rate, self.min_dropout)  # NEVER go below min_dropout
        
        for module in self.dropout_layers:
            module.p = rate
        
        print(f"Set dropout rate to {rate:.4f} ({len(self.dropout_layers)} layers)")
    
    def enable_mc_dropout(self):
        """Enable MC-Dropout for inference (sets to train mode)"""
        for module in self.dropout_layers:
            module.train()
    
    def disable_mc_dropout(self):
        """Disable dropout for standard inference"""
        for module in self.dropout_layers:
            module.eval()


# Export
__all__ = ['MCDropoutUncertainty', 'DropoutScheduler']

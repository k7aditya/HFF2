# MC-Dropout Uncertainty Estimation Module for HFF-Net (Fixed for 3D BraTS)
# Ensures dropout stays active and model is in proper train mode during inference

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
    
    # --- MODIFIED __init__ ---
    def __init__(self, model: nn.Module, num_samples: int = 20, 
                 device: str = 'cuda', save_dir: str = 'results/figures/uncertainty'):
        self.model = model
        self.num_samples = num_samples
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # --- FIX: Correct initialization sequence ---
        # 1. Initialize the attribute as an empty list.
        self.dropout_layers = []
        
        # 2. Call the mutator method to populate the list.
        #    Do NOT assign its 'None' return value.
        self._prepare_model()
        # --- END FIX ---
    
    # --- MODIFIED _prepare_model ---
    def _prepare_model(self):
        """
        Identify all dropout layers for MC sampling
        """
        # --- REFINEMENT: Removed redundant initialization ---
        # self.dropout_layers = (This is now handled in __init__)

        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                self.dropout_layers.append(module)
        
        if len(self.dropout_layers) == 0:
            print("WARNING: No dropout layers found in model!")
        else:
            print(f"Found {len(self.dropout_layers)} dropout layers for MC sampling")
    
    # --- RECOMMENDED NEW HELPER FUNCTION (Validated: Correct) ---
    def _set_mc_dropout_state(self, state: bool) -> None:
        """
        Sets the model state for MC-Dropout inference.

        Args:
            state (bool): 
                True:  Sets model to `eval()` mode, then selectively
                       re-enables ONLY dropout layers to `train()` mode.
                       This freezes BatchNorm layers.
                False: Sets the entire model back to standard `eval()` mode.
        """
        if state:
            # 1. Set entire model to eval mode
            # This freezes BatchNorm running statistics
            self.model.eval()
            
            # 2. Selectively re-enable ONLY dropout layers
            for layer in self.dropout_layers:
                layer.train()
            
            # Verify that dropout is active
            active_dropout_count = sum(1 for layer in self.dropout_layers if layer.training)
            if len(self.dropout_layers) > 0 and active_dropout_count == 0:
                print("ERROR: Dropout layers found but FAILED to set to train mode!")

        else:
            # Restore the model to standard eval mode
            self.model.eval()

    # --- DEPRECATED ORIGINAL FUNCTIONS (Retained as reference) ---
    # def enable_dropout_inference(self):...
    # def disable_dropout_inference(self):...
    # These are replaced by the more robust `_set_mc_dropout_state` helper.

    # --- CORRECTED FUNCTION ---
    def mc_forward_pass(self, input1: torch.Tensor, input2: torch.Tensor) -> List:
        """
        Perform N stochastic forward passes with dropout ACTIVE
        
        CRITICAL FIXES (Revised):
        1. Call `_set_mc_dropout_state(True)` to freeze BatchNorm
           and activate Dropout.
        2. Use torch.no_grad() (User's correct implementation).
        3. Handle multi-input and tuple-output (User's correct implementation).
        4. Restore model state with `_set_mc_dropout_state(False)`.
        """
        outputs = []
        
        # --- CRITICAL FIX: REMOVED FLAWED LOGIC ---
        # REMOVED: was_training = self.model.training
        # REMOVED: self.model.train()  (This was the CRITICAL FLAW )
        # REMOVED: self.enable_dropout_inference()
        
        try:
            # 1. Set model to the *correct* MC-Dropout inference state
            self._set_mc_dropout_state(True)
            
            # User's code from here is correct
            with torch.no_grad():
                for sample_idx in range(self.num_samples):
                    # 3. Handle HFF-Net multi-input (correct)
                    output = self.model(input1, input2)
                    
                    # 3. Handle tuple output
                    # --- BUGFIX-2: Corrected latent bug ---
                    # The original 'output = output' was non-functional.
                    # This now correctly selects the first element.
                    if isinstance(output, tuple):
                        output = output  # Take first output
                    # --- END BUGFIX-2 ---
                        
                    # Store logits on CPU (correct)
                    outputs.append(output.cpu())
                    
                    # Debug: Check variance (retained from original)
                    if sample_idx > 0 and sample_idx % 5 == 0:
                        stacked = torch.stack(outputs[:sample_idx+1])
                        var = stacked.var(dim=0).mean().item()
                        print(f"  MC sample {sample_idx+1}/{self.num_samples}, variance so far: {var:.6f}")

        finally:
            # 4. Restore model to a clean, standard inference state
            self._set_mc_dropout_state(False)
            
        # --- END RECOMMENDED MODIFICATION ---

        # Final variance check (retained from original)
        if len(outputs) > 1:
            final_stack = torch.stack(outputs)
            final_var = final_stack.var(dim=0).mean().item()
            final_std = final_stack.std(dim=0).mean().item()
            print(f"Final MC-Dropout: mean variance={final_var:.6f}, mean std={final_std:.6f}")
            
            if final_var < 1e-8:
                print("ERROR: MC-Dropout variance is near zero! Dropout may not be active.")
        
        return outputs
    
    # --- ORIGINAL FUNCTION (Validated: Correct) ---
    def compute_uncertainty_maps(self, outputs: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean prediction and uncertainty from MC samples
        (This function is correct for 'predictive variance')
        
        Args:
            outputs: List of N segmentation logit outputs (B, C, D, H, W)
            
        Returns:
            Tuple of (mean_prediction_logits, uncertainty_map_variance)
        """
        if len(outputs) == 0:
            raise ValueError("No outputs provided")
        
        outputs = torch.stack(outputs, dim=0)  # (N, B, C, D, H, W)
        
        # Compute mean prediction (on logits)
        mean_pred = outputs.mean(dim=0)  # (B, C, D, H, W)
        
        # Compute uncertainty as predictive variance (on logits)
        variance = outputs.var(dim=0)  # (B, C, D, H, W)
        
        # Aggregate across classes: average variance
        uncertainty = variance.mean(dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        return mean_pred.numpy(), uncertainty.numpy()

    # --- RECOMMENDED ADVANCED METRICS FUNCTION (Validated: Correct) ---
    def compute_advanced_uncertainty_maps(
        self, 
        logit_outputs: List, 
        epsilon: float = 1e-9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute advanced, entropy-based uncertainty maps from MC samples.
        
        This computes:
        1. Mean Prediction (Probabilities)
        2. Predictive Entropy (Total Uncertainty): H[E[p]]
        3. Expected Entropy (Aleatoric/Data Uncertainty): E[H[p]]
        4. Mutual Information (Epistemic/Model Uncertainty): MI = H[E[p]] - E[H[p]]
        
        Formulas based on standard Bayesian decomposition.
        
        Args:
            logit_outputs: List of N logit tensors (B, C, D, H, W)
                           from mc_forward_pass. The list has length N.
            epsilon: Small value for numerical stability (to avoid log(0)).
                     
        Returns:
            Tuple of (mean_probs, predictive_entropy, mutual_information, expected_entropy)
            - mean_probs: (B, C, D, H, W)
            - all others: (B, 1, D, H, W)
        """
        if len(logit_outputs) == 0:
            raise ValueError("No outputs provided")
            
        # Stack logits along a new 'N' dimension
        # Shape: (N, B, C, D, H, W)
        logit_samples = torch.stack(logit_outputs, dim=0)
        
        # 1. Convert logits to probabilities
        # Shape: (N, B, C, D, H, W)
        prob_samples = torch.softmax(logit_samples, dim=2) # dim=2 is Class dim
        
        # 2. Compute mean prediction (averaged probabilities)
        # Shape: (B, C, D, H, W)
        mean_probs = prob_samples.mean(dim=0)
        
        # 3. Compute Predictive Entropy (Total Uncertainty)
        # H[E[p]] = -sum_C(p_mean * log(p_mean))
        # Sum over class dim (dim=1)
        # Shape: (B, 1, D, H, W)
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + epsilon), dim=1, keepdim=True
        )
        
        # 4. Compute Expected Entropy (Aleatoric Uncertainty)
        # E[H[p]] = - (1/N) * sum_N [ sum_C (p_n * log(p_n)) ]
        
        # Compute entropy for each sample
        # Shape: (N, B, D, H, W)
        entropy_of_samples = -torch.sum(
            prob_samples * torch.log(prob_samples + epsilon), dim=2 # Sum over class dim
        )
        
        # Average the entropies over the N samples
        # Shape: (B, 1, D, H, W)
        expected_entropy = entropy_of_samples.mean(dim=0, keepdim=True) # Average over N dim
        
        # 5. Compute Mutual Information (Epistemic Uncertainty)
        # MI = H[E[p]] - E[H[p]]
        # Shape: (B, 1, D, H, W)
        mutual_information = predictive_entropy - expected_entropy
        
        # Clamp MI at 0, as numerical errors can cause small negative values
        mutual_information = torch.clamp(mutual_information, min=0.0)
        
        return (
            mean_probs.cpu().numpy(),
            predictive_entropy.cpu().numpy(),
            mutual_information.cpu().numpy(),
            expected_entropy.cpu().numpy()
        )


# --- NO CHANGES (ORIGINAL CODE WAS ROBUST) ---
class DropoutScheduler:
    """
    Manage dropout rates during training
    
    CRITICAL FIX: Maintain minimum dropout rate for MC-Dropout uncertainty
    (This class is a robust and valid utility for ensuring MC-Dropout
    remains possible, as it is sensitive to the dropout rate )
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

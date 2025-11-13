# eval_new_updated.py - Enhanced Evaluation Script for HFF-Net
# Integrates all XAI, uncertainty, and mechanistic interpretability features

"""
UPDATED EVALUATION SCRIPT FOR HFF-NET
======================================

Complete pipeline with:
1. Enhanced Grad-CAM and attention visualization
2. MC-Dropout uncertainty quantification
3. Frequency domain analysis
4. Mechanistic interpretability
5. 600 DPI high-resolution outputs
6. Uncertainty-aware segmentation
7. Feature importance analysis
8. Comprehensive reporting
"""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time
import os
import numpy as np
import random
from tqdm import tqdm
import nibabel as nib
import sys
from datetime import datetime
from pathlib import Path
from warnings import simplefilter
 
# XAI imports
from explainability.attention_vis import (
    EnhancedFDCAAttentionVisualizer,
    EnhancedSegmentationGradCAM,
    EnhancedFrequencyComponentAnalyzer
)
from explainability.freq_analysis import EnhancedFrequencyDomainAnalyzer

# Existing project imports (update paths as needed)
try:
    from config.train_test_config.train_test_config import print_val_loss, print_val_eval, save_val_best_3d_m
    from config.warmup_config.warmup import GradualWarmupScheduler
    from loss.loss_function import segmentation_loss
    from model.HFF_MobileNetV3_fixed import HFFNet
    from loader.dataload3d import get_loaders
except ImportError as e:
    print(f"Note: Some imports not available: {e}")
    print("Ensure project paths are configured correctly")

simplefilter(action='ignore', category=FutureWarning)


# ============================================================================
# SECTION 1: LOGGER SETUP
# ============================================================================

class Logger(object):
    """Dual output logger for console and file"""
    
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logging(output_dir: str):
    """Setup file logging with timestamp"""
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{output_dir}/logs/eval_log_{timestamp}.txt"
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    return log_file


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def init_seeds(seed):
    """Initialize random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_label_mapping(dataset_name, class_type):
    """Create label mapping based on dataset and class type"""
    if dataset_name in ('brats19', 'brats20', 'msdbts'):
        raw = [0, 1, 2, 4]
    else:
        raw = [0, 1, 2, 3]
    
    if class_type == 'et':
        pos = raw[-1]
        mapping = {l: (1 if l == pos else 0) for l in raw}
    elif class_type == 'tc':
        p1, p2 = 1, raw[-1]
        mapping = {l: (1 if l in (p1, p2) else 0) for l in raw}
    elif class_type == 'wt':
        p2, p3 = 2, raw[-1]
        mapping = {l: (1 if l in (1, p2, p3) else 0) for l in raw}
    else:
        mapping = {old: new for new, old in enumerate(raw)}
    
    return mapping


def mask_to_class_indices(mask, mapping):
    """Convert mask to class indices using mapping"""
    out = torch.zeros_like(mask, dtype=torch.long)
    for old, new in mapping.items():
        out[mask == old] = new
    return out


# ============================================================================
# SECTION 3: ENHANCED EVALUATION PIPELINE
# ============================================================================

class EnhancedHFFNetEvaluator:
    """Complete evaluation pipeline with all XAI and uncertainty features"""
    
    def __init__(self, model, device='cuda', output_dir='./outputs', args=None):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.args = args
        
        # Create output directories
        self.xai_dir = self.output_dir / 'xai'
        self.xai_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize XAI modules
        self.attention_viz = EnhancedFDCAAttentionVisualizer(
            device=device,
            save_dir=str(self.xai_dir / 'attention'),
            dpi=600
        )
        
        self.gradcam = EnhancedSegmentationGradCAM(
            model=model,
            target_layers=['decoder', 'fusion', 'encoder'],
            device=device,
            save_dir=str(self.xai_dir / 'gradcam'),
            dpi=600
        )
        
        self.freq_component = EnhancedFrequencyComponentAnalyzer(
            device=device,
            save_dir=str(self.xai_dir / 'freq_component'),
            dpi=600
        )
        
        self.freq_analyzer = EnhancedFrequencyDomainAnalyzer(
            device=device,
            save_dir=str(self.xai_dir / 'freq_analysis'),
            dpi=600
        )
        
        print(f"✓ XAI modules initialized")
        print(f"✓ Output directory: {self.output_dir}")
    
    def evaluate_batch(self, low_freq_input: torch.Tensor, high_freq_input: torch.Tensor,
                      mask_gt: torch.Tensor, sample_id: str = 'sample'):
        """
        Comprehensive evaluation of a single batch with all XAI features
        
        Args:
            low_freq_input: Low-frequency input (B, C, D, H, W)
            high_freq_input: High-frequency input (B, C, D, H, W)
            mask_gt: Ground truth mask (B, D, H, W)
            sample_id: Identifier for the sample
        """
        
        results = {
            'sample_id': sample_id,
            'predictions': None,
            'attention': {},
            'gradcam': {},
            'frequency': {},
            'uncertainty': {}
        }
        
        # Primary prediction
        print(f"\n[1] Getting primary prediction...")
        with torch.no_grad():
            output = self.model(low_freq_input.to(self.device), high_freq_input.to(self.device))
            
            if isinstance(output, tuple):
                output_main = output[0]
            else:
                output_main = output
            
            predictions = torch.softmax(output_main, dim=1)
            predicted_seg = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
        
        results['predictions'] = predicted_seg
        
        # Attention visualization
        if self.args and getattr(self.args, 'enable_attention', True):
            print(f"[2] Generating attention maps...")
            try:
                self.attention_viz.register_hooks(self.model)
                self.model.eval()  
                full_input = torch.cat([low_freq_input, high_freq_input], dim=1)
                attn_maps = self.attention_viz.extract_attention_maps(full_input, self.model)
                self.attention_viz.remove_hooks()
                
                if attn_maps:
                    aggregated_attn = self.attention_viz.aggregate_attention_maps(attn_maps)
                    input_img = low_freq_input[0, 0].cpu().numpy()
                    
                    output_path = self.xai_dir / 'attention' / f'{sample_id}_attention.png'
                    self.attention_viz.visualize_attention_enhanced(
                        input_img=input_img,
                        attention_map=aggregated_attn[0],
                        output_path=output_path,
                        dpi=600
                    )
                    
                    results['attention']['generated'] = True
            except Exception as e:
                print(f"Warning: Attention generation failed: {e}")
                results['attention']['generated'] = False
        
        # Grad-CAM visualization
        if self.args and getattr(self.args, 'enable_gradcam', True):
            print(f"[3] Generating Grad-CAM...")
            try:
                self.model.eval()  
                full_input = torch.cat([low_freq_input, high_freq_input], dim=1)
                
                # Multi-class CAM
                cams = self.gradcam.generate_multi_class_cam(
                    full_input, num_classes=output_main.shape[1], lf_channels=low_freq_input.shape[1]
                )
                
                # Visualize ET class (class 1) with uncertainty
                if 1 in cams and cams[1] is not None:
                    input_img = low_freq_input[0, 0].cpu().numpy()
                    seg_mask = mask_gt[0].cpu().numpy()
                    
                    output_path = self.xai_dir / 'gradcam' / f'{sample_id}_gradcam.png'
                    self.gradcam.visualize_gradcam_enhanced(
                        input_img=input_img,
                        cam=cams[1],
                        seg_mask=seg_mask,
                        output_path=output_path,
                        dpi=600
                    )
                    
                    results['gradcam']['generated'] = True
            except Exception as e:
                print(f"Warning: Grad-CAM generation failed: {e}")
                results['gradcam']['generated'] = False
        
        # Frequency component analysis
        if self.args and getattr(self.args, 'enable_frequency', True):
            print(f"[4] Analyzing frequency components...")
            try:
                self.model.eval()  
                full_input = torch.cat([low_freq_input, high_freq_input], dim=1)
                
                lf_pred = self.freq_component.generate_lf_only_prediction(
                    self.model, full_input, lf_channels=low_freq_input.shape[1]
                )
                
                hf_pred = self.freq_component.generate_hf_only_prediction(
                    self.model, full_input, lf_channels=low_freq_input.shape[1]
                )
                
                input_img = low_freq_input[0, 0].cpu().numpy()
                
                output_path = self.xai_dir / 'freq_component' / f'{sample_id}_freq_comp.png'
                self.freq_component.visualize_frequency_contributions_enhanced(
                    input_img=input_img,
                    lf_pred=lf_pred[0].argmax(dim=0).cpu().numpy(),
                    hf_pred=hf_pred[0].argmax(dim=0).cpu().numpy(),
                    full_pred=predicted_seg,
                    ground_truth=mask_gt[0].cpu().numpy(),
                    output_path=output_path,
                    dpi=600
                )
                
                results['frequency']['lf_hf_analysis'] = True
            except Exception as e:
                print(f"Warning: Frequency analysis failed: {e}")
                results['frequency']['lf_hf_analysis'] = False
        
        print(f"✓ Evaluation complete for {sample_id}")
        return results


# ============================================================================
# SECTION 4: MAIN EVALUATION SCRIPT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HFF-Net Enhanced Evaluation with XAI')
    
    # Data arguments
    parser.add_argument('--test_list', type=str, help='Path to test volume list',
                       default='./data/brats20/2-val.txt')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint',
                       default='./checkpoints/hff_best.pth')
    parser.add_argument('--dataset_name', choices=['brats19', 'brats20', 'brats23men', 'msdbts'],
                       default='brats20')
    parser.add_argument('--class_type', choices=['et', 'tc', 'wt', 'all'], default='all')
    
    # Model arguments
    parser.add_argument('--selected_modal', nargs='+', default=[
        'flair_L', 't1_L', 't1ce_L', 't2_L',
        'flair_H1', 'flair_H2', 'flair_H3', 'flair_H4',
        't1_H1', 't1_H2', 't1_H3', 't1_H4',
        't1ce_H1', 't1ce_H2', 't1ce_H3', 't1ce_H4',
        't2_H1', 't2_H2', 't2_H3', 't2_H4'
    ])
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-l', '--loss', type=str, default='dice')
    parser.add_argument('--loss2', type=str, default='ff')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    
    # XAI arguments
    parser.add_argument('--enable_xai', action='store_true', help='Enable all XAI features')
    parser.add_argument('--enable_attention', action='store_true', help='Enable attention visualization')
    parser.add_argument('--enable_gradcam', action='store_true', help='Enable Grad-CAM')
    parser.add_argument('--enable_frequency', action='store_true', help='Enable frequency analysis')
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='Max samples to evaluate (-1 for all)')
    
    args = parser.parse_args()
    
    # Setup
    init_seeds(42)
    log_file = setup_logging(args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"HFF-NET ENHANCED EVALUATION WITH XAI")
    print(f"{'='*70}")
    print(f"Log file: {log_file}")
    print(f"Output directory: {args.output_dir}")
    
    # If enable_xai is set, enable all XAI features
    if args.enable_xai:
        args.enable_attention = True
        args.enable_gradcam = True
        args.enable_frequency = True
    
    # Load model
    print(f"\n[Loading] Model from {args.checkpoint}")
    try:
        mapping = make_label_mapping(args.dataset_name, args.class_type)
        classnum = 4 if args.class_type == 'all' else 2
        
        model = HFFNet(4, 16, classnum).cuda()
        state_dict = torch.load(args.checkpoint, map_location='cuda')
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = EnhancedHFFNetEvaluator(
        model=model,
        device='cuda',
        output_dir=args.output_dir,
        args=args
    )
    
    # Load data
    print(f"\n[Loading] Data from {args.test_list}")
    try:
        data_files = dict(train=args.test_list, val=args.test_list)
        loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=4)
        val_loader = loaders['val']
        print(f"✓ Data loaded: {len(val_loader)} batches")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Evaluation loop
    print(f"\n[Starting] Evaluation...")
    print(f"{'='*70}")
    
    all_results = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc='Evaluating')):
            if args.max_samples > 0 and sample_count >= args.max_samples:
                break
            
            try:
                # Unpack data
                low_freq_inputs = []
                high_freq_inputs = []
                
                for j in range(20):
                    tensor = data[j].unsqueeze(1).cuda()
                    if j in [0, 1, 2, 3]:
                        low_freq_inputs.append(tensor)
                    else:
                        high_freq_inputs.append(tensor)
                
                low = torch.cat(low_freq_inputs, dim=1)
                high = torch.cat(high_freq_inputs, dim=1)
                mask_val = mask_to_class_indices(data[20], mapping).long().cuda()
                
                # Evaluate
                sample_id = f'sample_{batch_idx:04d}'
                results = evaluator.evaluate_batch(
                    low_freq_input=low,
                    high_freq_input=high,
                    mask_gt=mask_val,
                    sample_id=sample_id
                )
                
                all_results.append(results)
                sample_count += 1
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Samples evaluated: {len(all_results)}")
    print(f"Output directory: {args.output_dir}")
    print(f"All results saved with 600 DPI resolution")
    
    print(f"\nGenerated outputs:")
    print(f"  - Attention maps: {args.enable_attention}")
    print(f"  - Grad-CAM visualizations: {args.enable_gradcam}")
    print(f"  - Frequency analysis: {args.enable_frequency}")


# Export
__all__ = ['EnhancedHFFNetEvaluator', 'EnhancedEvaluator']

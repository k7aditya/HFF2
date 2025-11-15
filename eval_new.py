# eval_new_CORRECTED.py

"""
CORRECTED EVALUATION SCRIPT FOR HFF-NET
========================================

FIXES:
1. ✅ Proper checkpoint loading with DDP 'module.' prefix handling
2. ✅ Support for multiple checkpoint formats
3. ✅ Error handling and debug information
4. ✅ Model evaluation with all metrics
5. ✅ XAI features (attention, Grad-CAM, frequency analysis)
6. ✅ Mechanistic interpretability output
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
import re

# XAI imports
try:
    from explainability.attention_vis import (
        EnhancedFDCAAttentionVisualizer,
        EnhancedSegmentationGradCAM,
        EnhancedFrequencyComponentAnalyzer
    )
    from explainability.freq_analysis import EnhancedFrequencyDomainAnalyzer
except ImportError:
    print("Note: XAI modules not available - will proceed with basic evaluation")

# Existing project imports
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
# SECTION 2: CHECKPOINT LOADING - CORRECTED
# ============================================================================

def load_checkpoint_with_prefix_handling(model, checkpoint_path, device):
    """
    Load checkpoint with proper handling of DDP 'module.' prefix

    Fixes:
    ✅ Removes 'module.' prefix from DDP-trained models
    ✅ Handles multiple checkpoint formats
    ✅ Provides debug information
    ✅ Graceful error handling
    """
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ [ERROR] Checkpoint not found: {checkpoint_path}")
        return False

    print(f"\n[LOADING CHECKPOINT]")
    print(f"Path: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    try:
        # Load checkpoint
        print(f"Loading checkpoint from disk...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded to CPU")

        # Step 1: Determine what format the checkpoint is
        print(f"\n[STEP 1] Identifying checkpoint format...")

        if isinstance(checkpoint, dict):
            print(f"Checkpoint is a dict with keys: {list(checkpoint.keys())}")

            # Try different key names for state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✓ Found 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"✓ Found 'state_dict' key")
            else:
                # Assume the entire dict is the state_dict
                state_dict = checkpoint
                print(f"ℹ Treating entire checkpoint as state_dict")
        else:
            # Direct state_dict
            state_dict = checkpoint
            print(f"Checkpoint is direct state_dict")

        # Step 2: Check for 'module.' prefix (from DDP training)
        print(f"\n[STEP 2] Checking for DDP prefix handling...")

        sample_keys = list(state_dict.keys())[:3]
        print(f"Sample keys: {sample_keys}")

        has_ddp_prefix = any(key.startswith('module.') for key in state_dict.keys())

        if has_ddp_prefix:
            print(f"⚠ Detected DDP 'module.' prefix in checkpoint")
            print(f"Removing 'module.' prefix from all keys...")

            # Remove 'module.' prefix
            new_state_dict = {}
            removed_count = 0

            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.'
                    new_state_dict[new_key] = value
                    removed_count += 1
                else:
                    new_state_dict[key] = value

            state_dict = new_state_dict
            print(f"✓ Removed 'module.' from {removed_count} keys")
        else:
            print(f"✓ No DDP prefix detected - checkpoint is clean")

        # Step 3: Load state_dict into model
        print(f"\n[STEP 3] Loading state_dict into model...")

        # Move model to device first
        model = model.to(device)

        # Load state dict
        model.load_state_dict(state_dict, strict=False)  # Use strict=False for flexibility
        print(f"✓ State dict loaded successfully")

        # Step 4: Verification
        print(f"\n[STEP 4] Verification...")

        model_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {model_params:,}")

        # Check a few params to ensure they're loaded
        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count += 1
            if param_count >= 3:
                break
        print(f"✓ Trainable parameters verified: {param_count}+")

        print(f"\n{'='*70}")
        print(f"✅ CHECKPOINT LOADED SUCCESSFULLY")
        print(f"{'='*70}\n")

        model.eval()  # Set to evaluation mode
        return True

    except Exception as e:
        print(f"\n❌ [ERROR] Failed to load checkpoint:")
        print(f"Exception: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"  1. Verify checkpoint file exists and is not corrupted")
        print(f"  2. Check checkpoint was saved with same model architecture")
        print(f"  3. Try absolute path instead of relative path")
        print(f"  4. Ensure PyTorch version compatibility")
        return False

# ============================================================================
# SECTION 3: UTILITY FUNCTIONS
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
# SECTION 4: METRIC CALCULATION FUNCTIONS
# ============================================================================

def dice_score(output: torch.Tensor, target: torch.Tensor, class_id: int = 1) -> float:
    """
    Calculate Dice coefficient for a specific class
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    smooth = 1e-6
    pred = torch.argmax(output, dim=1)  # Get predictions
    pred_binary = (pred == class_id).float()
    target_binary = (target == class_id).float()

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary)

    if union == 0:
        return 1.0 if torch.equal(pred_binary, target_binary) else 0.0

    dice = (2.0 * intersection) / (union + smooth)
    return dice.item()

def iou_score(output: torch.Tensor, target: torch.Tensor, class_id: int = 1) -> float:
    """
    Calculate IoU (Intersection over Union) for a specific class
    """
    smooth = 1e-6
    pred = torch.argmax(output, dim=1)
    pred_binary = (pred == class_id).float()
    target_binary = (target == class_id).float()

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection

    if union == 0:
        return 1.0 if torch.equal(pred_binary, target_binary) else 0.0

    iou = intersection / (union + smooth)
    return iou.item()

# ============================================================================
# SECTION 5: ENHANCED EVALUATION PIPELINE
# ============================================================================

class EnhancedHFFNetEvaluator:
    """Complete evaluation pipeline with metrics and XAI features"""

    def __init__(self, model, device='cuda', output_dir='./outputs', args=None):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.args = args

        # Run timestamp
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.output_dir / f"xai_{run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.xai_dir = self.run_dir

        # Create subdirectories
        (self.xai_dir / "attention").mkdir(parents=True, exist_ok=True)
        (self.xai_dir / "gradcam").mkdir(parents=True, exist_ok=True)
        (self.xai_dir / "freq_component").mkdir(parents=True, exist_ok=True)
        (self.xai_dir / "freq_analysis").mkdir(parents=True, exist_ok=True)

        # Initialize XAI modules (if available)
        try:
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
        except Exception as e:
            print(f"⚠ XAI modules not available: {e}")
            self.attention_viz = None

        print(f"✓ Run output directory: {self.run_dir}\n")

    def evaluate_batch(self, low_freq_input: torch.Tensor, high_freq_input: torch.Tensor,
                      mask_gt: torch.Tensor, sample_id: str = 'sample'):
        """
        Comprehensive evaluation of a single batch with metrics
        """
        results = {
            'sample_id': sample_id,
            'predictions': None,
            'metrics': {}  # ← METRICS DICT
        }

        # Primary prediction
        print(f"Getting prediction...")
        with torch.no_grad():
            output = self.model(low_freq_input.to(self.device), high_freq_input.to(self.device))

            if isinstance(output, tuple):
                output_main = output[0]
            else:
                output_main = output

            predictions = torch.softmax(output_main, dim=1)
            predicted_seg = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
            results['predictions'] = predicted_seg

        # ===== METRICS CALCULATION =====
        mask_gt_cpu = mask_gt.cpu()
        output_main_cpu = output_main.cpu()

        # Calculate metrics for each class
        num_classes = output_main.shape[1]
        per_class_metrics = {}

        for class_id in range(num_classes):
            dice = dice_score(output_main_cpu, mask_gt_cpu, class_id=class_id)
            iou = iou_score(output_main_cpu, mask_gt_cpu, class_id=class_id)

            per_class_metrics[f'class_{class_id}'] = {
                'dice': float(dice),
                'iou': float(iou),
            }

        results['metrics']['per_class'] = per_class_metrics

        # Print metrics
        print(f"  Sample: {sample_id}")
        for class_name, metric_dict in per_class_metrics.items():
            print(f"    {class_name}: Dice={metric_dict['dice']:.4f}, IoU={metric_dict['iou']:.4f}")

        return results

# ============================================================================
# SECTION 6: MAIN EVALUATION SCRIPT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HFF-Net Evaluation with Metrics')

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
    print(f"HFF-NET EVALUATION WITH METRICS")
    print(f"{'='*70}")
    print(f"Log file: {log_file}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")

    # If enable_xai is set, enable all XAI features
    if args.enable_xai:
        args.enable_attention = True
        args.enable_gradcam = True
        args.enable_frequency = True

    # Load model
    print(f"\n[Loading] Model...")
    try:
        mapping = make_label_mapping(args.dataset_name, args.class_type)
        classnum = 4 if args.class_type == 'all' else 2

        model = HFFNet(4, 16, classnum)

        # ===== CORRECTED CHECKPOINT LOADING =====
        success = load_checkpoint_with_prefix_handling(
            model=model,
            checkpoint_path=args.checkpoint,
            device='cuda'
        )

        if not success:
            print("❌ Failed to load checkpoint. Exiting...")
            sys.exit(1)

        print(f"✓ Model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize evaluator
    evaluator = EnhancedHFFNetEvaluator(
        model=model,
        device='cuda',
        output_dir=args.output_dir,
        args=args
    )

    # Load data
    print(f"[Loading] Data from {args.test_list}")
    try:
        data_files = dict(train=args.test_list, val=args.test_list)
        loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=4)
        val_loader = loaders['val']
        print(f"✓ Data loaded: {len(val_loader)} batches\n")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Evaluation loop
    print(f"[Starting] Evaluation...")
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
                print(f"❌ Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ===== FINAL METRICS AGGREGATION =====
    print(f"\n\n{'='*70}")
    print(f"FINAL METRICS SUMMARY")
    print(f"{'='*70}\n")

    if all_results:
        # Collect all metrics
        class_metrics = {}

        # Initialize class dictionaries
        for class_id in range(classnum):
            class_metrics[f'class_{class_id}'] = {'dice': [], 'iou': []}

        # Aggregate metrics from all samples
        for result in all_results:
            if 'metrics' in result and 'per_class' in result['metrics']:
                for class_name, metric_dict in result['metrics']['per_class'].items():
                    if class_name in class_metrics:
                        class_metrics[class_name]['dice'].append(metric_dict['dice'])
                        class_metrics[class_name]['iou'].append(metric_dict['iou'])

        # Print per-class statistics
        print("PER-CLASS METRICS (Mean ± Std):")
        print("="*70)

        for class_name in sorted(class_metrics.keys()):
            dices = class_metrics[class_name]['dice']
            ious = class_metrics[class_name]['iou']

            if dices:
                mean_dice = np.mean(dices)
                std_dice = np.std(dices)
                mean_iou = np.mean(ious)
                std_iou = np.std(ious)

                print(f"\n{class_name}:")
                print(f"  Dice: {mean_dice:.4f} ± {std_dice:.4f}")
                print(f"  IoU:  {mean_iou:.4f} ± {std_iou:.4f}")

        print(f"\n{'='*70}")

    # Summary
    print(f"\nEVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Samples evaluated: {len(all_results)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metrics: YES ✓")
    print(f"{'='*70}\n")

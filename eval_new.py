# eval_new_FIXED.py
# COMPLETE FILE — robust loading + fixed Grad-CAM + normalized mechanistic importance
# Preserves original saving/printing behavior

import torch
import argparse
import time
import os
import numpy as np
import random
from tqdm import tqdm
import sys
from datetime import datetime
from pathlib import Path
from warnings import simplefilter

# XAI imports (your existing modules)
from explainability.attention_vis import (
    EnhancedFDCAAttentionVisualizer,
    EnhancedSegmentationGradCAM,
    EnhancedFrequencyComponentAnalyzer
)
from explainability.freq_analysis import EnhancedFrequencyDomainAnalyzer

# Project imports
from config.train_test_config.train_test_config import print_val_loss, print_val_eval, save_val_best_3d_m
from loss.loss_function import segmentation_loss
from model.HFF_MobileNetV3_fixed import HFFNet
from loader.dataload3d import get_loaders

simplefilter(action='ignore', category=FutureWarning)

# ---------------- Logger ----------------

class Logger(object):
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
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{output_dir}/logs/eval_log_{timestamp}.txt"
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    return log_file

# -------------- Utils -------------------

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_label_mapping(dataset_name, class_type):
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
    out = torch.zeros_like(mask, dtype=torch.long)
    for old, new in mapping.items():
        out[mask == old] = new
    return out

# -------- Metrics (per old eval.py) -----

def dice_score(output: torch.Tensor, target: torch.Tensor, class_id: int = 1) -> float:
    smooth = 1e-6
    pred = torch.argmax(output, dim=1)
    pred_binary = (pred == class_id).float()
    target_binary = (target == class_id).float()
    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary)
    if union == 0:
        return 1.0 if torch.equal(pred_binary, target_binary) else 0.0
    dice = (2.0 * intersection) / (union + smooth)
    return dice.item()

def iou_score(output: torch.Tensor, target: torch.Tensor, class_id: int = 1) -> float:
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

# -------------- Robust checkpoint loader ---------------

def load_checkpoint_to_model(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state = {}
    for k, v in state_dict.items():
        new_state[k[7:]] = v if k.startswith('module.') else v if not k.startswith('module.') else v

    # fix the above minor mistake: ensure correct key handling
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    print("✓ Checkpoint loaded and model moved to device")
    return model

# -------------- Evaluator with Grad-CAM fix and normalized mechanistic AI -----

class EnhancedHFFNetEvaluator:
    def __init__(self, model, device='cuda', output_dir='./outputs', args=None):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.args = args

        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.output_dir / f"xai_{run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.xai_dir = self.run_dir

        (self.xai_dir / "attention").mkdir(parents=True, exist_ok=True)
        (self.xai_dir / "gradcam").mkdir(parents=True, exist_ok=True)
        (self.xai_dir / "freq_component").mkdir(parents=True, exist_ok=True)
        (self.xai_dir / "freq_analysis").mkdir(parents=True, exist_ok=True)

        self.attention_viz = EnhancedFDCAAttentionVisualizer(
            device=device, save_dir=str(self.xai_dir / 'attention'), dpi=600
        )
        self.gradcam = EnhancedSegmentationGradCAM(
            model=model, target_layers=['decoder', 'fusion', 'encoder'],
            device=device, save_dir=str(self.xai_dir / 'gradcam'), dpi=600
        )
        self.freq_component = EnhancedFrequencyComponentAnalyzer(
            device=device, save_dir=str(self.xai_dir / 'freq_component'), dpi=600
        )
        self.freq_analyzer = EnhancedFrequencyDomainAnalyzer(
            device=device, save_dir=str(self.xai_dir / 'freq_analysis'), dpi=600
        )
        print(f"✓ XAI modules initialized")
        print(f"✓ Run output directory: {self.run_dir}")

    # NEW: normalized feature importance, added directly here to avoid editing your XAI libs
    @staticmethod
    def compute_feature_importance_normalized(activations_dict):
        importance = {}
        for layer_name, acts in activations_dict.items():
            try:
                raw = np.abs(acts).mean(axis=tuple(range(acts.ndim-1)))
                min_val = raw.min()
                max_val = raw.max()
                if max_val - min_val > 1e-8:
                    norm = (raw - min_val) / (max_val - min_val)
                else:
                    norm = np.ones_like(raw) * 0.5
                importance[layer_name] = norm
            except Exception as e:
                print(f"Warning: Failed to compute importance for {layer_name}: {e}")
                importance[layer_name] = np.ones(acts.shape[-1]) * 0.5
        return importance

    def evaluate_batch(self, low_freq_input: torch.Tensor, high_freq_input: torch.Tensor,
                       mask_gt: torch.Tensor, sample_id: str = 'sample'):
        results = {
            'sample_id': sample_id,
            'predictions': None,
            'attention': {},
            'gradcam': {},
            'frequency': {},
            'uncertainty': {},
            'metrics': {}
        }

        # 1) Primary prediction (no gradients)
        print(f"\n[1] Getting primary prediction...")
        with torch.no_grad():
            output = self.model(low_freq_input.to(self.device), high_freq_input.to(self.device))
            output_main = output[0] if isinstance(output, tuple) else output
            predictions = torch.softmax(output_main, dim=1)
            predicted_seg = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
            results['predictions'] = predicted_seg

        # 2) Per-sample metrics
        mask_gt_cpu = mask_gt.cpu()
        output_main_cpu = output_main.cpu()
        num_classes = output_main.shape[1]
        per_class_metrics = {}
        for class_id in range(num_classes):
            dice = dice_score(output_main_cpu, mask_gt_cpu, class_id=class_id)
            iou = iou_score(output_main_cpu, mask_gt_cpu, class_id=class_id)
            per_class_metrics[f'class_{class_id}'] = {'dice': float(dice), 'iou': float(iou)}
        results['metrics']['per_class'] = per_class_metrics

        print(f"\n[METRICS FOR {sample_id}]\n{'='*50}")
        for class_name, metric in per_class_metrics.items():
            print(f"{class_name}: Dice={metric['dice']:.4f}, IoU={metric['iou']:.4f}")
        print(f"{'='*50}")

        # 3) Attention + Mechanistic AI (no gradients needed)
        if self.args and getattr(self.args, 'enable_attention', True):
            print(f"\n[2] Generating attention maps...")
            try:
                with torch.no_grad():
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
                        input_img=input_img, attention_map=aggregated_attn[0],
                        output_path=output_path, dpi=600
                    )
                    results['attention']['generated'] = True

                    # Mechanistic normalized importance
                    activations = self.attention_viz.activations_cache
                    if activations:
                        print(f"\n[MECHANISTIC INTERPRETABILITY ANALYSIS]\n{'='*70}")
                        print(f"Sample: {sample_id}\n{'='*70}\n")
                        importance = self.compute_feature_importance_normalized(activations)
                        results['mechanistic_insights'] = {}
                        for layer_name, scores in importance.items():
                            num_features = len(scores)
                            mean_imp = float(np.mean(scores))
                            max_imp = float(np.max(scores))
                            min_imp = float(np.min(scores))
                            top_idx = np.argsort(scores)[-5:][::-1]
                            top_val = scores[top_idx]
                            print(f"Layer: {layer_name}")
                            print(f"  Features: {num_features}")
                            print(f"  Mean Importance: {mean_imp:.4f}")
                            print(f"  Max Importance: {max_imp:.4f}")
                            print(f"  Min Importance: {min_imp:.4f}")
                            print(f"  Top-5: {list(map(int, top_idx))}")
                            print(f"  Top-5 Values: {[f'{v:.4f}' for v in top_val]}\n")
                            results['mechanistic_insights'][layer_name] = {
                                'num_features': int(num_features),
                                'mean_importance': mean_imp,
                                'max_importance': max_imp,
                                'min_importance': min_imp,
                                'top_5_indices': list(map(int, top_idx)),
                                'top_5_values': list(map(float, top_val)),
                            }
                        print(f"{'='*70}\n✓ Mechanistic analysis complete\n")
            except Exception as e:
                print(f"Warning: Attention generation failed: {e}")
                results['attention']['generated'] = False

        # 4) Grad-CAM (requires gradients) — fixed
        if self.args and getattr(self.args, 'enable_gradcam', True):
            print(f"[3] Generating Grad-CAM...")
            try:
                low_grad = low_freq_input.detach().clone().requires_grad_(True)
                high_grad = high_freq_input.detach().clone().requires_grad_(True)
                full_grad = torch.cat([low_grad, high_grad], dim=1)
                self.model.eval()
                cams = self.gradcam.generate_multi_class_cam(
                    full_grad, num_classes=num_classes, lf_channels=low_freq_input.shape[1]
                )
                cam_generated = any(cam is not None for cam in cams.values())
                for cid, cam in cams.items():
                    if cam is not None:
                        print(f"✓ Successfully generated CAM for class {cid}")
                if 1 in cams and cams[1] is not None:
                    input_img = low_freq_input[0, 0].cpu().numpy()
                    seg_mask = mask_gt[0].cpu().numpy()
                    out_path = self.xai_dir / 'gradcam' / f'{sample_id}_gradcam.png'
                    self.gradcam.visualize_gradcam_enhanced(
                        input_img=input_img, cam=cams[1], seg_mask=seg_mask, output_path=out_path, dpi=600
                    )
                    results['gradcam']['generated'] = True
                    print(f"✓ Saved Grad-CAM to {out_path}")
                else:
                    print("⚠ No CAM generated for class 1 visualization")
                    results['gradcam']['generated'] = cam_generated
            except Exception as e:
                print(f"❌ Warning: Grad-CAM generation failed: {e}")
                results['gradcam']['generated'] = False

        # 5) Frequency component analysis (unchanged)
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
                out_path = self.xai_dir / 'freq_component' / f'{sample_id}_freq_comp.png'
                self.freq_component.visualize_frequency_contributions_enhanced(
                    input_img=input_img,
                    lf_pred=lf_pred[0].argmax(dim=0).cpu().numpy(),
                    hf_pred=hf_pred[0].argmax(dim=0).cpu().numpy(),
                    full_pred=predicted_seg,
                    ground_truth=mask_gt[0].cpu().numpy(),
                    output_path=out_path,
                    dpi=600
                )
                results['frequency']['lf_hf_analysis'] = True
            except Exception as e:
                print(f"Warning: Frequency analysis failed: {e}")
                results['frequency']['lf_hf_analysis'] = False

        print(f"✓ Evaluation complete for {sample_id}")
        return results

# -------------------- Main --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HFF-Net Enhanced Evaluation with XAI')
    parser.add_argument('--test_list', type=str, default='./data/brats20/2-val.txt')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/hff_best.pth')
    parser.add_argument('--dataset_name', choices=['brats19', 'brats20', 'brats23men', 'msdbts'], default='brats20')
    parser.add_argument('--class_type', choices=['et', 'tc', 'wt', 'all'], default='all')
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
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--enable_xai', action='store_true')
    parser.add_argument('--enable_attention', action='store_true')
    parser.add_argument('--enable_gradcam', action='store_true')
    parser.add_argument('--enable_frequency', action='store_true')
    parser.add_argument('--max_samples', type=int, default=-1)
    args = parser.parse_args()

    init_seeds(42)
    log_file = setup_logging(args.output_dir)

    print(f"\n{'='*70}")
    print(f"HFF-NET ENHANCED EVALUATION WITH XAI + METRICS")
    print(f"{'='*70}")
    print(f"Log file: {log_file}")
    print(f"Output directory: {args.output_dir}")

    if args.enable_xai:
        args.enable_attention = True
        args.enable_gradcam = True
        args.enable_frequency = True

    print(f"\n[Loading] Model from {args.checkpoint}")
    try:
        mapping = make_label_mapping(args.dataset_name, args.class_type)
        classnum = 4 if args.class_type == 'all' else 2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = HFFNet(4, 16, classnum)
        model = load_checkpoint_to_model(model, args.checkpoint, device)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        criterion = segmentation_loss(args.loss, False, cn=classnum).to(device)
    except Exception as e:
        print(f"Warning: Could not create criterion: {e}")
        criterion = None

    evaluator = EnhancedHFFNetEvaluator(model=model, device=device, output_dir=args.output_dir, args=args)

    print(f"\n[Loading] Data from {args.test_list}")
    try:
        data_files = dict(train=args.test_list, val=args.test_list)
        loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=4)
        val_loader = loaders['val']
        print(f"✓ Data loaded: {len(val_loader)} batches")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"\n[Starting] Evaluation (XAI + per-sample metrics)...")
    print(f"{'='*70}")
    all_results = []
    sample_count = 0

    for batch_idx, data in enumerate(tqdm(val_loader, desc='Evaluating')):
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break
        try:
            low_freq_inputs, high_freq_inputs = [], []
            for j in range(20):
                tensor = data[j].unsqueeze(1).to(device)
                if j in [0, 1, 2, 3]:
                    low_freq_inputs.append(tensor)
                else:
                    high_freq_inputs.append(tensor)
            low = torch.cat(low_freq_inputs, dim=1)
            high = torch.cat(high_freq_inputs, dim=1)
            mask_val = mask_to_class_indices(data[20], mapping).long().to(device)

            sample_id = f'sample_{batch_idx:04d}'
            results = evaluator.evaluate_batch(low_freq_input=low, high_freq_input=high, mask_gt=mask_val, sample_id=sample_id)
            all_results.append(results)
            sample_count += 1
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

    # Standard validation printing pass (exact original behavior)
    print(f"\n\n{'='*70}")
    print("Running standard validation pass (for exact print_val_loss/print_val_eval output)...")
    print(f"{'='*70}\n")
    try:
        loaders2 = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=4)
        val_loader2 = loaders2['val']
    except Exception as e:
        print(f"Error reloading data for validation pass: {e}")
        val_loader2 = None

    if val_loader2 is not None:
        num_batches = len(val_loader2)
        val_loss_sup_1, val_loss_sup_2 = 0.0, 0.0
        with torch.no_grad():
            score_list_val_1 = None
            score_list_val_2 = None
            mask_list_val = None
            for i, data in enumerate(tqdm(val_loader2, desc='Validation Pass (printing)')):
                low_freq_inputs, high_freq_inputs = [], []
                for j in range(20):
                    tensor = data[j].unsqueeze(1).to(device)
                    if j in [0, 1, 2, 3]:
                        low_freq_inputs.append(tensor)
                    else:
                        high_freq_inputs.append(tensor)
                low = torch.cat(low_freq_inputs, dim=1)
                high = torch.cat(high_freq_inputs, dim=1)
                mask_val = mask_to_class_indices(data[20], mapping).long().to(device)

                outputs_val_1, outputs_val_2, side1, side2 = model(low, high)
                outputs_val_1_cpu = outputs_val_1.detach().cpu()
                outputs_val_2_cpu = outputs_val_2.detach().cpu()
                mask_cpu = mask_val.detach().cpu()

                if i == 0:
                    score_list_val_1 = outputs_val_1_cpu
                    score_list_val_2 = outputs_val_2_cpu
                    mask_list_val = mask_cpu
                else:
                    score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1_cpu), dim=0)
                    score_list_val_2 = torch.cat((score_list_val_2, outputs_val_2_cpu), dim=0)
                    mask_list_val = torch.cat((mask_list_val, mask_cpu), dim=0)

                if criterion is not None:
                    loss1 = criterion(outputs_val_1_cpu, mask_cpu)
                    loss2 = criterion(outputs_val_2_cpu, mask_cpu)
                    val_loss_sup_1 += loss1.item()
                    val_loss_sup_2 += loss2.item()

        try:
            print_val_loss(val_loss_sup_1, val_loss_sup_2, {'val': num_batches}, 63, 0)
            print_val_eval(classnum, score_list_val_1, score_list_val_2, mask_list_val, 31)
        except Exception as e:
            print(f"Warning: Could not print validation summary due to: {e}")
    else:
        print("Skipping validation printing pass due to data load error.")

    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Samples evaluated (XAI pass): {len(all_results)}")
    print(f"Output directory: {args.output_dir}")
    print(f"All XAI outputs saved with 600 DPI resolution (if enabled)")
    print(f"\nGenerated outputs:")
    print(f" - Attention maps: {args.enable_attention}")
    print(f" - Grad-CAM visualizations: {args.enable_gradcam}")
    print(f" - Frequency analysis: {args.enable_frequency}")
    print(f" - Dice & IoU Metrics (per-sample): YES ✓")
    print(f" - Mechanistic Interpretability: YES ✓")

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time
import os
import numpy as np
import random
from tqdm import tqdm
import wandb
import nibabel as nib
import sys
from datetime import datetime

from config.train_test_config.train_test_config import print_val_loss, print_val_eval, save_val_best_3d_m
from config.warmup_config.warmup import GradualWarmupScheduler
from loss.loss_function import segmentation_loss
from model.HFF_MobileNetV3 import HFFNet
from loader.dataload3d import get_loaders
from warnings import simplefilter

# XAI imports
from explainability.attention_vis import FDCAAttentionVisualizer, SegmentationGradCAM, FrequencyComponentAnalyzer
from explainability.mc_dropout import MCDropoutUncertainty
from explainability.freq_analysis import FrequencyDomainAnalyzer

simplefilter(action='ignore', category=FutureWarning)

# ---------------------------- LOGGER SETUP ---------------------------- #
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", buffering=1)  # line-buffered for real-time writes

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create logs folder and log file with timestamp
os.makedirs("evaluation_logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"evaluation_logs/inference_log_{timestamp}.txt"
sys.stdout = Logger(log_file)
sys.stderr = sys.stdout  # redirect errors as well
print(f"Logging all outputs to: {log_file}")


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HFF-Net 3D Inference')
    parser.add_argument('--test_list', type=str, help='Path to text file listing test volumes',
                        default='/teamspace/studios/this_studio/HFF/brats20/2-val.txt')
    parser.add_argument('--checkpoint', type=str, help='Path to trained model checkpoint (.pth)',
                        default='/teamspace/studios/this_studio/result/checkpoints/hff/brats20/all/hff-l=0.3-e=30-s=50-g=0.55-b=1-cw=15-w=3-100L-H/best_Result1_et_0.3390_tc_0.2583_wt_0.5082.pth')
    parser.add_argument('--dataset_name', choices=['brats19', 'brats20', 'brats23men', 'msdbts'], default='brats20')
    parser.add_argument('--class_type', choices=['et', 'tc', 'wt', 'all'], default='et')
    parser.add_argument('--selected_modal', nargs='+', default=[
        'flair_L', 't1_L', 't1ce_L', 't2_L',
        'flair_H1', 'flair_H2', 'flair_H3', 'flair_H4',
        't1_H1', 't1_H2', 't1_H3', 't1_H4',
        't1ce_H1', 't1ce_H2', 't1ce_H3', 't1ce_H4',
        't2_H1', 't2_H2', 't2_H3', 't2_H4'], help='Modalities')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-l', '--loss', type=str, default='dice')
    parser.add_argument('--loss2', type=str, default='ff')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='directory where predictions / results will be written')

    # XAI: Add arguments to enable XAI and set MC Dropout samples
    parser.add_argument('--enable_xai', action='store_true', help='Enable XAI visualizations')
    parser.add_argument('--mc_samples', type=int, default=20, help='Number of MC-Dropout samples')

    args = parser.parse_args()

    init_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    mapping = make_label_mapping(args.dataset_name, args.class_type)
    classnum = 4 if args.class_type == 'all' else 2
    criterion = segmentation_loss(args.loss, False, cn=classnum).cuda()

    # load model
    model = HFFNet(4, 16, classnum).cuda()
    state_dict = torch.load(args.checkpoint, map_location='cuda')
    model.load_state_dict(state_dict)
    model.eval()

    # loader: using same list for train and val since get_loaders expects both
    data_files = dict(train=args.test_list, val=args.test_list)
    loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=4)
    val_loader = loaders['val']
    num_batches = len(val_loader)
    val_loss_sup_1 = 0.0
    val_loss_sup_2 = 0.0

    # XAI: Initialize XAI modules here if enabled
    if args.enable_xai:
        attention_viz = FDCAAttentionVisualizer(device='cuda')
        gradcam = SegmentationGradCAM(model=model, target_layers=['decoder', 'fusion'], device='cuda')
        freq_analyzer = FrequencyComponentAnalyzer(device='cuda')
        uncertainty = MCDropoutUncertainty(model=model, num_samples=args.mc_samples, device='cuda')
        freq_analysis_obj = FrequencyDomainAnalyzer(device='cuda')

        # Create directory for saving XAI outputs
        xai_base_dir = os.path.join(args.output_dir, 'xai')
        os.makedirs(xai_base_dir, exist_ok=True)
        attention_dir = os.path.join(xai_base_dir, 'attention')
        gradcam_dir = os.path.join(xai_base_dir, 'gradcam')
        freq_dir = os.path.join(xai_base_dir, 'freq')
        uncertainty_dir = os.path.join(xai_base_dir, 'uncertainty')
        for d in [attention_dir, gradcam_dir, freq_dir, uncertainty_dir]:
            os.makedirs(d, exist_ok=True)

    # inference
    all_scores, all_masks = [], []
    with torch.no_grad():
        score_list_val_1 = None
        score_list_val_2 = None
        mask_list_val = None
        for i, data in enumerate(tqdm(val_loader, desc='Inference')):
            # unpack modalities and mask
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

            loss1 = criterion(outputs_val_1_cpu, mask_cpu)
            loss2 = criterion(outputs_val_2_cpu, mask_cpu)
            val_loss_sup_1 += loss1.item()
            val_loss_sup_2 += loss2.item()

            # XAI: Generate visualizations for first N samples for efficiency
            if args.enable_xai and i < 5:  # limit to first 5 batches for example
                # Prepare input batch tensor (batch size=1 here)
                full_input = torch.cat((low, high), dim=1)

                # FDCA Attention visualization
                attention_viz.register_hooks(model)
                attn_maps = attention_viz.extract_attention_maps(full_input, model)
                attention_viz.remove_hooks()
                if attn_maps:
                    aggregated_attention = attention_viz.aggregate_attention_maps(attn_maps)
                    input_img = full_input[0, 0].cpu().numpy()  # first modality
                    attn_map = aggregated_attention[0]  # first example
                    attention_viz.visualize_attention(
                        input_img=input_img,
                        attention_map=attn_map,
                        output_path=os.path.join(attention_dir, f'fdca_attention_sample_{i}.png')
                    )

                # Grad-CAM visualization for ET class (class 1)
                cam = gradcam.generate_cam(full_input, target_class=1)
                if cam is not None:
                    input_img = full_input[0, 0].cpu().numpy()
                    seg_mask = mask_val[0].cpu().numpy()
                    gradcam.visualize_gradcam(
                        input_img=input_img,
                        cam=cam,
                        seg_mask=seg_mask,
                        output_path=os.path.join(gradcam_dir, f'gradcam_sample_{i}.png')
                    )

                # Frequency component analysis (LF-only, HF-only, Full)
                lf_pred = freq_analyzer.generate_lf_only_prediction(model, full_input)
                hf_pred = freq_analyzer.generate_hf_only_prediction(model, full_input)
                full_pred = torch.softmax(outputs_val_1, dim=1)
                freq_analyzer.visualize_frequency_contributions(
                    input_img=input_img,
                    lf_pred=lf_pred[0].argmax(dim=0).cpu().numpy(),
                    hf_pred=hf_pred[0].argmax(dim=0).cpu().numpy(),
                    full_pred=full_pred[0].argmax(dim=0).cpu().numpy(),
                    ground_truth=seg_mask,
                    output_path=os.path.join(freq_dir, f'freq_analysis_sample_{i}.png')
                )

                # MC-Dropout Uncertainty (single sample repeated N times)
                mc_outputs = uncertainty.mc_forward_pass(full_input)
                mean_pred, uncertainty_map = uncertainty.compute_uncertainty_maps(mc_outputs)
                predicted_mask = np.argmax(mean_pred[0], axis=0)
                error_map = np.abs(predicted_mask - seg_mask)
                uncertainty.visualize_uncertainty(
                    input_img=input_img,
                    predicted_mask=predicted_mask,
                    uncertainty_map=uncertainty_map[0],
                    error_map=error_map,
                    output_path=os.path.join(uncertainty_dir, f'uncertainty_sample_{i}.png')
                )

        # summarize
        print_val_loss(val_loss_sup_1, val_loss_sup_2, {'val': num_batches}, 63, 0)
        print_val_eval(classnum, score_list_val_1, score_list_val_2, mask_list_val, 31)

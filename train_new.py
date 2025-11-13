# train_new_MULTIGPU_WITH_MANUAL_CHECKPOINT.py
# MULTI-GPU TRAINING WITH MANUAL CHECKPOINT PATH & EPOCH IN ARGS
# 
# Usage:
# python -m torch.distributed.launch --nproc_per_node=4 train_new.py \
#     --resume_checkpoint /path/to/checkpoint_epoch_61.pth \
#     --resume_epoch 61 \
#     --batch_size 1 \
#     --num_epochs 200

"""
Multi-GPU Training for HFF-Net with Manual Checkpoint Resume
=============================================================
- Users specify checkpoint path directly via --resume_checkpoint
- Users specify resume epoch via --resume_epoch
- No auto-detection of checkpoints
- Simple and straightforward
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time
import os
import numpy as np
import random
from tqdm import tqdm
import wandb
from datetime import datetime
from config.train_test_config.train_test_config import print_train_loss, print_val_loss, \
print_val_eval, print_best, save_val_best_3d_m
from config.warmup_config.warmup import GradualWarmupScheduler
from loss.loss_function import segmentation_loss
from model.HFF_MobileNetV3_fixed import HFFNet
from loader.dataload3d import get_loaders
from warnings import simplefilter
from explainability.mc_dropout_fixed import DropoutScheduler, MCDropoutUncertainty
import gc

simplefilter(action='ignore', category=FutureWarning)

import sys

# ============================================================================
# SECTION 1: LOGGER & DISTRIBUTED SETUP
# ============================================================================

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

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

def setup_distributed_logging(rank: int, log_file: str):
    """Setup logging for distributed training"""
    if rank == 0:
        sys.stdout = Logger(log_file)
        sys.stderr = sys.stdout
    else:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

def init_distributed_mode(rank=None, world_size=None):
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1, 'cuda:0'

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    device = f'cuda:{gpu}'

    if rank == 0:
        print(f'✓ Distributed mode initialized')
        print(f'✓ Rank: {rank}/{world_size}')
        print(f'✓ GPU: {gpu}')

    return rank, world_size, device

# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

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
    """Returns a dict old_label->new_label"""
    if dataset_name in ('brats19','brats20','msdbts'):
        raw = [0,1,2,4]
    else:
        raw = [0,1,2,3]

    if class_type == 'et':
        pos = raw[-1]
        mapping = {l:(1 if l==pos else 0) for l in raw}
    elif class_type == 'tc':
        p1, p2 = 1, raw[-1]
        mapping = {l:(1 if l in (p1,p2) else 0) for l in raw}
    elif class_type == 'wt':
        p2, p3 = 2, raw[-1]
        mapping = {l:(1 if l in (1,p2,p3) else 0) for l in raw}
    else:
        mapping = { old:new for new, old in enumerate(raw) }

    return mapping

def mask_to_class_indices(mask, mapping):
    """Recode mask according to mapping"""
    out = torch.zeros_like(mask, dtype=torch.long)
    for old, new in mapping.items():
        out[mask == old] = new
    return out

def compute_fisher_information(model, dataloader, criterion, rank, label_mapping):
    model.eval()
    fisher_information = {}
    for name, param in model.named_parameters():
        if name =='input_ed.conv.weight':
            fisher_information[name] = torch.zeros_like(param)

    for i, data in enumerate(dataloader):
        low_freq_inputs = []
        high_freq_inputs = []
        for j in range(20):
            input_tensor = data[j].unsqueeze(dim=1).type(torch.cuda.FloatTensor)
            if j in [0, 1, 2, 3]:
                low_freq_inputs.append(input_tensor)
            else:
                high_freq_inputs.append(input_tensor)

        low_freq_inputs = torch.cat(low_freq_inputs, dim=1)
        high_freq_inputs = torch.cat(high_freq_inputs, dim=1)
        target = mask_to_class_indices(data[20], label_mapping).long().cuda()

        outputs_train_1, outputs_train_2, side1, side2 = model(low_freq_inputs, high_freq_inputs)
        loss_train_sup1 = criterion(outputs_train_1, target)
        loss_train_sup2 = criterion(outputs_train_2, target)
        total_loss = loss_train_sup1 + loss_train_sup2

        model.zero_grad()
        total_loss.backward()

        for name, param in model.named_parameters():
            if name in fisher_information:
                fisher_information[name] += param.grad ** 2

    for name in fisher_information:
        fisher_information[name] /= len(dataloader)

    return fisher_information

def important_weights_with_fisher(model, fisher_info, std_multiplier=1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == 'input_ed.conv.weight':
                importance = fisher_info[name]
                mean_importance = torch.mean(importance, dim=[2, 3, 4], keepdim=True)
                std_importance = torch.std(importance, dim=[2, 3, 4], keepdim=True)
                threshold = mean_importance + std_multiplier * std_importance
                important_weights = importance > threshold
                return important_weights

# ============================================================================
# SECTION 3: MANUAL CHECKPOINT LOADING (FROM ARGS)
# ============================================================================

def load_checkpoint_from_args(model, optimizer, scheduler, checkpoint_path: str, 
                              resume_epoch: int, rank: int, device):
    """
    Load checkpoint from manually specified path
    Args:
        checkpoint_path: Full path to checkpoint file (e.g., '/path/to/checkpoint_epoch_61.pth')
        resume_epoch: Epoch number to resume from (provided via args)
        rank: Rank of current process
        device: GPU device
    """
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        return 0

    if rank == 0:
        print(f"\n[RESUME] Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if exists
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if rank == 0:
            print(f"[RESUME] ✓ Checkpoint loaded successfully")
            print(f"[RESUME] ✓ Resuming from epoch {resume_epoch + 1}")
            print(f"[RESUME] ✓ Optimizer state restored")
            print(f"[RESUME] ✓ Learning rate scheduler state restored\n")

        return resume_epoch

    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print(f"[ERROR] Starting training from scratch\n")
        return 0

def save_checkpoint(model, optimizer, scheduler, epoch: int, path_trained_models: str, rank: int):
    """Save checkpoint for resuming training"""
    if rank != 0:
        return

    checkpoint_path = os.path.join(path_trained_models, f'checkpoint_epoch_{epoch}.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)

# ============================================================================
# SECTION 4: MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ===== CHECKPOINT RESUME ARGS =====
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume from (e.g., /path/to/checkpoint_epoch_61.pth)')
    parser.add_argument('--resume_epoch', type=int, default=0,
                       help='Epoch number to resume from (e.g., 61)')
    # ===== END RESUME ARGS =====

    # Data arguments
    parser.add_argument('--train_list', type=str, default='./data/brats20/2-train.txt')
    parser.add_argument('--val_list', type=str, default='./data/brats20/2-val.txt')
    parser.add_argument('--path_trained_models', default='./result/checkpoints/hff')
    parser.add_argument('--dataset_name', choices=['brats19','brats20','brats23men','msdbts'], default='brats20')
    parser.add_argument('--class_type', choices=['et','tc','wt','all'], default='all')
    parser.add_argument('--selected_modal', nargs='+',
        default=['flair_L', 't1_L', 't1ce_L', 't2_L', 'flair_H1', 'flair_H2', 'flair_H3', 'flair_H4',
                 't1_H1', 't1_H2', 't1_H3', 't1_H4', 't1ce_H1', 't1ce_H2', 't1ce_H3', 't1ce_H4',
                 't2_H1', 't2_H2', 't2_H3', 't2_H4'])
    parser.add_argument('--input1', default='L')
    parser.add_argument('--input2', default='H')
    parser.add_argument('--sup_mark', default='100')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.3, type=float)
    parser.add_argument('-g', '--gamma', default=0.55, type=float)
    parser.add_argument('-u', '--unsup_weight', default=15, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('--loss2', default='ff', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=3)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float)
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int)
    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('-n', '--network', default='hff', type=str)

    args = parser.parse_args()

    # ===== MULTI-GPU SETUP =====
    rank, world_size, device = init_distributed_mode()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"training_logs/training_log_{timestamp}.txt"
    os.makedirs("training_logs", exist_ok=True)
    setup_distributed_logging(rank, log_file)
    
    if rank == 0:
        wandb.init(project=f'HFF_MultiGPU_{args.dataset_name}_{args.class_type}')
        wandb.config.update(args)

    init_seeds(42)

    if args.class_type == 'all':
        classnum = 4
    else:
        classnum = 2

    label_mapping = make_label_mapping(args.dataset_name, args.class_type)

    print_num = 63
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # Model save path
    path_trained_models = (
        args.path_trained_models + '/' +
        str(os.path.split(args.dataset_name)[1] + '/' + str(args.class_type))
    )

    if not os.path.exists(path_trained_models):
        os.makedirs(path_trained_models)

    path_trained_models = (
        path_trained_models + '/' +
        str(args.network) + '-l=' + str(args.lr) + '-e=' + str(args.num_epochs) +
        '-s=' + str(args.step_size) + '-g=' + str(args.gamma) + '-b=' + str(args.batch_size) +
        '-cw=' + str(args.unsup_weight) + '-w=' + str(args.warm_up_duration) + '-' +
        str(args.sup_mark) + str(args.input1) + '-' + str(args.input2) + '-' + timestamp
    )

    if not os.path.exists(path_trained_models):
        os.makedirs(path_trained_models)

    if rank == 0:
        print(f"\n{'='*63}")
        print(f"HFF-NET MULTI-GPU TRAINING")
        print(f"{'='*63}")
        print(f"Rank: {rank}/{world_size}")
        print(f"Device: {device}")
        print(f"GPUs: {torch.cuda.device_count()}")
        print(f"Log file: {log_file}")
        print(f"Models saved to: {path_trained_models}")
        
        # Print resume info
        if args.resume_checkpoint:
            print(f"\n[RESUME INFO]")
            print(f"Checkpoint: {args.resume_checkpoint}")
            print(f"Resume epoch: {args.resume_epoch}")

    # ===== DATA LOADING =====
    data_files = dict(train=args.train_list, val=args.val_list)
    loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=8)
    loaders = {x: loaders[x] for x in ('train', 'val')}
    num_batches = {'train_sup': len(loaders['train']), 'val': len(loaders['val'])}

    if rank == 0:
        print(f"✓ Data loaded: {num_batches['train_sup']} train batches, {num_batches['val']} val batches")

    # ===== MODEL SETUP =====
    model = HFFNet(4, 16, classnum)
    model = model.to(device)

    # ===== MULTI-GPU: WRAP WITH DDP =====
    if world_size > 1:
        model = DDP(model, device_ids=[int(device.split(':')[1])], find_unused_parameters=True)
        if rank == 0:
            print(f"✓ Model wrapped with DistributedDataParallel")
    else:
        if rank == 0:
            print(f"✓ Model on single GPU")

    # Dropout scheduler
    dropout_scheduler = DropoutScheduler(model, base_dropout=0.5)
    dropout_scheduler.set_dropout_rate(0.5)

    # Training setup
    criterion = segmentation_loss(args.loss, False, cn=classnum).to(device)
    FFcriterion = segmentation_loss(args.loss2, True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=5.3 * 10 ** args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0,
                                             total_epoch=args.warm_up_duration,
                                             after_scheduler=exp_lr_scheduler)

    # ===== LOAD CHECKPOINT FROM ARGS =====
    start_epoch = 0
    if args.resume_checkpoint is not None:
        start_epoch = load_checkpoint_from_args(model, optimizer, scheduler_warmup, 
                                               args.resume_checkpoint, args.resume_epoch, 
                                               rank, device)
    elif rank == 0:
        print(f"[TRAINING] Starting from epoch 0 (no checkpoint specified)\n")

    # ===== TRAINING LOOP =====
    since = time.time()
    count_iter = 0
    important_weights = None
    best_model = model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(1)]

    for epoch in range(start_epoch, args.num_epochs):
        # Linearly decay dropout
        current_dropout = 0.5 - 0.4 * (epoch / args.num_epochs)
        dropout_scheduler.set_dropout_rate(max(current_dropout, 0.1))

        if rank == 0 and epoch % 5 == 0:
            print(f"Epoch {epoch+1} - Dropout rate: {max(current_dropout, 0.1):.4f}")

        count_iter += 1

        # ===== TRAINING =====
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        model.train()
        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_unsup = 0.0
        train_loss_reg = 0.0
        train_loss_side1 = 0.0
        train_loss_side2 = 0.0
        train_loss = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
        reg_weight = 0.000005

        if epoch == args.warm_up_duration:
            fisher_info = compute_fisher_information(model, loaders['train'], criterion, rank, label_mapping)
            important_weights = important_weights_with_fisher(model, fisher_info, std_multiplier=1)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name == 'input_ed.conv.weight':
                        param[important_weights] = model.laplacian_target[important_weights]

        for i, data in enumerate(tqdm(loaders['train'], disable=(rank != 0))):
            low_freq_inputs = []
            high_freq_inputs = []

            for j in range(20):
                input_tensor = data[j].unsqueeze(dim=1).type(torch.cuda.FloatTensor)
                if j in [0, 1, 2, 3]:
                    low_freq_inputs.append(input_tensor)
                else:
                    high_freq_inputs.append(input_tensor)

            low_freq_inputs = torch.cat(low_freq_inputs, dim=1)
            high_freq_inputs = torch.cat(high_freq_inputs, dim=1)
            mask_train = mask_to_class_indices(data[20], label_mapping).long().to(device)

            optimizer.zero_grad()

            outputs_train_1, outputs_train_2, side1, side2 = model(low_freq_inputs, high_freq_inputs)

            loss_train_sup1 = criterion(outputs_train_1, mask_train)
            loss_train_sup2 = criterion(outputs_train_2, mask_train)
            loss_train_side1 = criterion(side1, mask_train)
            loss_train_side2 = criterion(side2, mask_train)
            reg_loss = torch.sum((model.input_ed.conv.weight - model.laplacian_target) ** 2) * reg_weight
            loss_train_unsup = FFcriterion(outputs_train_1, outputs_train_2)
            loss_train_unsup = loss_train_unsup * unsup_weight

            loss_train_total = loss_train_sup1 + loss_train_sup2 + loss_train_unsup + loss_train_side1 + loss_train_side2 + reg_loss

            loss_train_total.backward()

            if important_weights is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name == "input_ed.conv.weight":
                            param.grad[important_weights] = 0

            optimizer.step()

            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_reg += reg_loss.item()
            train_loss_side1 += loss_train_side1.item()
            train_loss_side2 += loss_train_side2.item()
            train_loss_unsup += loss_train_unsup.item()
            train_loss += loss_train_total.item()

            # Clear cache every 5 batches
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        scheduler_warmup.step()

        # ===== VALIDATION =====
        torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()

            val_loss_sup_1 = 0.0
            val_loss_sup_2 = 0.0

            for i, data in enumerate(tqdm(loaders['val'], disable=(rank != 0))):
                low_freq_inputs = []
                high_freq_inputs = []

                for j in range(20):
                    input_tensor = data[j].unsqueeze(dim=1).type(torch.cuda.FloatTensor)
                    if j in [0, 1, 2, 3]:
                        low_freq_inputs.append(input_tensor)
                    else:
                        high_freq_inputs.append(input_tensor)

                low_freq_inputs = torch.cat(low_freq_inputs, dim=1)
                high_freq_inputs = torch.cat(high_freq_inputs, dim=1)
                mask_val = mask_to_class_indices(data[20], label_mapping).long().to(device)

                outputs_val_1, outputs_val_2, side1, side2 = model(low_freq_inputs, high_freq_inputs)

                outputs_val_1 = outputs_val_1.detach().cpu()
                outputs_val_2 = outputs_val_2.detach().cpu()
                mask_val = mask_val.detach().cpu()

                if i == 0:
                    score_list_val_1 = outputs_val_1
                    score_list_val_2 = outputs_val_2
                    mask_list_val = mask_val
                else:
                    score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1), dim=0)
                    score_list_val_2 = torch.cat((score_list_val_2, outputs_val_2), dim=0)
                    mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)

                loss_val_sup_1 = criterion(outputs_val_1, mask_val)
                loss_val_sup_2 = criterion(outputs_val_2, mask_val)

                val_loss_sup_1 += loss_val_sup_1.item()
                val_loss_sup_2 += loss_val_sup_2.item()

        # ===== SAVE CHECKPOINT =====
        save_checkpoint(model, optimizer, scheduler_warmup, epoch, path_trained_models, rank)

        # ===== LOG & PRINT =====
        if rank == 0 and (count_iter % args.display_iter == 0):
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')

            train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_cps, reg1, train_epoch_loss = \
                print_train_loss(train_loss_sup_1, train_loss_sup_2, train_loss_unsup, train_loss_reg,
                               train_loss, num_batches, print_num, print_num_half)

            val_epoch_loss_sup_1, val_epoch_loss_sup_2 = \
                print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half)

            val_eval_list_1, val_eval_list_2, val_m_dc_1, val_m_dc_2 = \
                print_val_eval(classnum, score_list_val_1, score_list_val_2, mask_list_val,
                             print_num_half)

            if rank == 0:
                wandb.log({
                    "train_loss_sup1": train_epoch_loss_sup_1,
                    "train_loss_sup2": train_epoch_loss_sup_2,
                    "train_loss": train_epoch_loss,
                    "train_loss_unsup": train_epoch_loss_cps,
                    "epoch": epoch
                })

            print('-' * print_num)
            print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(
                print_num_minus, ' '), '|')

        torch.cuda.empty_cache()

    # ===== TRAINING COMPLETE =====
    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    if rank == 0:
        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(
            print_num_minus, ' '), '|')
        print('=' * print_num)
        wandb.finish()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

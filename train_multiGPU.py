# train_new_DDP_READY.py
# FINAL PRODUCTION VERSION - DDP-READY WITH YOUR EXACT LOGGING
#
# Features:
# - DistributedDataParallel for multi-GPU training
# - Your exact logging format (print_train_loss, print_val_loss, save_val_best_3d_m)
# - Checkpoint resume via --resume_checkpoint and --resume_epoch
# - torchrun/torch.distributed.launch compatible
# - AMP optional (--use_amp)
# - Gradient accumulation optional (--grad_accum_steps)
# - Fisher-based weight masking
# - MC-Dropout uncertainty
# - Rank-0 only logging

"""
RUN COMMANDS:

1. Start fresh (all 4 GPUs):
   python -m torch.distributed.launch --nproc_per_node=4 train_new.py \\
       --train_list ./data/brats20/final/0-train.txt \\
       --val_list ./data/brats20/final/0-val.txt \\
       --batch_size 1 \\
       --num_epochs 200

2. Resume from checkpoint:
   python -m torch.distributed.launch --nproc_per_node=4 train_new.py \\
       --resume_checkpoint /path/to/checkpoint_epoch_61.pth \\
       --resume_epoch 61 \\
       --train_list ./data/brats20/final/0-train.txt \\
       --val_list ./data/brats20/final/0-val.txt \\
       --batch_size 1 \\
       --num_epochs 200

3. Using torchrun (newer PyTorch):
   torchrun --nproc_per_node=4 train_new.py [args]

4. Single GPU (fallback):
   python train_new.py [args]
"""

import os
import sys
import time
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

from config.train_test_config.train_test_config import print_train_loss, print_val_loss, print_val_eval, print_best, save_val_best_3d_m
from config.warmup_config.warmup import GradualWarmupScheduler
from loss.loss_function import segmentation_loss
from model.HFF_MobileNetV3_fixed import HFFNet
from loader.dataload3d import get_loaders
from warnings import simplefilter
from explainability.mc_dropout_fixed import DropoutScheduler, MCDropoutUncertainty

simplefilter(action='ignore', category=FutureWarning)

# ============================================================================
# LOGGER - Your original logging format
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

# ============================================================================
# UTILS - Your original utility functions
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
    out = torch.zeros_like(mask, dtype=torch.long)
    for old, new in mapping.items():
        out[mask == old] = new
    return out

def compute_fisher_information(model, dataloader, criterion, rank):
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
# DDP SETUP
# ============================================================================

def setup_ddp():
    """Returns (is_ddp, rank, world_size, device)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return True, rank, world_size, device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return False, 0, 1, device

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_checkpoint_from_args(model, optimizer, scheduler, checkpoint_path, resume_epoch, rank, device):
    """Load checkpoint from manually specified path"""
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        return 0

    if rank == 0:
        print(f"\n[RESUME] Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if rank == 0:
            print(f"[RESUME] ✓ Checkpoint loaded successfully")
            print(f"[RESUME] ✓ Resuming from epoch {resume_epoch + 1}")
            print(f"[RESUME] ✓ Optimizer and scheduler state restored\n")

        return resume_epoch

    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print(f"[ERROR] Starting training from scratch\n")
        return 0

def save_checkpoint(model, optimizer, scheduler, epoch, path_trained_models, rank):
    """Save checkpoint - only on rank 0"""
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
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # DDP args
    parser.add_argument('--local-rank', type=int, default=-1, dest='local_rank',
                       help='Local rank from torch.distributed.launch')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume from')
    parser.add_argument('--resume_epoch', type=int, default=0,
                       help='Epoch number to resume from')

    # Your original args
    parser.add_argument('--train_list', type=str, default='./data/brats20/2-train.txt')
    parser.add_argument('--val_list', type=str, default='./data/brats20/2-val.txt')
    parser.add_argument('--path_trained_models', default='./result/checkpoints/hff')
    parser.add_argument('--dataset_name', choices=['brats19','brats20','brats23men','msdbts'], default='brats20')
    parser.add_argument('--class_type', choices=['et','tc','wt','all'], default='all')
    parser.add_argument('--selected_modal', nargs='+',
        default=['flair_L','t1_L','t1ce_L','t2_L','flair_H1','flair_H2','flair_H3','flair_H4',
                 't1_H1','t1_H2','t1_H3','t1_H4','t1ce_H1','t1ce_H2','t1ce_H3','t1ce_H4',
                 't2_H1','t2_H2','t2_H3','t2_H4'])
    parser.add_argument('--input1', default='L')
    parser.add_argument('--input2', default='H')
    parser.add_argument('--sup_mark', default='100')
    parser.add_argument('-b','--batch_size', default=1, type=int)
    parser.add_argument('-e','--num_epochs', default=200, type=int)
    parser.add_argument('-s','--step_size', default=50, type=int)
    parser.add_argument('-l','--lr', default=0.3, type=float)
    parser.add_argument('-g','--gamma', default=0.55, type=float)
    parser.add_argument('-u','--unsup_weight', default=15, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('--loss2', default='ff', type=str)
    parser.add_argument('-w','--warm_up_duration', default=3, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float)
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int)
    parser.add_argument('-i','--display_iter', default=1, type=int)
    parser.add_argument('-n','--network', default='hff', type=str)
    parser.add_argument('--grad_accum_steps', default=1, type=int)
    parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()

    # DDP SETUP
    is_ddp, rank, world_size, device = setup_ddp()
    is_main_process = (rank == 0)

    # LOGGING - only on rank 0
    if is_main_process:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"training_logs/training_log_{timestamp}.txt"
        os.makedirs("training_logs", exist_ok=True)
        sys.stdout = Logger(log_file)
        sys.stderr = sys.stdout
        print(f"[DDP] Distributed mode: {is_ddp}, world_size: {world_size}, rank: {rank}")

    # INIT
    init_seeds(42 + rank)

    # WANDB - only on rank 0
    if is_main_process:
        wandb.init(project=f'learning_rate={args.lr}_epochs={args.num_epochs}_network={args.network}_{args.dataset_name}_{args.class_type}')
        wandb.config.update(args)

    # LABEL MAPPING
    if args.class_type == 'all':
        classnum = 4
    else:
        classnum = 2

    label_mapping = make_label_mapping(args.dataset_name, args.class_type)

    print_num = 63
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    # PATHS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_trained_models = os.path.join(
        args.path_trained_models,
        args.dataset_name,
        args.class_type,
        f"{args.network}-l={args.lr}-e={args.num_epochs}-s={args.step_size}-g={args.gamma}-b={args.batch_size}-cw={args.unsup_weight}-w={args.warm_up_duration}-{args.sup_mark}{args.input1}-{args.input2}-{timestamp}"
    )
    os.makedirs(path_trained_models, exist_ok=True)

    if is_main_process:
        print(f"Models saved to: {path_trained_models}")

    # DATA LOADERS
    data_files = dict(train=args.train_list, val=args.val_list)
    raw_loaders = get_loaders(data_files, args.selected_modal, args.batch_size, num_workers=8)

    if is_ddp:
        train_sampler = DistributedSampler(raw_loaders['train'].dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(raw_loaders['val'].dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = torch.utils.data.DataLoader(
            raw_loaders['train'].dataset,
            batch_size=max(1, args.batch_size // world_size),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            raw_loaders['val'].dataset,
            batch_size=max(1, args.batch_size // world_size),
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = raw_loaders['train']
        val_loader = raw_loaders['val']

    num_batches = {'train_sup': len(train_loader), 'val': len(val_loader)}

    if is_main_process:
        print(f"Data loaded: {num_batches['train_sup']} train, {num_batches['val']} val batches")

    # MODEL
    model = HFFNet(4, 16, classnum)
    model=model.cuda()
    # model.to(device)

    if is_ddp:
        model = DDP(model, device_ids=[int(str(device).split(':')[1])], find_unused_parameters=True)
        if is_main_process:
            print(f"Model wrapped with DDP")
    else:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            if is_main_process:
                print(f"Model wrapped with DataParallel")

    # DROPOUT SCHEDULER
    dropout_scheduler = DropoutScheduler(model, base_dropout=0.5)
    dropout_scheduler.set_dropout_rate(0.5)

    # CRITERIONS
    criterion = segmentation_loss(args.loss, False, cn=classnum).to(device)
    FFcriterion = segmentation_loss(args.loss2, True).to(device)

    # OPTIMIZER & SCHEDULER
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5.3 * 10 ** args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)

    # AMP SCALER
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # LOAD CHECKPOINT IF PROVIDED
    start_epoch = 0
    if args.resume_checkpoint is not None:
        start_epoch = load_checkpoint_from_args(model, optimizer, scheduler_warmup, args.resume_checkpoint, args.resume_epoch, rank, device)

    # TRAINING LOOP
    since = time.time()
    count_iter = 0
    important_weights = None
    best_model = model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(1)]

    for epoch in range(start_epoch, args.num_epochs):
        # Set epoch for distributed sampler
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        # Dropout decay
        current_dropout = 0.5 - 0.4 * (epoch / args.num_epochs)
        dropout_scheduler.set_dropout_rate(max(current_dropout, 0.1))

        if is_main_process and epoch % 5 == 0:
            print(f"Epoch {epoch+1} - Dropout rate: {max(current_dropout, 0.1):.4f}")

        count_iter += 1

        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        # MC-DROPOUT (only on main, every 10 epochs)
        if epoch % 10 == 0 and is_main_process:
            print(f"Epoch {epoch+1} - Running MC-Dropout Uncertainty Estimation...")
            model.eval()
            mc_dropout = MCDropoutUncertainty(model, num_samples=20, device=device)
            val_iter = iter(val_loader)
            try:
                data = next(val_iter)
                low_freq_inputs = []
                high_freq_inputs = []
                for j in range(20):
                    input_tensor = data[j].unsqueeze(1).type(torch.cuda.FloatTensor)
                    if j in [0,1,2,3]:
                        low_freq_inputs.append(input_tensor)
                    else:
                        high_freq_inputs.append(input_tensor)
                low_freq_inputs = torch.cat(low_freq_inputs, dim=1)
                high_freq_inputs = torch.cat(high_freq_inputs, dim=1)

                mc_outputs = mc_dropout.mc_forward_pass(low_freq_inputs, high_freq_inputs)
                mean_pred, uncertainty_map = mc_dropout.compute_uncertainty_maps(mc_outputs)
                print(f"Epoch {epoch+1} - Mean MC-Dropout Uncertainty: {uncertainty_map.mean():.6f}")
                print(f"Epoch {epoch+1} - Std MC-Dropout Uncertainty: {uncertainty_map.std():.6f}")
            except:
                pass
            model.train()

        # TRAINING
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

        if epoch == args.warm_up_duration and is_main_process:
            fisher_info = compute_fisher_information(model, train_loader, criterion, rank)
            important_weights = important_weights_with_fisher(model, fisher_info, std_multiplier=1)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name == 'input_ed.conv.weight':
                        param[important_weights] = (model.module if isinstance(model, DDP) else model).laplacian_target[important_weights]

        optimizer.zero_grad()
        for i, data in enumerate(tqdm(train_loader, disable=not is_main_process)):
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

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs_train_1, outputs_train_2, side1, side2 = model(low_freq_inputs, high_freq_inputs)

                loss_train_sup1 = criterion(outputs_train_1, mask_train)
                loss_train_sup2 = criterion(outputs_train_2, mask_train)
                loss_train_side1 = criterion(side1, mask_train)
                loss_train_side2 = criterion(side2, mask_train)
                
                model_obj = model.module if isinstance(model, DDP) else model
                reg_loss = torch.sum((model_obj.input_ed.conv.weight - model_obj.laplacian_target) ** 2) * reg_weight
                loss_train_unsup = FFcriterion(outputs_train_1, outputs_train_2) * unsup_weight
                loss_train_total = loss_train_sup1 + loss_train_sup2 + loss_train_unsup + loss_train_side1 + loss_train_side2 + reg_loss
                loss = loss_train_total / args.grad_accum_steps

            scaler.scale(loss).backward()

            if important_weights is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name == 'input_ed.conv.weight' and param.grad is not None:
                            param.grad[important_weights] = 0

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_reg += reg_loss.item()
            train_loss_side1 += loss_train_side1.item()
            train_loss_side2 += loss_train_side2.item()
            train_loss_unsup += loss_train_unsup.item()
            train_loss += loss_train_total.item()

            torch.cuda.empty_cache()

        scheduler_warmup.step()

        # LOGGING
        if is_main_process:
            wandb.log({
                "train_loss_sup1": train_loss_sup_1 / num_batches['train_sup'],
                "train_loss_sup2": train_loss_sup_2 / num_batches['train_sup'],
                "train_loss": train_loss / num_batches['train_sup'],
                "train_loss_side1": train_loss_side1 / num_batches['train_sup'],
                "train_loss_side2": train_loss_side2 / num_batches['train_sup'],
                "train_loss_unsup": train_loss_unsup / num_batches['train_sup'],
                "reg_loss": train_loss_reg / num_batches['train_sup'],
                "epoch": epoch
            })

        # VALIDATION
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            val_loss_sup_1 = 0.0
            val_loss_sup_2 = 0.0

            for i, data in enumerate(tqdm(val_loader, disable=not is_main_process)):
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

        # CHECKPOINT SAVING (only rank 0)
        save_checkpoint(model, optimizer, scheduler_warmup, epoch, path_trained_models, rank)

        # PRINT & LOGGING (only rank 0)
        if is_main_process and (count_iter % args.display_iter == 0):
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')

            train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_cps, reg1, train_epoch_loss = print_train_loss(
                train_loss_sup_1, train_loss_sup_2, train_loss_unsup, train_loss_reg, train_loss, num_batches, print_num, print_num_half)

            val_epoch_loss_sup_1, val_epoch_loss_sup_2 = print_val_loss(
                val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half)

            val_eval_list_1, val_eval_list_2, val_m_dc_1, val_m_dc_2 = print_val_eval(
                classnum, score_list_val_1, score_list_val_2, mask_list_val, print_num_half)

            wandb.log({
                'val_loss_sup1': val_loss_sup_1 / num_batches['val'],
                'val_loss_sup2': val_loss_sup_2 / num_batches['val'],
                'Val WT Dice 1': val_eval_list_1[0],
                'Val WT HD95 1': val_eval_list_1[1],
                'Val WT Dice 2': val_eval_list_2[0],
                'Val WT HD95 2': val_eval_list_2[1]
            })

            best_val_eval_list, best_model, best_result = save_val_best_3d_m(
                classnum, best_model, best_val_eval_list, best_result, model, model,
                score_list_val_1, score_list_val_2, mask_list_val,
                val_eval_list_1, val_eval_list_2, path_trained_models)

            print('-' * print_num)
            print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')

        torch.cuda.empty_cache()

    # TRAINING COMPLETE
    time_elapsed = time.time() - since
    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)

    if is_main_process:
        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)
        print_best(classnum, best_val_eval_list, best_model, best_result, path_trained_models, print_num_minus)
        print('=' * print_num)
        wandb.finish()

    # CLEANUP
    cleanup_ddp()

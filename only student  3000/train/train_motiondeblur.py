import os
import functools
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image motion deblurring')).parse_args()
if opt.use_kernel_response:
    if not (opt.online_2d and opt.ddem_stage == 2 and opt.ddem_scratch_student):
        raise ValueError('Kernel response experiment requires online2d scratch blur-only student')
    if not 0 <= opt.response_loss_weight < float('inf'):
        raise ValueError('response_loss_weight must be finite and nonnegative')
if opt.response_gate_log_interval < 0:
    raise ValueError('response_gate_log_interval must be nonnegative')
if not opt.use_kernel_response and (opt.kernel_response_groups != 1 or opt.response_gate_log_interval):
    raise ValueError('Response groups/statistics require --use_kernel_response')
if not opt.use_kernel_response and opt.kernel_response_variant != 'serial':
    raise ValueError('Non-serial response control requires --use_kernel_response')
if opt.max_iters < 0:
    raise ValueError('max_iters must be nonnegative')
print(opt)

import utils
from dataset.dataset_motiondeblur import *
import online_2d
from training_schedule import cosine_learning_rate_factor
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import csv
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss, DDEMDistillationLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from ema import ModelEMA, rng_state, restore_rng, scheduler_state, restore_scheduler




######### Logs dir ###########
log_dir = os.path.join(opt.save_dir,'motiondeblur',opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
validation_logname = os.path.join(log_dir, 'validation_results.csv')
lr_range_logname = os.path.join(log_dir, 'lr_range_results.csv')
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)
if opt.use_kernel_response:
    import json
    from kernel_response import label_spec, response_loss, gate_stat_fields
    from pathlib import Path
    response_stat_fields = gate_stat_fields(opt.kernel_response_variant)
    config = dict(options=vars(opt), kernel_response=label_spec().specification(opt.online_2d_backend))
    config_path = Path(log_dir) / 'kernel_response_config.json'
    if not opt.resume and config_path.exists():
        raise FileExistsError('Existing response experiment; use resume or a new experiment name')
    if opt.resume and config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved['kernel_response'] != config['kernel_response']:
            raise ValueError('Cannot resume with a different response label definition')
        if saved['options'].get('ema', False) != opt.ema:
            raise ValueError('Resume configuration mismatch: ema')
        # Configs from the previous response experiment have no groups key.
        if saved['options'].get('kernel_response_groups', 1) != opt.kernel_response_groups:
            raise ValueError('Resume configuration mismatch: kernel_response_groups')
        if saved['options'].get('kernel_response_variant', 'serial') != opt.kernel_response_variant:
            raise ValueError('Resume configuration mismatch: kernel_response_variant')
        for key in ('arch', 'ddem_dim', 'ddem_blocks', 'ddem_backbone',
                    'response_loss_mode', 'response_loss_weight', 'train_ps',
                    'batch_size', 'nepoch', 'online_2d_seed',
                    'online_2d_gt_root', 'online_2d_variants_per_gt',
                    'online_2d_pool_refresh_epochs'):
            if saved['options'][key] != vars(opt)[key]:
                raise ValueError('Resume configuration mismatch: ' + key)
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    response_logname = Path(log_dir) / 'kernel_response_losses.csv'
    response_gate_logname = Path(log_dir) / 'kernel_response_gate_stats.csv'


def append_response_gate_stats(epoch, batch_index, stats):
    """Log this batch only; this is not a full-epoch gate average.

DataParallel returns each stage as [global_batch, groups, metrics], so samples
from both GPUs receive equal weight. All inputs are detached summaries.
"""
    fields = ['epoch', 'batch_index', 'stage', 'group', 'samples'] + list(response_stat_fields)
    new_file = not response_gate_logname.exists()
    with response_gate_logname.open('a', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for stage, values in stats.items():
            if values.shape[-1] != len(response_stat_fields):
                raise ValueError('Controller statistics do not match the configured CSV schema')
            means = values.detach().float().mean(dim=0).cpu().tolist()
            for group, metrics in enumerate(means):
                writer.writerow(dict(epoch=epoch, batch_index=batch_index, stage=stage,
                                     group=group, samples=values.shape[0],
                                     **dict(zip(response_stat_fields, metrics))))


def learning_rate_summary(optimizer):
    return ';'.join(
        '{}={:.8e}'.format(
            group.get('group_name', 'model'), group['lr'])
        for group in optimizer.param_groups)


def append_validation_result(
        record_type, epoch, global_iteration, train_epoch_loss,
        validation_psnr, input_psnr, best_psnr, best_epoch,
        is_new_best, learning_rates):
    with open(validation_logname, 'a', newline='') as csv_file:
        csv.writer(csv_file).writerow([
            record_type,
            epoch,
            global_iteration,
            '' if train_epoch_loss is None else '{:.8f}'.format(
                train_epoch_loss),
            '{:.6f}'.format(validation_psnr),
            '' if input_psnr is None else '{:.6f}'.format(input_psnr),
            '' if best_psnr is None else '{:.6f}'.format(best_psnr),
            '' if best_epoch is None else best_epoch,
            int(is_new_best),
            learning_rates,
        ])

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
requested_gpu_count = len([gpu for gpu in opt.gpu.split(',') if gpu.strip()])
visible_gpu_count = torch.cuda.device_count()
if visible_gpu_count < 1:
    raise RuntimeError("No CUDA GPU is visible to PyTorch.")
if visible_gpu_count != requested_gpu_count:
    raise RuntimeError(
        "Requested {} GPU(s) via --gpu {}, but PyTorch sees {} visible GPU(s). "
        "Check CUDA_VISIBLE_DEVICES and the machine GPU count.".format(
            requested_gpu_count, opt.gpu, visible_gpu_count))
if opt.batch_size % visible_gpu_count != 0:
    raise ValueError(
        "Global batch size {} must be divisible by {} visible GPUs.".format(
            opt.batch_size, visible_gpu_count))
print("Using {} GPUs with global batch size {} ({} samples/GPU).".format(
    visible_gpu_count, opt.batch_size, opt.batch_size // visible_gpu_count))

model_restoration = utils.get_arch(opt)

if opt.resume and opt.init_weights:
    raise ValueError("--resume and --init_weights are mutually exclusive")
if opt.adapter_warmup_epochs < 0:
    raise ValueError("--adapter_warmup_epochs must be >= 0")
if not 0.0 < opt.backbone_lr_scale <= 1.0:
    raise ValueError("--backbone_lr_scale must be in (0, 1]")
if opt.kd_grad_stat_interval < 0:
    raise ValueError("--kd_grad_stat_interval must be >= 0")
if opt.lr_range_test:
    if opt.resume or opt.init_weights:
        raise ValueError(
            "--lr_range_test must start from random initialization; "
            "do not use --resume or --init_weights")
    if not opt.disable_validation:
        raise ValueError(
            "--lr_range_test does not use validation; add "
            "--disable_validation")
    if not 0.0 < opt.lr_range_min < opt.lr_range_max:
        raise ValueError(
            "--lr_range_test requires 0 < lr_range_min < lr_range_max")
    if opt.lr_range_iters < 2:
        raise ValueError("--lr_range_iters must be >= 2")
    if not 0.0 <= opt.lr_range_smooth_beta < 1.0:
        raise ValueError("--lr_range_smooth_beta must be in [0, 1)")
    if opt.lr_range_diverge_factor <= 1.0:
        raise ValueError("--lr_range_diverge_factor must be > 1")
    if not 0 <= opt.lr_range_burnin < opt.lr_range_iters:
        raise ValueError(
            "--lr_range_burnin must be in [0, lr_range_iters)")

scratch_student = (
    opt.use_ddem and opt.ddem_stage == 2 and opt.ddem_scratch_student)
distillation_student = (
    opt.use_ddem and opt.ddem_stage == 2 and not scratch_student)

if opt.ddem_scratch_student and (
        not opt.use_ddem or opt.ddem_stage != 2):
    raise ValueError(
        "--ddem_scratch_student requires --use_ddem --ddem_stage 2")
if opt.lr_range_test and not scratch_student:
    raise ValueError(
        "--lr_range_test currently requires the pure Blur-only setup: "
        "--use_ddem --ddem_stage 2 --ddem_scratch_student")
if scratch_student:
    if opt.ddem_stage1_weights:
        raise ValueError(
            "Scratch student must not receive --ddem_stage1_weights")
    if opt.ddem_kd_weight != 0.0:
        raise ValueError(
            "Scratch student has no teacher; set --ddem_kd_weight 0")
    print(
        "Pure scratch student: Blur-only DDEM, random initialization, "
        "no teacher network, and no KD loss.")

if distillation_student:
    if not opt.ddem_stage1_weights:
        raise ValueError(
            "DDEM stage 2 requires --ddem_stage1_weights")
    if not os.path.isfile(opt.ddem_stage1_weights):
        raise FileNotFoundError(opt.ddem_stage1_weights)
    if not opt.resume:
        utils.load_stage2_from_stage1(
            model_restoration, opt.ddem_stage1_weights)

if opt.init_weights:
    if not os.path.isfile(opt.init_weights):
        raise FileNotFoundError(opt.init_weights)
    if distillation_student:
        raise ValueError(
            "Stage-2 already initializes from --ddem_stage1_weights; "
            "do not also pass --init_weights")
    utils.load_initial_weights(model_restoration, opt.init_weights)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
named_parameters = list(model_restoration.named_parameters())

def is_prior_parameter(name):
    return (
        name.startswith('ddem.') or
        '.z0_adapter_second.' in name or
        '.z1_adapter_second.' in name or
        '.response_local_second.' in name)

def calculate_ddem_gradient_stats(
        restoration_loss, raw_kd_loss, parameters, kd_weight, eps=1e-12):
    """Measure both losses on the DDEM parameters without changing ``.grad``."""
    parameters = [
        parameter for parameter in parameters if parameter.requires_grad]
    restoration_gradients = torch.autograd.grad(
        restoration_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True)
    kd_gradients = torch.autograd.grad(
        raw_kd_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True)

    rest_squared_norm = torch.zeros(
        (), device=restoration_loss.device, dtype=torch.float32)
    kd_squared_norm = torch.zeros_like(rest_squared_norm)
    gradient_dot = torch.zeros_like(rest_squared_norm)
    for restoration_gradient, kd_gradient in zip(
            restoration_gradients, kd_gradients):
        if restoration_gradient is not None:
            restoration_gradient = restoration_gradient.detach().float()
            rest_squared_norm += restoration_gradient.square().sum()
        if kd_gradient is not None:
            kd_gradient = kd_gradient.detach().float()
            kd_squared_norm += kd_gradient.square().sum()
        if restoration_gradient is not None and kd_gradient is not None:
            gradient_dot += (restoration_gradient * kd_gradient).sum()

    g_rest = torch.sqrt(rest_squared_norm)
    g_kd_raw = torch.sqrt(kd_squared_norm)
    g_kd_effective = abs(float(kd_weight)) * g_kd_raw
    return {
        'g_rest': g_rest.item(),
        'g_kd_raw': g_kd_raw.item(),
        'g_kd_effective': g_kd_effective.item(),
        'kd_rest_ratio': (
            g_kd_effective / (g_rest + eps)).item(),
        'gradient_cosine': (
            gradient_dot / (g_rest * g_kd_raw + eps)).item(),
    }

if opt.use_ddem:
    prior_parameters = [
        parameter for name, parameter in named_parameters
        if is_prior_parameter(name)]
    ddem_parameters = [
        parameter for name, parameter in named_parameters
        if name.startswith('ddem.')]
    backbone_parameters = [
        parameter for name, parameter in named_parameters
        if not is_prior_parameter(name)]
    if not prior_parameters or not ddem_parameters or not backbone_parameters:
        raise RuntimeError(
            "Failed to separate DDEM/adapter and restoration parameters")
    optimizer_parameters = [
        {
            'params': backbone_parameters,
            'lr': opt.lr_initial * opt.backbone_lr_scale,
            'group_name': 'backbone',
        },
        {
            'params': prior_parameters,
            'lr': opt.lr_initial,
            'group_name': 'ddem_adapter',
        },
    ]
    print(
        "Optimizer groups: restoration={} tensors at {:.3e}; "
        "DDEM/adapters={} tensors at {:.3e}.".format(
            len(backbone_parameters),
            opt.lr_initial * opt.backbone_lr_scale,
            len(prior_parameters), opt.lr_initial))
else:
    prior_parameters = []
    ddem_parameters = []
    backbone_parameters = [
        parameter for _, parameter in named_parameters]
    optimizer_parameters = model_restoration.parameters()

if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(optimizer_parameters, lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(optimizer_parameters, lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration) 
model_restoration.cuda() 

teacher_ddem = None
if distillation_student:
    from model import DualDegradationEstimationModule
    teacher_ddem = DualDegradationEstimationModule(
        in_channels=opt.dd_in * 2, prior_dim=opt.ddem_dim,
        num_blocks=opt.ddem_blocks, backbone=opt.ddem_backbone,
        num_heads=opt.ddem_heads, win_size=opt.ddem_win_size)
    utils.load_teacher_ddem(teacher_ddem, opt.ddem_stage1_weights)
    teacher_ddem.eval()
    for parameter in teacher_ddem.parameters():
        parameter.requires_grad_(False)
    teacher_ddem = torch.nn.DataParallel(teacher_ddem).cuda()
     

######### Scheduler ###########
if opt.lr_range_test:
    scheduler = None
    print(
        "LR range test mode: warm-up and the normal LR scheduler are "
        "disabled.")
elif opt.max_iters > 0:
    # LambdaLR evaluates this function at step 0 before the first training batch.
    lr_factor = functools.partial(
        cosine_learning_rate_factor,
        total_iters=opt.max_iters,
        warmup_iters=opt.warmup_iters if opt.warmup else 0,
        min_factor=1e-6 / opt.lr_initial,
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    print(f'Using iteration warmup/cosine for {opt.max_iters:,} batches')
elif opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

ema = ModelEMA() if opt.ema else None
resume_state = None
continuation_state = None
######### Resume ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 

    continuation_state = torch.load(path_chk_rest, map_location='cpu')
    if opt.max_iters > 0:
        if continuation_state.get('max_iters', 0) != opt.max_iters:
            raise ValueError(
                'Resume requires the original max_iters. Start a new run '
                'when changing the training budget.')
        if continuation_state.get('warmup_iters') != opt.warmup_iters:
            raise ValueError('Resume requires the original warmup_iters')
        restore_scheduler(scheduler, continuation_state['scheduler'])
        # Loading scheduler state does not update the optimizer's current LR.
        for group, learning_rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group['lr'] = learning_rate

    if opt.ema:
        resume_state = continuation_state
        if 'ema' not in resume_state:
            raise ValueError('EMA resume requires an EMA training checkpoint')
        if resume_state.get('weights_kind', 'raw') != 'raw':
            raise ValueError('Resume from model_latest/model_epoch, not EMA inference weights')
        ema.load_state_dict(resume_state['ema'], model_restoration)
        if not opt.max_iters:
            restore_scheduler(scheduler, resume_state['scheduler'])
    elif not opt.max_iters:
        for i in range(1, start_epoch):
            scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')


######### Loss ###########
criterion = CharbonnierLoss().cuda()
criterion_ddem = None
if distillation_student:
    criterion_ddem = DDEMDistillationLoss(
        # Keep the criterion raw so gradient statistics can apply lambda
        # explicitly. The optimized total remains mathematically unchanged.
        weight=1.0,
        temperature=opt.ddem_temperature).cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
if opt.use_kernel_response:
    img_options_train.update(use_kernel_response=True, response_backend=opt.online_2d_backend)


def build_train_loader_for_epoch(epoch):
    if opt.online_2d:
        manifest_path = online_2d.generate_epoch(opt, epoch)
        dataset = get_online_2d_training_data(
            manifest_path,
            img_options_train,
            training_epoch=epoch,
            sample_seed=opt.online_2d_seed,
        )
        print(
            "[Online 2D] Epoch {} training pairs: {}".format(
                epoch, len(dataset)),
            flush=True,
        )
    else:
        dataset = get_training_data(opt.train_dir, img_options_train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.train_workers,
        pin_memory=False,
        drop_last=opt.train_drop_last,
    )
    return dataset, loader


train_dataset, train_loader = build_train_loader_for_epoch(start_epoch)

if not len(train_loader):
    raise ValueError('Training loader is empty')
if opt.max_iters:
    # The final epoch can be partial; the batch loop enforces the exact budget.
    opt.nepoch = (opt.max_iters + len(train_loader) - 1) // len(train_loader)
    print('Iteration budget: {}; batches/epoch: {}; final epoch: {}'.format(
        opt.max_iters, len(train_loader), opt.nepoch))
len_trainset = train_dataset.__len__()
if opt.disable_validation:
    val_dataset = None
    val_loader = None
    len_valset = 0
    print("Sizeof training set: ", len_trainset)
    print("Validation is disabled; model_best.pth will not be generated.")
else:
    img_options_val = {'patch_size':opt.val_ps}
    val_dataset = get_validation_deblur_data(opt.val_dir, img_options_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
    len_valset = val_dataset.__len__()
    print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
    if not opt.resume or not os.path.isfile(validation_logname):
        with open(validation_logname, 'w', newline='') as csv_file:
            csv.writer(csv_file).writerow([
                'record_type',
                'epoch',
                'global_iteration',
                'train_epoch_loss',
                'validation_psnr',
                'input_psnr',
                'best_psnr',
                'best_epoch',
                'is_new_best',
                'learning_rates',
            ])
    print("Validation results: ", validation_logname)

    ######### initial validation ###########
    with torch.no_grad():
        model_restoration.eval()
        psnr_dataset = []
        psnr_model_init = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            with torch.cuda.amp.autocast():
                if opt.use_ddem and opt.ddem_stage == 1:
                    restored = model_restoration(input_, gt=target)
                else:
                    restored = model_restoration(input_)
                restored = torch.clamp(restored,0,1)
            psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
            psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
        psnr_dataset = sum(psnr_dataset)/len_valset
        psnr_model_init = sum(psnr_model_init)/len_valset
        print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))
        append_validation_result(
            record_type='initial',
            epoch=start_epoch - 1,
            global_iteration=(start_epoch - 1) * len(train_loader),
            train_epoch_loss=None,
            validation_psnr=psnr_model_init,
            input_psnr=psnr_dataset,
            best_psnr=None,
            best_epoch=None,
            is_new_best=False,
            learning_rates=learning_rate_summary(optimizer))
    model_restoration.train()

######### train ###########
loss_scaler = NativeScaler()
if resume_state is not None:
    loss_scaler.load_state_dict(resume_state['scaler'])
elif opt.max_iters and continuation_state is not None:
    loss_scaler.load_state_dict(continuation_state['scaler'])
torch.cuda.empty_cache()

if opt.lr_range_test:
    print(
        "===> LR range test: {:.3e} -> {:.3e} over {} iterations".format(
            opt.lr_range_min, opt.lr_range_max, opt.lr_range_iters))
    print("LR range results: ", lr_range_logname)

    with open(lr_range_logname, 'w', newline='') as csv_file:
        csv.writer(csv_file).writerow([
            'iteration',
            'epoch_fraction',
            'lr',
            'raw_loss',
            'smoothed_loss',
            'best_smoothed_loss',
            'learning_rates',
        ])

    group_lr_scales = [
        group['lr'] / opt.lr_initial
        for group in optimizer.param_groups]
    lr_growth = (
        opt.lr_range_max / opt.lr_range_min
    ) ** (1.0 / float(opt.lr_range_iters - 1))
    smoothed_loss_ema = 0.0
    best_smoothed_loss = float('inf')
    completed_iterations = 0
    stop_reason = 'maximum_iterations_reached'
    train_iterator = iter(train_loader)
    model_restoration.train()

    for range_iteration in range(1, opt.lr_range_iters + 1):
        scan_lr = opt.lr_range_min * (
            lr_growth ** (range_iteration - 1))
        for group, group_scale in zip(
                optimizer.param_groups, group_lr_scales):
            group['lr'] = scan_lr * group_scale

        try:
            data = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            data = next(train_iterator)

        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()

        with torch.cuda.amp.autocast():
            restored, _ = model_restoration(
                input_, return_ddem_features=True)
            range_loss = criterion(restored, target)

        raw_loss = float(range_loss.detach().item())
        if not np.isfinite(raw_loss):
            stop_reason = 'non_finite_loss'
            break

        smoothed_loss_ema = (
            opt.lr_range_smooth_beta * smoothed_loss_ema
            + (1.0 - opt.lr_range_smooth_beta) * raw_loss)
        bias_correction = (
            1.0 - opt.lr_range_smooth_beta ** range_iteration)
        smoothed_loss = smoothed_loss_ema / bias_correction
        best_smoothed_loss = min(best_smoothed_loss, smoothed_loss)

        grad_clip = opt.grad_clip if opt.grad_clip > 0 else None
        loss_scaler(
            range_loss,
            optimizer,
            clip_grad=grad_clip,
            parameters=model_restoration.parameters())
        completed_iterations = range_iteration

        with open(lr_range_logname, 'a', newline='') as csv_file:
            csv.writer(csv_file).writerow([
                range_iteration,
                '{:.6f}'.format(
                    range_iteration / float(len(train_loader))),
                '{:.10e}'.format(scan_lr),
                '{:.8f}'.format(raw_loss),
                '{:.8f}'.format(smoothed_loss),
                '{:.8f}'.format(best_smoothed_loss),
                learning_rate_summary(optimizer),
            ])

        if (
                range_iteration % 50 == 0
                or range_iteration == 1):
            print(
                "[LR range] iter={}/{} lr={:.3e} "
                "raw_loss={:.6f} smooth_loss={:.6f} best={:.6f}".format(
                    range_iteration,
                    opt.lr_range_iters,
                    scan_lr,
                    raw_loss,
                    smoothed_loss,
                    best_smoothed_loss))

        if (
                range_iteration >= opt.lr_range_burnin
                and smoothed_loss
                > opt.lr_range_diverge_factor * best_smoothed_loss):
            stop_reason = 'smoothed_loss_diverged'
            break

    summary_line = (
        "LR range test finished: completed_iterations={}, "
        "last_lr={:.6e}, best_smoothed_loss={:.8f}, reason={}, "
        "results={}".format(
            completed_iterations,
            opt.lr_range_min * (
                lr_growth ** max(0, completed_iterations - 1)),
            best_smoothed_loss,
            stop_reason,
            lr_range_logname))
    print(summary_line)
    with open(logname, 'a') as f:
        f.write(summary_line + '\n')
    sys.exit(0)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
if opt.resume and os.path.isfile(validation_logname):
    with open(validation_logname, 'r', newline='') as csv_file:
        for validation_row in csv.DictReader(csv_file):
            if validation_row['record_type'].endswith('_ema'):
                continue
            saved_best_psnr = validation_row.get('best_psnr', '')
            saved_best_epoch = validation_row.get('best_epoch', '')
            if saved_best_psnr:
                best_psnr = max(best_psnr, float(saved_best_psnr))
            if (
                    saved_best_psnr
                    and float(saved_best_psnr) >= best_psnr
                    and saved_best_epoch):
                best_epoch = int(saved_best_epoch)
    best_iter = len(train_loader) - 1
    print(
        "Restored validation best from CSV: epoch={}, PSNR={:.6f}".format(
            best_epoch, best_psnr))
best_ema_psnr, best_ema_epoch = 0, 0
if resume_state is not None:
    best_psnr, best_epoch, best_iter = resume_state['best_raw']
    best_ema_psnr, best_ema_epoch = resume_state['best_ema']
    restore_rng(resume_state['rng'])


def training_checkpoint(epoch):
    """Build a checkpoint with the batch count and state needed for resuming."""
    state = dict(epoch=epoch, state_dict=model_restoration.state_dict(),
                 optimizer=optimizer.state_dict(), global_iteration=global_iteration,
                 max_iters=opt.max_iters, warmup_iters=opt.warmup_iters,
                 kernel_response_variant=opt.kernel_response_variant)
    if opt.max_iters > 0:
        state.update(
            scheduler=scheduler_state(scheduler),
            scaler=loss_scaler.state_dict(),
        )
    if ema is not None:
        state.update(ema=ema.state_dict(), scheduler=scheduler_state(scheduler),
                     scaler=loss_scaler.state_dict(), rng=rng_state(),
                     best_raw=(best_psnr, best_epoch, best_iter),
                     best_ema=(best_ema_psnr, best_ema_epoch), weights_kind='raw')
    return state


val_epoch_interval = max(1, opt.val_epoch_interval)
if not opt.disable_validation:
    print(
        "\nValidation once every {} epoch(s), at the end of the epoch.\n".format(
            val_epoch_interval))

global_iteration = (start_epoch - 1) * len(train_loader)
if opt.max_iters > 0 and continuation_state is not None:
    global_iteration = continuation_state['global_iteration']

# An iteration means one processed global batch. AMP can skip an optimizer
# update on overflow; the iteration budget still counts that batch.
iteration_mode = opt.max_iters > 0
if distillation_student and opt.kd_grad_stat_interval > 0:
    print(
        "DDEM gradient statistics every {} training iterations "
        "(approximately {:.2f} epochs).".format(
            opt.kd_grad_stat_interval,
            opt.kd_grad_stat_interval / float(len(train_loader))))
for epoch in range(start_epoch, opt.nepoch + 1):
    if iteration_mode and global_iteration >= opt.max_iters:
        break
    if ema is not None:
        ema.begin_epoch(model_restoration, epoch)
    if opt.online_2d and epoch != start_epoch:
        train_dataset, train_loader = build_train_loader_for_epoch(epoch)
        len_trainset = len(train_dataset)
    adapter_warmup_active = (
        opt.use_ddem and epoch <= opt.adapter_warmup_epochs)
    for parameter in backbone_parameters:
        parameter.requires_grad_(not adapter_warmup_active)
    if adapter_warmup_active and epoch == start_epoch:
        print(
            "Adapter warm-up active through epoch {}: restoration backbone "
            "is frozen; only DDEM and stage adapters are trainable.".format(
                opt.adapter_warmup_epochs))
    if (opt.use_ddem and opt.adapter_warmup_epochs > 0 and
            epoch == opt.adapter_warmup_epochs + 1):
        print(
            "Adapter warm-up complete: restoration backbone unfrozen at "
            "its scheduled {:.2f}x LR.".format(opt.backbone_lr_scale))
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_kd_loss = 0
    response_sums = {}
    response_batches = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0): 
        # Count processed global batches; AMP may skip an update on overflow.
        global_iteration += 1
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()
        collect_response_stats = (
            opt.use_kernel_response and opt.response_gate_log_interval > 0 and i == 0
            and (epoch - 1) % opt.response_gate_log_interval == 0)

        with torch.cuda.amp.autocast():
            ddem_features = None
            if opt.use_ddem:
                if opt.ddem_stage == 1:
                    restored, ddem_features = model_restoration(
                        input_, gt=target, return_ddem_features=True)
                else:
                    restored, ddem_features = model_restoration(
                        input_, return_ddem_features=True,
                        collect_response_stats=collect_response_stats)
            else:
                restored = model_restoration(input_)

            restoration_loss = criterion(restored, target)
            loss = restoration_loss
            if opt.use_kernel_response:
                response_target = data[4].cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=False):
                    response_raw, response_metrics = response_loss(
                        ddem_features['kernel_response'], response_target, opt.response_loss_mode)
                loss = loss + opt.response_loss_weight * response_raw
                response_metrics['auxiliary_raw'] = response_raw.detach()
                response_metrics['restoration'] = restoration_loss.detach()
                for key, value in response_metrics.items():
                    response_sums[key] = response_sums.get(key, 0.0) + float(value)
                response_batches += 1
            raw_kd_loss = None
            if distillation_student:
                with torch.no_grad():
                    teacher_features = teacher_ddem(
                        torch.cat([input_, target], dim=1))
                raw_kd_loss = criterion_ddem(
                    ddem_features['z_tilde'],
                    teacher_features['z_tilde'])
                kd_loss = opt.ddem_kd_weight * raw_kd_loss
                loss = loss + kd_loss
                epoch_kd_loss += kd_loss.item()

        if collect_response_stats:
            append_response_gate_stats(epoch, i, ddem_features['response_gate_stats'])

        # Stop at the first invalid forward pass. Continuing after a NaN loss
        # can corrupt AdamW's moving averages and every later checkpoint.
        if not torch.isfinite(loss).item():
            finite_details = [
                "restored={:.2%}".format(
                    torch.isfinite(restored).float().mean().item())
            ]
            if ddem_features is not None:
                for name in ('z_tilde', 'z0', 'z1', 'kernel_response'):
                    value = ddem_features.get(name)
                    if torch.is_tensor(value):
                        finite_details.append(
                            "{}={:.2%}".format(
                                name,
                                torch.isfinite(value).float().mean().item()))
            raise FloatingPointError(
                "Non-finite loss at epoch {}, iteration {}. Finite ratios: "
                "{}. Training stopped before backward/optimizer.step().".format(
                    epoch, i + 1, ", ".join(finite_details)))

        if (
                raw_kd_loss is not None and
                opt.kd_grad_stat_interval > 0 and
                global_iteration % opt.kd_grad_stat_interval == 0):
            gradient_stats = calculate_ddem_gradient_stats(
                restoration_loss,
                raw_kd_loss,
                ddem_parameters,
                opt.ddem_kd_weight)
            gradient_stats_line = (
                "[KD gradient] global_iter={} epoch={} iter={}/{} "
                "g_rest={:.6e} g_kd_raw={:.6e} "
                "lambda_g_kd={:.6e} ratio={:.6f} cosine={:.6f}".format(
                    global_iteration,
                    epoch,
                    i + 1,
                    len(train_loader),
                    gradient_stats['g_rest'],
                    gradient_stats['g_kd_raw'],
                    gradient_stats['g_kd_effective'],
                    gradient_stats['kd_rest_ratio'],
                    gradient_stats['gradient_cosine']))
            print(gradient_stats_line)
            with open(logname, 'a') as f:
                f.write(gradient_stats_line + '\n')

        # The optimization protocol must not change when ablating DDEM.
        grad_clip = opt.grad_clip if opt.grad_clip > 0 else None
        scale_before = loss_scaler.state_dict().get('scale', 1.0) if ema is not None else None
        loss_scaler(
                loss, optimizer, clip_grad=grad_clip,
                parameters=model_restoration.parameters())
        if ema is not None:
            ema.update(model_restoration,
                       loss_scaler.state_dict().get('scale', 1.0) >= scale_before)
        if iteration_mode:
            scheduler.step()
        epoch_loss += loss.item()

        # Always validate the final batch, including a partial final epoch.
        reached_iteration_limit = iteration_mode and global_iteration >= opt.max_iters
        completed_epoch = i + 1 == len(train_loader)
        scheduled_validation = completed_epoch and epoch % val_epoch_interval == 0
        should_validate = (
            not opt.disable_validation
            and (scheduled_validation or reached_iteration_limit)
        )
        if should_validate:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        if opt.use_ddem and opt.ddem_stage == 1:
                            restored = model_restoration(input_, gt=target)
                        else:
                            restored = model_restoration(input_)
                    restored = torch.clamp(restored,0,1)  
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                
                is_new_best = psnr_val_rgb > best_psnr
                if is_new_best:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'global_iteration': global_iteration,
                                'max_iters': opt.max_iters,
                                'kernel_response_variant': opt.kernel_response_variant
                                }, os.path.join(model_dir,"model_best.pth"))

                append_validation_result(
                    record_type='periodic',
                    epoch=epoch,
                    global_iteration=global_iteration,
                    train_epoch_loss=epoch_loss,
                    validation_psnr=psnr_val_rgb,
                    input_psnr=None,
                    best_psnr=best_psnr,
                    best_epoch=best_epoch,
                    is_new_best=is_new_best,
                    learning_rates=learning_rate_summary(optimizer))
                print("[Ep %d it %d\t Validation PSNR: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t Validation PSNR: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                if ema is not None and ema.model is not None:
                    # Extra validation must not advance training/data-loader RNGs.
                    saved_rng = rng_state()
                    try:
                        ema_psnr = []
                        for data_val in val_loader:
                            target, input_ = data_val[0].cuda(), data_val[1].cuda()
                            with torch.cuda.amp.autocast():
                                restored = (ema.model(input_, gt=target)
                                            if opt.use_ddem and opt.ddem_stage == 1
                                            else ema.model(input_))
                            ema_psnr.append(utils.batch_PSNR(
                                restored.clamp(0, 1), target, False).item())
                        score = sum(ema_psnr) / len_valset
                        improved = score > best_ema_psnr
                        if improved:
                            best_ema_psnr, best_ema_epoch = score, epoch
                            torch.save(dict(epoch=epoch, state_dict=ema.model.state_dict(),
                                            weights_kind='ema', psnr=score,
                                            kernel_response_variant=opt.kernel_response_variant),
                                       os.path.join(model_dir, 'model_best_ema.pth'))
                        append_validation_result(
                            'periodic_ema', epoch, global_iteration, epoch_loss,
                            score, None, best_ema_psnr, best_ema_epoch, improved,
                            learning_rate_summary(optimizer))
                        print('[EMA] epoch={} PSNR={:.4f} best={:.4f} epoch={}'.format(
                            epoch, score, best_ema_psnr, best_ema_epoch))
                    finally:
                        restore_rng(saved_rng)
                model_restoration.train()
                torch.cuda.empty_cache()
        if reached_iteration_limit:
            break
    if not iteration_mode:
        scheduler.step()
    if opt.use_kernel_response:
        metrics = {k: v / max(1, response_batches) for k, v in response_sums.items()}
        record = dict(epoch=epoch, **metrics)
        new_file = not response_logname.exists()
        with response_logname.open('a', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(record))
            if new_file:
                writer.writeheader()
            writer.writerow(record)
        print('[Kernel response] ' + str(record), flush=True)
    
    print("------------------------------------------------------------------")
    current_lrs = ", ".join(
        "{}={:.6e}".format(
            group.get('group_name', 'model'), group['lr'])
        for group in optimizer.param_groups)
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRates {}".format(epoch, time.time()-epoch_start_time,epoch_loss,current_lrs))
    if distillation_student:
        print("DDEM KD Loss: {:.6f}".format(epoch_kd_loss))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRates {}".format(epoch, time.time()-epoch_start_time,epoch_loss,current_lrs)+'\n')
        if distillation_student:
            f.write("DDEM_KD: {:.6f}".format(epoch_kd_loss)+'\n')

    # Save the terminal state even when training ends inside an epoch.
    should_save_latest = opt.ema or should_validate or reached_iteration_limit
    should_save_checkpoint = epoch % opt.checkpoint == 0 or reached_iteration_limit
    if should_save_latest:
        torch.save(training_checkpoint(epoch), os.path.join(model_dir, 'model_latest.pth'))
    if should_save_checkpoint:
        torch.save(training_checkpoint(epoch), os.path.join(model_dir, 'model_epoch_{}.pth'.format(epoch)))
    elif (opt.use_ddem and opt.adapter_warmup_epochs > 0 and epoch == opt.adapter_warmup_epochs):
        torch.save(training_checkpoint(epoch), os.path.join(
            model_dir, 'model_epoch_{}_adapter_warmup.pth'.format(epoch)))
    if (
            opt.online_2d
            and online_2d.cache_consumed_after_epoch(
                opt, epoch, opt.nepoch)):
        # Keep the pool through its 50-epoch block, then remove it only after
        # the final DataLoader using that pool has finished.
        online_2d.remove_consumed_training_cache(opt, epoch)
print("Now time is : ",datetime.datetime.now().isoformat())

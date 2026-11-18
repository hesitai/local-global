import torch
import torch.nn as nn
import os
from collections import OrderedDict


def _without_data_parallel_prefix(state_dict):
    return OrderedDict(
        (key[7:] if key.startswith('module.') else key, value)
        for key, value in state_dict.items())

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_stage2_from_stage1(model, weights):
    """Initialize the student/restorer from a stage-1 blur+GT checkpoint.

    The teacher DDEM stem accepts six channels whereas the student stem accepts
    three, so only keys whose names and shapes match are loaded.
    """
    checkpoint = torch.load(weights, map_location='cpu')
    source = _without_data_parallel_prefix(checkpoint['state_dict'])
    target = model.state_dict()
    matched = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(matched, strict=False)
    skipped = sorted(set(source) - set(matched))
    print(
        "Stage-2 initialization: loaded {} tensors; skipped {} tensors "
        "(expected: six-channel teacher stem).".format(
            len(matched), len(skipped)))
    return result


def load_initial_weights(model, weights):
    """Load only matching model tensors for a new experiment.

    This intentionally ignores optimizer, scheduler and checkpoint epoch. It
    is used to initialize the V2 DDEM experiment from the exact Stage-last
    baseline while leaving newly added DDEM/adapters at their own init.
    """
    checkpoint = torch.load(weights, map_location='cpu')
    source = _without_data_parallel_prefix(checkpoint['state_dict'])
    target = model.state_dict()
    matched = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(matched, strict=False)
    shape_mismatch = sorted(
        key for key, value in source.items()
        if key in target and target[key].shape != value.shape)
    unexpected = sorted(key for key in source if key not in target)
    print(
        "Model-only initialization: loaded {} tensors from epoch {}; "
        "new/missing={}, shape-mismatch={}, source-only={}.".format(
            len(matched), checkpoint.get('epoch', 'unknown'),
            len(result.missing_keys), len(shape_mismatch), len(unexpected)))
    return result


def load_teacher_ddem(ddem, weights):
    """Load only the DDEM submodule from a stage-1 checkpoint."""
    checkpoint = torch.load(weights, map_location='cpu')
    source = _without_data_parallel_prefix(checkpoint['state_dict'])
    prefix = 'ddem.'
    ddem_state = OrderedDict(
        (key[len(prefix):], value)
        for key, value in source.items() if key.startswith(prefix))
    return ddem.load_state_dict(ddem_state, strict=True)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import Uformer, UNet

    arch = opt.arch
    experimental_kwargs = {
        'use_local_branch': getattr(opt, 'use_local_branch', False),
        'double_process': getattr(opt, 'double_process', False),
        'local_expansion': getattr(opt, 'local_expansion', 2),
        'use_channel_aware': getattr(opt, 'use_channel_aware', False),
        'use_ddem': getattr(opt, 'use_ddem', False),
        'ddem_stage': getattr(opt, 'ddem_stage', 2),
        'ddem_dim': getattr(opt, 'ddem_dim', 32),
        'ddem_blocks': getattr(opt, 'ddem_blocks', 2),
        'ddem_backbone': getattr(opt, 'ddem_backbone', 'local_global'),
        'ddem_heads': getattr(opt, 'ddem_heads', 1),
        'ddem_win_size': getattr(opt, 'ddem_win_size', 8),
        'use_kernel_response': getattr(opt, 'use_kernel_response', False),
        'kernel_response_groups': getattr(opt, 'kernel_response_groups', 1),
        'kernel_response_variant': getattr(opt, 'kernel_response_variant', 'serial'),
    }

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'Uformer':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=8,token_projection='linear',token_mlp='leff',modulator=True,
            **experimental_kwargs)
    elif arch == 'Uformer_T':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',modulator=True,
            **experimental_kwargs)
    elif arch == 'Uformer_S':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True,
            **experimental_kwargs)
    elif arch == 'Uformer_S_noshift':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True,
            shift_flag=False, **experimental_kwargs)
    elif arch == 'Uformer_B_fastleff':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='fastleff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True, **experimental_kwargs)
    elif arch == 'Uformer_B':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=opt.dd_in,
            **experimental_kwargs)
    else:
        raise Exception("Arch error!")

    return model_restoration

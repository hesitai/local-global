"""Supervised response prediction and versioned local-feature controllers.

``serial`` preserves the original expert mixture and checkpoint format.
``joint_residual`` jointly conditions expert selection and residual strength
on image content and the predicted response, retaining the same expert types.
"""
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def label_spec():
    global _spec
    if '_spec' not in globals():
        path = Path(__file__).resolve().parent.parent / 'motion-flow-syn-python-flat/kernel_response_labels.py'
        spec = importlib.util.spec_from_file_location('_joint2d_response_spec', path)
        _spec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_spec)
    return _spec


class ResponseHead(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, channels, 3, 2, 1), nn.SiLU(),
                         nn.Conv2d(channels, channels, 3, 2, 1), nn.SiLU(),
                         nn.Conv2d(channels, 64, 1))


class ResponseConditionedLocal(nn.Module):
    """Four DW experts with spatial, optionally group-wise mixture weights.

Each expert starts near identity. Response describes INPUT pixel frequencies,
not a literal inverse filter on learned features. Spatial alignment and window
rolling happen before mixing; frequency channels are not rescaled per stage.
Groups partition feature channels, not frequencies or attention heads. There
is no external residual/scale. groups=1 preserves the previous weight format.
"""
    def __init__(self, channels, groups=1):
        super().__init__()
        if not isinstance(groups, int) or groups < 1 or channels % groups:
            raise ValueError('Response groups must be positive and divide channels')
        self.groups = groups
        self.num_experts = 4
        self.response_proj = nn.Conv2d(64, 16, 1)
        self.content_proj = nn.Conv2d(channels, 16, 1)
        # Construct the entire legacy controller first, in the original RNG
        # order. Expanding only the final gate must not reinitialize experts,
        # projections, subsequent stages, or alter the training RNG stream.
        self.gate = nn.Conv2d(32, 4, 1)
        self.experts = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                      groups=channels, bias=False) for d in (1, 2, 3, 4)])
        with torch.no_grad():
            for expert in self.experts:
                nn.init.normal_(expert.weight, std=1e-3)
                expert.weight[:, 0, 1, 1] += 1.0
        if groups > 1:
            with torch.random.fork_rng(devices=[]):
                grouped_gate = nn.Conv2d(32, groups * self.num_experts, 1)
                with torch.no_grad():
                    grouped_gate.weight.copy_(self.gate.weight.repeat(groups, 1, 1, 1))
                    grouped_gate.bias.copy_(self.gate.bias.repeat(groups))
            self.gate = grouped_gate

    def forward(self, local, response, shift_size=0, return_stats=False):
        # local is already shifted BHWC; response remains in original BCHW.
        local = local.permute(0, 3, 1, 2).contiguous()
        response = self.response_proj(response.to(dtype=local.dtype))
        response = F.interpolate(response, size=local.shape[-2:], mode='bilinear',
                                 align_corners=False)
        if shift_size:
            response = torch.roll(response, (-shift_size, -shift_size), (-2, -1))
        logits = self.gate(torch.cat((F.silu(self.content_proj(local)),
                                     F.silu(response)), dim=1))
        batch, channels, height, width = local.shape
        # Softmax is over EXPERTS independently for each group and position,
        # never over the flattened groups*experts dimension.
        if self.groups == 1:
            scores = logits.softmax(dim=1)  # exact legacy computation path
        else:
            scores = logits.reshape(batch, self.groups, self.num_experts,
                                    height, width).softmax(dim=2)
        output = None
        for i, expert in enumerate(self.experts):
            filtered = expert(local)  # still depthwise: groups=channels
            if self.groups == 1:
                value = scores[:, i:i+1] * filtered
            else:
                filtered = filtered.reshape(batch, self.groups,
                                            channels // self.groups, height, width)
                value = (scores[:, :, i:i+1] * filtered).reshape_as(local)
            output = value if output is None else output + value
        output = output.permute(0, 2, 3, 1).contiguous()
        if return_stats:
            grouped_scores = scores.unsqueeze(1) if self.groups == 1 else scores
            return output, summarize_gate_scores(grouped_scores)
        return output


class JointResidualResponseConditionedLocal(nn.Module):
    """Content/response interaction with grouped, spatial residual control.

    For projected content C and aligned response Q, the joint feature is
    H = SiLU(Conv([C, Q, C * Q])). Separate heads predict expert probabilities
    w (softmax over experts) and correction strength alpha (sigmoid per group).
    The result is L + alpha * sum_k(w_k * delta_E_k(L)).

    The four delta_E experts retain the original 3x3 depthwise convolutions
    with dilations 1, 2, 3, 4. They start with independent nonzero Kaiming
    weights, not identity filters: the explicit skip already preserves L.
    Gate computations and residual accumulation use FP32 under AMP. Expert
    convolutions follow the surrounding autocast policy.
    """

    def __init__(self, channels, groups=1, initial_strength=0.1):
        super().__init__()
        if not isinstance(groups, int) or groups < 1 or channels % groups:
            raise ValueError('Response groups must be positive and divide channels')
        if not 0.0 < initial_strength < 1.0:
            raise ValueError('initial_strength must lie strictly between 0 and 1')
        self.channels = channels
        self.groups = groups
        self.num_experts = 4
        self.response_proj = nn.Conv2d(64, 16, 1)
        self.content_proj = nn.Conv2d(channels, 16, 1)
        self.fusion = nn.Sequential(nn.Conv2d(48, 32, 1), nn.SiLU())
        self.gate = nn.Conv2d(32, groups * self.num_experts, 1)
        self.strength = nn.Conv2d(32, groups, 1)
        self.experts = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                      groups=channels, bias=False) for d in (1, 2, 3, 4)
        ])
        # Small nonzero heads allow gradients into the joint feature and R
        # immediately. The alpha bias gives approximately 10% initial strength.
        nn.init.normal_(self.gate.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.strength.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.strength.bias,
                          math.log(initial_strength / (1.0 - initial_strength)))

    def forward(self, local, response, shift_size=0, return_stats=False):
        if local.ndim != 4 or local.shape[-1] != self.channels:
            raise ValueError('local must have shape [batch, height, width, channels]')
        if (response.ndim != 4 or response.shape[1] != 64
                or response.shape[0] != local.shape[0]):
            raise ValueError('response must have shape [batch, 64, height, width]')
        local = local.permute(0, 3, 1, 2).contiguous()
        batch, channels, height, width = local.shape
        with torch.cuda.amp.autocast(enabled=False):
            content = F.silu(self.content_proj(local.float()))
            aligned_response = self.response_proj(response.float())
            aligned_response = F.interpolate(
                aligned_response, size=(height, width), mode='bilinear',
                align_corners=False)
            if shift_size:
                aligned_response = torch.roll(
                    aligned_response, (-shift_size, -shift_size), (-2, -1))
            response_feature = F.silu(aligned_response)
            joint = self.fusion(torch.cat(
                (content, response_feature, content * response_feature), dim=1))
            scores = self.gate(joint).reshape(
                batch, self.groups, self.num_experts, height, width).softmax(dim=2)
            strength = self.strength(joint).sigmoid().unsqueeze(2)

        # Accumulate expert corrections in FP32 without materializing a
        # [B, G, experts, channels/G, H, W] stack of all expert outputs.
        correction = None
        for index, expert in enumerate(self.experts):
            filtered = expert(local).float().reshape(
                batch, self.groups, channels // self.groups, height, width)
            weighted = scores[:, :, index:index + 1] * filtered
            correction = weighted if correction is None else correction + weighted
        update = strength * correction
        grouped_local = local.float().reshape(
            batch, self.groups, channels // self.groups, height, width)
        output = (grouped_local + update).reshape_as(local).to(dtype=local.dtype)
        output = output.permute(0, 2, 3, 1).contiguous()
        if return_stats:
            return output, summarize_joint_residual_scores(
                scores, strength, update, grouped_local)
        return output


GATE_STAT_FIELDS = ('expert_1', 'expert_2', 'expert_3', 'expert_4',
                    'entropy', 'spatial_std', 'group_deviation')
JOINT_GATE_STAT_FIELDS = GATE_STAT_FIELDS + (
    'strength_mean', 'strength_spatial_std', 'relative_update_rms')


def gate_stat_fields(variant):
    """Return the CSV/stat tensor schema for a controller variant."""
    if variant == 'serial':
        return GATE_STAT_FIELDS
    if variant == 'joint_residual':
        return JOINT_GATE_STAT_FIELDS
    raise ValueError('Unknown kernel response variant: ' + str(variant))


def summarize_gate_scores(scores):
    """Detached [B,G,7] diagnostics, so DataParallel gathers on batch dim.

Entropy is normalized by log(4). group_deviation measures the per-position
absolute difference from the across-group mean, not just global mean weights.
No hooks, random sampling, graph retention, or training-state updates.
"""
    with torch.no_grad():
        p = scores.detach().float()
        means = p.mean(dim=(-2, -1))
        entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=2).mean(dim=(-2, -1)) / math.log(4)
        spatial_std = p.std(dim=(-2, -1), unbiased=False).mean(dim=2)
        deviation = (p - p.mean(dim=1, keepdim=True)).abs().mean(dim=(2, 3, 4))
        return torch.cat((means, entropy.unsqueeze(-1), spatial_std.unsqueeze(-1),
                          deviation.unsqueeze(-1)), dim=-1)


def summarize_joint_residual_scores(scores, strength, update, local):
    """Return detached [B, G, 10] routing/strength diagnostics, without RNG use."""
    with torch.no_grad():
        routing = summarize_gate_scores(scores)
        alpha = strength.detach().float().squeeze(2)
        mean = alpha.mean(dim=(-2, -1))
        spatial_std = alpha.std(dim=(-2, -1), unbiased=False)
        update_rms = update.detach().float().square().mean(dim=(2, 3, 4)).sqrt()
        local_rms = local.detach().float().square().mean(dim=(2, 3, 4)).sqrt()
        relative_rms = update_rms / local_rms.clamp_min(1e-6)
        return torch.cat((routing, mean.unsqueeze(-1), spatial_std.unsqueeze(-1),
                          relative_rms.unsqueeze(-1)), dim=-1)


def augment_response(response, mode):
    if mode & 1:
        response = response.flip(-1)
    if mode & 2:
        response = response.flip(-2)
    if mode & 4:
        response = response.transpose(-2, -1)
    indices, signs = label_spec().augmentation_channels(mode)
    return (response[indices] * response.new_tensor(signs)[:, None, None]).contiguous()


def response_loss(predicted, target, mode='complex'):
    if predicted.shape != target.shape or predicted.shape[1] != 64:
        raise ValueError(f'Response shape mismatch {predicted.shape}, {target.shape}')
    p, t = predicted.float(), target.float()
    if not torch.isfinite(t).all():
        raise ValueError('Nonfinite response targets')
    # [batch, radius, direction, re/im, height, width]
    p = p.reshape(p.shape[0], 4, 8, 2, *p.shape[-2:])
    t = t.reshape_as(p)
    complex_error = F.smooth_l1_loss(p, t, reduction='none').mean((0, 2, 3, 4, 5))
    pm = (p.square().sum(3) + 1e-8).sqrt()
    tm = (t.square().sum(3) + 1e-8).sqrt()
    mag_error = F.smooth_l1_loss(pm, tm, reduction='none').mean((0, 2, 3, 4))
    if mode not in ('complex', 'magnitude'):
        raise ValueError(mode)
    loss = (complex_error if mode == 'complex' else mag_error).mean()
    metrics = {'complex_low': complex_error[0], 'complex_mid': complex_error[1:3].mean(),
               'complex_high': complex_error[3], 'magnitude_low': mag_error[0],
               'magnitude_mid': mag_error[1:3].mean(), 'magnitude_high': mag_error[3]}
    return loss, {k: v.detach() for k, v in metrics.items()}

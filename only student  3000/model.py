import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum


class FastLeFF(nn.Module):
    
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()

        from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, H=None, W=None):
        # bs x hw x c
        bs, hw, c = x.size()
        H = H or int(math.sqrt(hw))
        W = W or int(math.sqrt(hw))
        if H * W != hw:
            raise ValueError(f"FastLeFF spatial size {(H, W)} does not match token count {hw}")

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)

        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim 
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        return flops

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
        
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


#########################################
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    def flops(self, H, W): 
        flops = H*W*self.in_channel*self.out_channel*(3*3+1)+H*W*self.out_channel*self.out_channel*3*3
        return flops

class UNet(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out

    def flops(self, H, W): 
        flops = 0
        flops += self.ConvBlock1.flops(H, W)
        flops += H/2*W/2*self.dim*self.dim*4*4
        flops += self.ConvBlock2.flops(H/2, W/2)
        flops += H/4*W/4*self.dim*2*self.dim*2*4*4
        flops += self.ConvBlock3.flops(H/4, W/4)
        flops += H/8*W/8*self.dim*4*self.dim*4*4*4
        flops += self.ConvBlock4.flops(H/8, W/8)
        flops += H/16*W/16*self.dim*8*self.dim*8*4*4

        flops += self.ConvBlock5.flops(H/16, W/16)

        flops += H/8*W/8*self.dim*16*self.dim*8*2*2
        flops += self.ConvBlock6.flops(H/8, W/8)
        flops += H/4*W/4*self.dim*8*self.dim*4*2*2
        flops += self.ConvBlock7.flops(H/4, W/4)
        flops += H/2*W/2*self.dim*4*self.dim*2*2*2
        flops += self.ConvBlock8.flops(H/2, W/2)
        flops += H*W*self.dim*2*self.dim*2*2
        flops += self.ConvBlock9.flops(H, W)

        flops += H*W*self.dim*3*3*3
        return flops


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(LPU, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1, groups = in_channels, bias = True
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        result = (self.depthwise(x) + x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return result

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.out_channels*3*3
        return flops

#########################################
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.reduction = reduction

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

    def flops(self): 
        flops = 0
        flops += self.channel*self.channel/self.reduction*2
        
        return flops

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self): 
        flops = 0
        flops += self.channel*self.channel*self.k_size
        
        return flops

class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self): 
        flops = 0
        flops += self.channel*self.channel*self.k_size
        
        return flops

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW): 
        flops = 0
        flops += HW*self.in_channels*self.kernel_size**2/self.stride**2
        flops += HW*self.in_channels*self.out_channels
        print("SeqConv2d:{%.2f}"%(flops/1e9))
        return flops


class OmniLocalBranch(nn.Module):
    """Local information extractor used by Omni-Deblurring.

    Data path: DWConv -> PWConv -> PWConv -> DWConv.  The module accepts and
    returns channel-last feature maps (B, H, W, C), which is the layout used
    inside :class:`LeWinTransformerBlock` before window partitioning.
    """

    def __init__(self, dim, expansion=2, kernel_size=3, bias=False):
        super().__init__()
        if expansion < 1:
            raise ValueError("local branch expansion must be >= 1")
        hidden_dim = dim // expansion
        if hidden_dim < 1:
            raise ValueError(
                f"local branch hidden dimension is zero: dim={dim}, expansion={expansion}"
            )

        padding = kernel_size // 2
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.activation = nn.ReLU6(inplace=True)
        self.depthwise1 = nn.Conv2d(
            dim, dim, kernel_size, stride=1, padding=padding,
            groups=dim, bias=bias)
        self.pointwise1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.pointwise2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        self.depthwise2 = nn.Conv2d(
            dim, dim, kernel_size, stride=1, padding=padding,
            groups=dim, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.activation(self.depthwise1(x))
        x = self.pointwise1(x)
        x = self.activation(self.pointwise2(x))
        x = self.depthwise2(x)
        return x.permute(0, 2, 3, 1).contiguous()

    def flops(self, H, W):
        depthwise = 2 * H * W * self.dim * self.kernel_size ** 2
        pointwise = 2 * H * W * self.dim * self.hidden_dim
        return depthwise + pointwise


class ChannelAwareModule(nn.Module):
    """DiNAT-IR attention-output gate for channel-last feature maps.

    Global spatial pooling followed by a lightweight 1-D convolution models
    interactions between neighboring channels.  The returned (B, 1, 1, C)
    sigmoid scores broadcast over the attention output.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("CAM kernel size must be odd")
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False)
        # With 2*sigmoid below, zero logits make the newly inserted CAM an
        # exact identity at initialization while retaining non-zero gradients.
        nn.init.zeros_(self.conv.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B, H, W, C; score: B, 1, C
        score = x.mean(dim=(1, 2)).unsqueeze(1)
        score = 2.0 * self.sigmoid(self.conv(score))
        return score.unsqueeze(1)

    def flops(self, channels):
        return channels * self.conv.kernel_size[0]


class DDEMResidualBlock(nn.Module):
    """Lightweight feature block used by the degradation estimator."""

    def __init__(self, dim):
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pointwise1 = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.pointwise2 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.depthwise(x))
        x = self.act(self.pointwise1(x))
        return residual + self.pointwise2(x)


class DualDegradationEstimationModule(nn.Module):
    """MODEM-style global descriptor Z0 and adaptive kernel Z1.

    The feature extractor supports legacy convolutional blocks, standard
    Uformer shifted-window blocks, and the proposed unconditioned local-global
    base blocks. The two prior heads follow MODEM: masked global pooling
    produces Z0, while two local projections produce a sample-dependent
    channel correlation matrix Z1.
    """

    def __init__(self, in_channels=3, prior_dim=32, num_blocks=2,
                 backbone='local_global', num_heads=1, win_size=8):
        super().__init__()
        if backbone not in ('conv', 'transformer', 'local_global'):
            raise ValueError(
                "DDEM backbone must be 'conv', 'transformer' or "
                f"'local_global', got {backbone}")
        if num_blocks < 1:
            raise ValueError("DDEM must contain at least one feature block")
        if backbone != 'conv' and prior_dim % num_heads != 0:
            raise ValueError(
                f"DDEM prior_dim ({prior_dim}) must be divisible by "
                f"num_heads ({num_heads})")
        self.in_channels = in_channels
        self.prior_dim = prior_dim
        self.backbone = backbone
        self.num_heads = num_heads
        self.win_size = win_size
        # Two overlapping stride-2 projections serve as a 1/4-scale patch
        # embedding. Spatial modelling after this point is performed by the
        # selected DDEM backbone.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, prior_dim, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(prior_dim, prior_dim, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )
        if backbone == 'conv':
            self.blocks = nn.Sequential(*[
                DDEMResidualBlock(prior_dim) for _ in range(num_blocks)])
        else:
            # LeWinTransformerBlock is defined later in this module but is
            # available when Uformer/DDEM instances are constructed. Alternating
            # W-MSA and SW-MSA mirrors Uformer's own stage structure. The
            # nominal resolution only prevents init-time window shrinking;
            # actual rectangular H/W are supplied on every forward call.
            nominal_resolution = (win_size * 2, win_size * 2)
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(
                    dim=prior_dim,
                    input_resolution=nominal_resolution,
                    num_heads=num_heads,
                    win_size=win_size,
                    shift_size=0 if index % 2 == 0 else win_size // 2,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    token_projection='linear',
                    token_mlp='leff',
                    modulator=True,
                    use_local_branch=backbone == 'local_global',
                    double_process=False,
                    use_channel_aware=backbone == 'local_global')
                for index in range(num_blocks)
            ])

        hidden_dim = prior_dim * 4
        self.global_norm = nn.LayerNorm(prior_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.descriptor_norm = nn.LayerNorm(hidden_dim)
        self.z0_head = nn.Sequential(
            nn.Linear(hidden_dim, prior_dim),
            nn.SiLU(inplace=True),
        )
        self.z1_phi = nn.Conv2d(prior_dim, prior_dim, 3, padding=1, bias=False)
        self.z1_psi = nn.Conv2d(prior_dim, prior_dim, 3, padding=1, bias=False)

    @staticmethod
    def _feature_mask(valid_mask, feature):
        if valid_mask is None:
            return feature.new_ones(
                feature.shape[0], 1, feature.shape[2], feature.shape[3])
        return F.interpolate(
            valid_mask.to(dtype=feature.dtype),
            size=feature.shape[-2:], mode='area').clamp_(0, 1)

    def forward(self, x, valid_mask=None):
        # MODEM computes its degradation representation in FP32. Uformer is
        # trained under an outer AMP autocast context, so explicitly disable
        # autocast for the complete DDEM path. In particular, the Z1 channel
        # correlation accumulates thousands of spatial products; performing
        # that reduction in FP16 can overflow before valid_count is applied.
        with torch.cuda.amp.autocast(enabled=False):
            feature = self.stem(x.float())
            if self.backbone == 'conv':
                feature = self.blocks(feature)
            else:
                height, width = feature.shape[-2:]
                if height % self.win_size != 0 or width % self.win_size != 0:
                    raise ValueError(
                        f"DDEM feature size {(height, width)} must be divisible "
                        f"by its window size {self.win_size}")
                tokens = feature.flatten(2).transpose(1, 2).contiguous()
                for block in self.blocks:
                    tokens = block(tokens, x_size=(height, width))
                feature = tokens.transpose(1, 2).reshape(
                    x.shape[0], self.prior_dim, height, width).contiguous()
            feature_mask = self._feature_mask(valid_mask, feature)
            valid_count = feature_mask.sum(dim=(2, 3)).clamp_min(1.0)

            pooled = ((feature * feature_mask).sum(dim=(2, 3)) /
                      valid_count)
            z_tilde = self.descriptor_norm(
                self.global_mlp(self.global_norm(pooled)))
            z0 = self.z0_head(z_tilde)

            phi = self.z1_phi(feature).flatten(2)
            psi = self.z1_psi(feature).flatten(2)
            # sqrt(mask) on both factors yields one mask weight in the product
            # and supports fractional area masks after spatial downsampling.
            spatial_weight = feature_mask.flatten(2).sqrt()
            z1 = ((phi * spatial_weight) @
                  (psi * spatial_weight).transpose(1, 2))
            z1 = z1 / valid_count.unsqueeze(-1)

        result = {
            'z_tilde': z_tilde,
            'z0': z0,
            'z1': z1,
        }
        if hasattr(self, 'response_head'):
            with torch.cuda.amp.autocast(enabled=False):
                result['kernel_response'] = self.response_head(feature.float())
        return result


class Z0StageAdapter(nn.Module):
    """V2: preserve Z0 direction/magnitude without a static bias shortcut."""

    def __init__(self, prior_dim, feature_dim, modulation_limit=0.1):
        super().__init__()
        self.prior_dim = prior_dim
        self.feature_dim = feature_dim
        self.modulation_limit = modulation_limit
        self.norm = nn.LayerNorm(prior_dim, elementwise_affine=False)
        self.affine = nn.Linear(
            prior_dim + 1, feature_dim * 2, bias=False)
        self.guidance_enabled = True
        self.reset_v2_parameters()

    def reset_v2_parameters(self):
        # A small non-zero initialization lets gradients reach Z0/DDEM from
        # the first update while keeping the pretrained backbone nearly intact.
        nn.init.normal_(self.affine.weight, mean=0.0, std=1e-3)

    def disable_guidance(self):
        self.guidance_enabled = False

    def modulation(self, z0):
        if z0.shape[-1] != self.prior_dim:
            raise ValueError(
                f"Z0 dim {z0.shape[-1]} does not match adapter prior dim "
                f"{self.prior_dim}")
        with torch.cuda.amp.autocast(enabled=False):
            z0_fp32 = z0.float()
            z_direction = self.norm(z0_fp32)
            z_magnitude = torch.log(
                z0_fp32.norm(dim=-1, keepdim=True) + 1e-6)
            condition = torch.cat(
                [z_direction, z_magnitude], dim=-1)
            gamma, beta = self.affine(condition).chunk(2, dim=-1)
            gamma = self.modulation_limit * torch.tanh(gamma)
            beta = self.modulation_limit * torch.tanh(beta)
        return gamma, beta

    def forward(self, feature, z0):
        if not self.guidance_enabled:
            zeros = feature.new_zeros(
                feature.shape[0], self.feature_dim)
            return feature, zeros, zeros
        gamma, beta = self.modulation(z0)
        gamma = gamma.to(dtype=feature.dtype)
        beta = beta.to(dtype=feature.dtype)
        guided = feature * (1.0 + gamma[:, None, None, :])
        guided = guided + beta[:, None, None, :]
        return guided, gamma, beta


class Z1StageAdapter(nn.Module):
    """Project Z1 to a stage-wise channel transform for local features.

    A stage-specific projection P maps the DDEM relation matrix from
    ``prior_dim`` to the current block width ``feature_dim``:

        logits_l = P_l Z1 P_l^T

    The normalized row-stochastic matrix is then used by the guided process
    to transform the complete Local Branch feature before K/V are generated.
    """

    def __init__(self, prior_dim, feature_dim, alpha_max=0.2,
                 alpha_init=0.01):
        super().__init__()
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError(
                f"alpha_init must be in (0, {alpha_max}), got {alpha_init}")
        self.prior_dim = prior_dim
        self.feature_dim = feature_dim
        self.alpha_max = alpha_max
        self.alpha_init = alpha_init
        self.projection = nn.Linear(
            prior_dim, feature_dim, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self.guidance_enabled = True
        self.reset_v2_parameters()

    def reset_v2_parameters(self):
        with torch.no_grad():
            # Orthogonal columns provide a non-collapsed, deterministic-scale
            # starting map for both square and expanded stage dimensions.
            nn.init.orthogonal_(self.projection.weight)
            self.log_tau.zero_()
            ratio = self.alpha_init / self.alpha_max
            self.alpha_raw.fill_(math.log(ratio / (1.0 - ratio)))

    def disable_guidance(self):
        self.guidance_enabled = False

    def forward(self, z1):
        if z1.shape[-2:] != (self.prior_dim, self.prior_dim):
            raise ValueError(
                f"Z1 shape {tuple(z1.shape[-2:])} does not match adapter "
                f"prior shape {(self.prior_dim, self.prior_dim)}")
        # Keep the complete adapter in FP32. Normalize both before and after
        # projection so neither a small raw Z1 scale nor the stage projection
        # can collapse the softmax to a sample-independent uniform matrix.
        with torch.cuda.amp.autocast(enabled=False):
            normalized = z1.float()
            normalized = normalized - normalized.mean(
                dim=-1, keepdim=True)
            rms = normalized.pow(2).mean(
                dim=(-2, -1), keepdim=True).sqrt()
            normalized = normalized / (rms + 1e-6)

            projection = self.projection.weight.float()
            projected = torch.einsum(
                'cp,bpq,dq->bcd',
                projection, normalized, projection)
            projected = projected - projected.mean(
                dim=-1, keepdim=True)
            projected_rms = projected.pow(2).mean(
                dim=(-2, -1), keepdim=True).sqrt()
            projected = projected / (projected_rms + 1e-6)

            temperature = self.log_tau.float().exp().clamp(0.05, 5.0)
            matrix = torch.softmax(
                projected / temperature, dim=-1)
            if self.guidance_enabled:
                strength = self.alpha_max * torch.sigmoid(
                    self.alpha_raw.float())
            else:
                strength = self.alpha_raw.new_zeros(1).float()
        return matrix, strength
        
######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    
    
    def flops(self, q_L, kv_L=None): 
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            # A 2-D tensor is the original global modulator token table; a
            # 3-D tensor is per-window K/V supplied by OmniLocalBranch.
            if attn_kv.dim() == 2:
                attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
            elif attn_kv.dim() == 3:
                if attn_kv.size(0) != B_:
                    raise ValueError(
                        f"attn_kv batch ({attn_kv.size(0)}) must match x batch ({B_})"
                    )
            else:
                raise ValueError("attn_kv must have shape (N, C) or (B, N, C)")
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

    def flops(self, q_L, kv_L=None): 
        kv_L = kv_L or q_L
        flops = q_L*self.dim*self.inner_dim+kv_L*self.dim*self.inner_dim*2
        return flops 


#########################################
########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 degradation_prior_dim=None):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.last_prior_stats = None

    def forward(self, x, attn_kv=None, mask=None,
                degradation_prior=None, collect_prior_stats=False,
                q_reference=None, z0_stats=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0]*self.win_size[1]
        nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H*W, H*W)
        
        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}"%(flops/1e9))
        return flops


########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
            
        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        
        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}"%(flops/1e9))
        return flops


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.in_features*self.hidden_features 
        # fc2
        flops += H*W*self.hidden_features*self.out_features
        print("MLP:{%.2f}"%(flops/1e9))
        return flops


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x, H=None, W=None):
        # bs x hw x c
        bs, hw, c = x.size()
        H = H or int(math.sqrt(hw))
        W = W or int(math.sqrt(hw))
        if H * W != hw:
            raise ValueError(f"LeFF spatial size {(H, W)} does not match token count {hw}")

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)

        x = self.linear2(x)
        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim 
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        # eca 
        if hasattr(self.eca, 'flops'): 
            flops += self.eca.flops()
        return flops


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        H = H or int(math.sqrt(L))
        W = W or int(math.sqrt(L))
        if H * W != L:
            raise ValueError(f"Downsample spatial size {(H, W)} does not match token count {L}")
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        H = H or int(math.sqrt(L))
        W = W or int(math.sqrt(L))
        if H * W != L:
            raise ValueError(f"Upsample spatial size {(H, W)} does not match token count {L}")
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        H = H or int(math.sqrt(L))
        W = W or int(math.sqrt(L))
        if H * W != L:
            raise ValueError(f"OutputProj spatial size {(H, W)} does not match token count {L}")
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops

#########################################
########### LeWinTransformer #############
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                 modulator=False,cross_modulator=False, use_local_branch=False,
                 double_process=False, local_expansion=2,
                 use_channel_aware=False, degradation_prior_dim=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        self.use_local_branch = use_local_branch
        self.double_process = double_process
        self.local_expansion = local_expansion
        self.use_channel_aware = use_channel_aware
        self.degradation_prior_dim = degradation_prior_dim
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size*win_size, dim) # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size*win_size, dim) # cross_modulator
            self.cross_attn = Attention(dim,num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    token_projection=token_projection,)
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        def build_mlp():
            if token_mlp in ['ffn', 'mlp']:
                return Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)
            if token_mlp == 'leff':
                return LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
            if token_mlp == 'fastleff':
                return FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
            raise Exception("FFN error!")

        self.mlp = build_mlp()
        self.local_branch = (
            OmniLocalBranch(dim, expansion=local_expansion)
            if use_local_branch else None
        )
        self.channel_aware = (
            ChannelAwareModule() if use_local_branch and use_channel_aware else None
        )

        # The second process has its own parameters.  Keeping it unconstructed
        # when disabled preserves the original Uformer state_dict exactly.
        if double_process:
            self.norm1_second = norm_layer(dim)
            self.attn_second = WindowAttention(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, token_projection=token_projection)
            self.z0_adapter_second = (
                Z0StageAdapter(degradation_prior_dim, dim)
                if degradation_prior_dim is not None else None)
            self.z1_adapter_second = (
                Z1StageAdapter(degradation_prior_dim, dim)
                if degradation_prior_dim is not None and use_local_branch
                else None)
            self.norm2_second = norm_layer(dim)
            self.mlp_second = build_mlp()
            self.local_branch_second = (
                OmniLocalBranch(dim, expansion=local_expansion)
                if use_local_branch else None
            )
            self.channel_aware_second = (
                ChannelAwareModule()
                if use_local_branch and use_channel_aware else None
            )


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}," \
               f"modulator={self.modulator}, use_local_branch={self.use_local_branch}, " \
               f"double_process={self.double_process}, use_channel_aware={self.use_channel_aware}"

    def _forward_process(self, x, norm1, attn, norm2, mlp, local_branch,
                         channel_aware, z0_adapter, z1_adapter,
                         attn_mask, H, W, degradation_prior=None,
                         collect_prior_stats=False):
        """Run one attention/FFN process with optional feature-level priors.

        In the guided stage-last second process, Z0 conditions the complete
        normalized block feature used by Q, Local Branch and CAM. Z1 is first
        projected to the current stage width and then transforms the complete
        Local Branch output before that feature constructs K/V and enters the
        direct local residual path.
        """
        B, _, C = x.shape
        shortcut = x
        normalized = norm1(x).view(B, H, W, C)
        conditioned = normalized
        gamma = beta = None
        if degradation_prior is not None and z0_adapter is not None:
            conditioned, gamma, beta = z0_adapter(
                normalized, degradation_prior['z0'])

        if self.shift_size > 0:
            shifted_reference = torch.roll(
                normalized,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            shifted_feature = torch.roll(
                conditioned,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
        else:
            shifted_reference = normalized
            shifted_feature = conditioned

        local_shifted_x = (
            local_branch(shifted_feature)
            if local_branch is not None else None)
        local_reference = None
        if (collect_prior_stats and degradation_prior is not None and
                local_branch is not None):
            local_reference = local_branch(shifted_reference)

        z1_matrix = z1_strength = None
        if z1_adapter is not None and hasattr(self, 'response_local_second'):
            if degradation_prior is None or 'kernel_response' not in degradation_prior:
                raise ValueError('Kernel response control requires a predicted response')
            collect_response_stats = 'response_gate_stats' in degradation_prior
            controlled = self.response_local_second(
                local_shifted_x, degradation_prior['kernel_response'], self.shift_size,
                return_stats=collect_response_stats)
            if collect_response_stats:
                local_shifted_x, gate_stats = controlled
                degradation_prior['response_gate_stats'][self.response_stage_name] = gate_stats
            else:
                local_shifted_x = controlled
        if degradation_prior is not None and z1_adapter is not None:
            if local_shifted_x is None:
                raise RuntimeError(
                    "Z1 local-feature guidance requires Local Branch")
            z1_matrix, z1_strength = z1_adapter(
                degradation_prior['z1'])
            local_dynamic = torch.einsum(
                'bhwc,bcd->bhwd',
                local_shifted_x,
                z1_matrix.to(dtype=local_shifted_x.dtype))
            # Identity-centered stage-specific local feature transformation:
            #   Ld = L + alpha * (L M_l - L)
            #      = (1-alpha)L + alpha L M_l.
            local_shifted_x = local_shifted_x + z1_strength.to(
                dtype=local_shifted_x.dtype) * (
                    local_dynamic - local_shifted_x)

        x_windows = window_partition(shifted_feature, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)
        reference_windows = window_partition(
            shifted_reference, self.win_size)
        reference_windows = reference_windows.view(
            -1, self.win_size * self.win_size, C)

        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
            reference_windows = self.with_pos_embed(
                reference_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows

        local_windows = None
        local_reference_windows = None
        if local_shifted_x is not None:
            # The degradation-adaptive local feature is the common K/V source.
            local_windows = window_partition(
                local_shifted_x, self.win_size).view(
                    -1, self.win_size * self.win_size, C)
            if local_reference is not None:
                local_reference_windows = window_partition(
                    local_reference, self.win_size).view(
                        -1, self.win_size * self.win_size, C)

        # Q, Local K/V and CAM now share the Z0-conditioned block feature;
        # Z1 has already adapted the complete local K/V source above.
        attn_windows = attn(
            wmsa_in, attn_kv=local_windows, mask=attn_mask)

        if collect_prior_stats:
            attn.last_prior_stats = None
            if degradation_prior is not None:
                with torch.no_grad():
                    eps = 1e-8
                    q_after, k_after, v_after = attn.qkv(
                        wmsa_in, local_windows)
                    q_before, k_before, v_before = attn.qkv(
                        reference_windows, local_reference_windows)
                    stats = {
                        'q_relative_change': float(
                            (q_after - q_before).norm() /
                            (q_before.norm() + eps)),
                        'k_relative_change': float(
                            (k_after - k_before).norm() /
                            (k_before.norm() + eps)),
                        'v_relative_change': float(
                            (v_after - v_before).norm() /
                            (v_before.norm() + eps)),
                    }
                    if gamma is not None:
                        stats.update({
                            'z0_gamma_mean': float(gamma.mean()),
                            'z0_gamma_std': float(
                                gamma.std(unbiased=False)),
                            'z0_beta_mean': float(beta.mean()),
                            'z0_beta_std': float(
                                beta.std(unbiased=False)),
                        })
                    if z1_matrix is not None:
                        stats.update({
                            'local_relative_change': float(
                                (local_shifted_x - local_reference).norm() /
                                (local_reference.norm() + eps)),
                            'z1_strength': float(z1_strength),
                            'local_matrix_mean': float(z1_matrix.mean()),
                            'local_matrix_std': float(
                                z1_matrix.std(unbiased=False)),
                            'local_matrix_entropy': float(
                                (-(z1_matrix.clamp_min(1e-8) *
                                   z1_matrix.clamp_min(1e-8).log()).sum(
                                       dim=-1) /
                                 math.log(float(z1_matrix.shape[-1]))).mean()),
                            'local_matrix_max': float(
                                z1_matrix.max(dim=-1).values.mean()),
                        })
                    attn.last_prior_stats = stats
        else:
            attn.last_prior_stats = None

        attn_windows = attn_windows.view(
            -1, self.win_size, self.win_size, C)
        shifted_out = window_reverse(
            attn_windows, self.win_size, H, W)

        if self.shift_size > 0:
            attn_out = torch.roll(
                shifted_out,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
            if local_shifted_x is not None:
                local_out = torch.roll(
                    local_shifted_x,
                    shifts=(self.shift_size, self.shift_size),
                    dims=(1, 2))
        else:
            attn_out = shifted_out
            if local_shifted_x is not None:
                local_out = local_shifted_x

        # DiNAT-IR fusion: channel-aware scores modulate the completed
        # attention update, not the K/V source or the local residual.
        if channel_aware is not None:
            attn_out = attn_out * channel_aware(conditioned)
        attn_out = attn_out.view(B, H * W, C)
        x = shortcut + self.drop_path(attn_out)
        if local_shifted_x is not None:
            # Match Omni-Deblurring: local information participates both in
            # K/V construction and as a residual feature.
            x = x + local_out.reshape(B, H * W, C)
        x = x + self.drop_path(mlp(norm2(x), H, W))
        return x

    def forward(self, x, mask=None, x_size=None, degradation_prior=None,
                collect_prior_stats=False):
        B, L, C = x.shape
        if x_size is None:
            H = int(math.sqrt(L))
            W = H
        else:
            H, W = x_size
        if H * W != L:
            raise ValueError(f"LeWin block spatial size {(H, W)} does not match token count {L}")
        
        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask


        if self.cross_modulator is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross
    
        x = self._forward_process(
            x, self.norm1, self.attn, self.norm2, self.mlp,
            self.local_branch, self.channel_aware, None, None,
            attn_mask, H, W, None, False)
        if self.double_process:
            x = self._forward_process(
                x, self.norm1_second, self.attn_second,
                self.norm2_second, self.mlp_second,
                self.local_branch_second, self.channel_aware_second,
                self.z0_adapter_second, self.z1_adapter_second,
                attn_mask, H, W,
                degradation_prior, collect_prior_stats)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H*W, self.win_size*self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H,W)
        if self.local_branch is not None:
            flops += self.local_branch.flops(H, W)
        if self.channel_aware is not None:
            flops += H * W * self.dim + self.channel_aware.flops(self.dim)
        if self.double_process:
            # second norm1 + attention + norm2 + mlp
            flops += self.dim * H * W
            flops += self.attn_second.flops(H, W)
            flops += self.dim * H * W
            flops += self.mlp_second.flops(H, W)
            if self.local_branch_second is not None:
                flops += self.local_branch_second.flops(H, W)
            if self.channel_aware_second is not None:
                flops += H * W * self.dim + self.channel_aware_second.flops(self.dim)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


#########################################
########### Basic layer of Uformer ################
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn', shift_flag=True,
                 modulator=False,cross_modulator=False, use_local_branch=False,
                 double_process=False, local_expansion=2,
                 use_channel_aware=False, degradation_prior_dim=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0 if (i % 2 == 0) else win_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    modulator=modulator,cross_modulator=cross_modulator,
                                    use_local_branch=use_local_branch,
                                    double_process=double_process and i == depth - 1,
                                    local_expansion=local_expansion,
                                    use_channel_aware=use_channel_aware,
                                    degradation_prior_dim=(
                                        degradation_prior_dim
                                        if double_process and i == depth - 1
                                        else None))
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    modulator=modulator,cross_modulator=cross_modulator,
                                    use_local_branch=use_local_branch,
                                    double_process=double_process and i == depth - 1,
                                    local_expansion=local_expansion,
                                    use_channel_aware=use_channel_aware,
                                    degradation_prior_dim=(
                                        degradation_prior_dim
                                        if double_process and i == depth - 1
                                        else None))
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"    

    def forward(self, x, mask=None, x_size=None, degradation_prior=None,
                collect_prior_stats=False):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    lambda inputs: blk(
                        inputs, mask=mask, x_size=x_size,
                        degradation_prior=degradation_prior,
                        collect_prior_stats=collect_prior_stats), x)
            else:
                x = blk(
                    x, mask, x_size,
                    degradation_prior=degradation_prior,
                    collect_prior_stats=collect_prior_stats)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class Uformer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True, modulator=False, 
                 cross_modulator=False, use_local_branch=False,
                 double_process=False, local_expansion=2,
                 use_channel_aware=False, use_ddem=False, ddem_stage=2,
                 ddem_dim=32, ddem_blocks=2, ddem_backbone='local_global',
                 ddem_heads=1, ddem_win_size=8, use_kernel_response=False,
                 kernel_response_groups=1, kernel_response_variant='serial', **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in
        self.use_local_branch = use_local_branch
        self.double_process = double_process
        self.local_expansion = local_expansion
        self.use_channel_aware = use_channel_aware
        self.use_ddem = use_ddem
        self.ddem_stage = ddem_stage
        self.ddem_dim = ddem_dim
        self.ddem_backbone = ddem_backbone
        self.use_kernel_response = use_kernel_response
        self.kernel_response_groups = kernel_response_groups
        self.kernel_response_variant = kernel_response_variant
        if kernel_response_variant not in ('serial', 'joint_residual'):
            raise ValueError('Unknown kernel_response_variant: ' + str(kernel_response_variant))
        if not use_kernel_response and kernel_response_variant != 'serial':
            raise ValueError('Non-serial response control requires use_kernel_response=True')
        if not isinstance(kernel_response_groups, int) or kernel_response_groups < 1:
            raise ValueError('kernel_response_groups must be a positive integer')
        if not use_kernel_response and kernel_response_groups != 1:
            raise ValueError('Grouped response control requires use_kernel_response=True')
        if use_kernel_response and not (use_ddem and use_local_branch and double_process and ddem_stage == 2):
            raise ValueError('Kernel response requires blur-only DDEM, local branch and double_process')
        if use_ddem and ddem_stage not in (1, 2):
            raise ValueError(f"ddem_stage must be 1 or 2, got {ddem_stage}")
        self.ddem = None
        if use_ddem:
            ddem_in_channels = dd_in * 2 if ddem_stage == 1 else dd_in
            self.ddem = DualDegradationEstimationModule(
                in_channels=ddem_in_channels,
                prior_dim=ddem_dim, num_blocks=ddem_blocks,
                backbone=ddem_backbone, num_heads=ddem_heads,
                win_size=ddem_win_size)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            use_local_branch=use_local_branch,double_process=False,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)

        # Decoder
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)
        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)
        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)
        self.upsample_3 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            modulator=modulator,cross_modulator=cross_modulator,
                            use_local_branch=use_local_branch,double_process=double_process,
                            local_expansion=local_expansion,use_channel_aware=use_channel_aware,
                            degradation_prior_dim=ddem_dim if use_ddem else None)

        self.apply(self._init_weights)
        if self.ddem is not None:
            # Uformer's global Linear initialization runs after submodule
            # construction, so restore the intended small V2 initialization.
            for module in self.modules():
                if isinstance(module, Z0StageAdapter):
                    module.reset_v2_parameters()
                elif isinstance(module, Z1StageAdapter):
                    module.reset_v2_parameters()

        if use_kernel_response:
            # Attach AFTER original initialization. Original parameters and RNG
            # state for them stay identical to a same-seed legacy construction.
            from kernel_response import (
                ResponseHead, ResponseConditionedLocal,
                JointResidualResponseConditionedLocal,
            )
            controller_type = (
                JointResidualResponseConditionedLocal
                if kernel_response_variant == 'joint_residual'
                else ResponseConditionedLocal
            )
            with torch.random.fork_rng(devices=[]):
                self.ddem.response_head = ResponseHead(ddem_dim)
                for stage_name in ('encoderlayer_0', 'encoderlayer_1',
                                   'encoderlayer_2', 'encoderlayer_3',
                                   'decoderlayer_0', 'decoderlayer_1',
                                   'decoderlayer_2', 'decoderlayer_3'):
                    stage = getattr(self, stage_name)
                    block = stage.blocks[-1]
                    block.response_stage_name = stage_name
                    block.response_local_second = controller_type(
                        block.dim, groups=kernel_response_groups)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, " \
               f"token_mlp={self.mlp}, win_size={self.win_size}, " \
               f"use_local_branch={self.use_local_branch}, use_channel_aware={self.use_channel_aware}, " \
               f"double_process={self.double_process}, use_ddem={self.use_ddem}, " \
               f"ddem_stage={self.ddem_stage}, ddem_backbone={self.ddem_backbone}"

    def forward(self, x, mask=None, gt=None, valid_mask=None,
                return_ddem_features=False, collect_prior_stats=False,
                collect_response_stats=False):
        if collect_response_stats and not (self.use_kernel_response and return_ddem_features):
            raise ValueError('Response statistics require kernel response and return_ddem_features')
        _, _, H, W = x.shape
        # Each stage must still be divisible by its effective window size.
        # Shallow test models may reduce win_size at deep stages, while the
        # production Uformer-B resolves to 8 * 2**4 = 128.
        required_multiple = max(
            self.encoderlayer_0.blocks[0].win_size,
            self.encoderlayer_1.blocks[0].win_size * 2,
            self.encoderlayer_2.blocks[0].win_size * 4,
            self.encoderlayer_3.blocks[0].win_size * 8,
            self.conv.blocks[0].win_size * 16,
        )
        if H % required_multiple != 0 or W % required_multiple != 0:
            raise ValueError(
                f"Uformer input size {(H, W)} must be divisible by "
                f"{required_multiple}; "
                "pad the image before calling the model")
        degradation_prior = None
        if self.ddem is not None:
            if self.ddem_stage == 1:
                if gt is None:
                    raise ValueError(
                        "DDEM stage 1 requires paired ground truth input")
                if gt.shape != x.shape:
                    raise ValueError(
                        f"DDEM input/GT shapes differ: {x.shape} vs {gt.shape}")
                ddem_input = torch.cat([x, gt], dim=1)
            else:
                ddem_input = x
            degradation_prior = self.ddem(ddem_input, valid_mask)
            if collect_response_stats:
                # This dictionary belongs to this forward/replica, not shared
                # module state. Only small detached per-sample tensors return.
                degradation_prior['response_gate_stats'] = {}
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        #Encoder
        layer_kwargs = {
            'mask': mask,
            'degradation_prior': degradation_prior,
            'collect_prior_stats': collect_prior_stats,
        }
        conv0 = self.encoderlayer_0(
            y, x_size=(H, W), **layer_kwargs)
        pool0 = self.dowsample_0(conv0, H, W)
        H1, W1 = H // 2, W // 2
        conv1 = self.encoderlayer_1(
            pool0, x_size=(H1, W1), **layer_kwargs)
        pool1 = self.dowsample_1(conv1, H1, W1)
        H2, W2 = H1 // 2, W1 // 2
        conv2 = self.encoderlayer_2(
            pool1, x_size=(H2, W2), **layer_kwargs)
        pool2 = self.dowsample_2(conv2, H2, W2)
        H3, W3 = H2 // 2, W2 // 2
        conv3 = self.encoderlayer_3(
            pool2, x_size=(H3, W3), **layer_kwargs)
        pool3 = self.dowsample_3(conv3, H3, W3)
        H4, W4 = H3 // 2, W3 // 2

        # Bottleneck
        conv4 = self.conv(pool3, x_size=(H4, W4), **layer_kwargs)

        #Decoder
        up0 = self.upsample_0(conv4, H4, W4)
        deconv0 = torch.cat([up0,conv3],-1)
        deconv0 = self.decoderlayer_0(
            deconv0, x_size=(H3, W3), **layer_kwargs)
        
        up1 = self.upsample_1(deconv0, H3, W3)
        deconv1 = torch.cat([up1,conv2],-1)
        deconv1 = self.decoderlayer_1(
            deconv1, x_size=(H2, W2), **layer_kwargs)

        up2 = self.upsample_2(deconv1, H2, W2)
        deconv2 = torch.cat([up2,conv1],-1)
        deconv2 = self.decoderlayer_2(
            deconv2, x_size=(H1, W1), **layer_kwargs)

        up3 = self.upsample_3(deconv2, H1, W1)
        deconv3 = torch.cat([up3,conv0],-1)
        deconv3 = self.decoderlayer_3(
            deconv3, x_size=(H, W), **layer_kwargs)

        # Output Projection
        y = self.output_proj(deconv3, H, W)
        restored = x + y if self.dd_in ==3 else y
        if return_ddem_features:
            if degradation_prior is None:
                raise RuntimeError(
                    "return_ddem_features requires use_ddem=True")
            return restored, degradation_prior
        return restored

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso,self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops()+self.dowsample_0.flops(self.reso,self.reso)
        flops += self.encoderlayer_1.flops()+self.dowsample_1.flops(self.reso//2,self.reso//2)
        flops += self.encoderlayer_2.flops()+self.dowsample_2.flops(self.reso//2**2,self.reso//2**2)
        flops += self.encoderlayer_3.flops()+self.dowsample_3.flops(self.reso//2**3,self.reso//2**3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso//2**4,self.reso//2**4)+self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso//2**3,self.reso//2**3)+self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso//2**2,self.reso//2**2)+self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso//2,self.reso//2)+self.decoderlayer_3.flops()
        
        # Output Projection
        flops += self.output_proj.flops(self.reso,self.reso)
        return flops


if __name__ == "__main__":
    input_size = 256
    arch = Uformer
    depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    model_restoration = Uformer(img_size=input_size, embed_dim=16,depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
    print(model_restoration)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model_restoration.parameters())/ 1e6))
    print("number of GFLOPs: %.2f G"%(model_restoration.flops() / 1e9))

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

CACHE_T = 2
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x

def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)

def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


def block_causal_mask(x, block_size):
    # params
    b, n, s, _, device = *x.size(), x.device
    assert s % block_size == 0
    num_blocks = s // block_size

    # build mask
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :,
             i * block_size:(i + 1) * block_size, :(i + 1) * block_size] = 1
    return mask


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim,
                                          dim * 2, (3, 1, 1),
                                          padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad3d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad3d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim,
                                          dim, (3, 1, 1),
                                          stride=(2, 1, 1),
                                          padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            #attn_mask=block_causal_mask(q, block_size=h * w)
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128, #96
                 z_dim=4, # 32
                 dim_mult=[1, 2, 4, 4],#[1,2,4,4]
                 num_res_blocks=2,#2
                 attn_scales=[], #[]
                 temperal_downsample=[True, True, False], #[False,True,True]
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        # import ipdb;ipdb.set_trace()
        # dimensions
        dims = [dim * u for u in [1] + dim_mult] #[96,96,192,384,384]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if check_is_instance(m, CausalConv3d):
            count += 1
    return count


class VideoVAE_(nn.Module):

    def __init__(self,
                 dim=96,
                 z_dim=16,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim #96
        self.z_dim = z_dim #16
        self.dim_mult = dim_mult #[1,2,4,4]
        self.num_res_blocks = num_res_blocks #2
        self.attn_scales = attn_scales #[]
        self.temperal_downsample = temperal_downsample # [False,True,True]
        self.temperal_upsample = temperal_downsample[::-1]# [True,True,False]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale, return_var=False):
        # import ipdb;ipdb.set_trace() # x.shape 1,3,45,272,272
        self.clear_cache()
        ## cache
        t = x.shape[2] #45
        iter_ = 1 + (t - 1) // 4

        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :],
                                   feat_cache=self._enc_feat_map,
                                   feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :], #1,32,1,34,34 ！！看这一块怎么降采样的！！
                                    feat_cache=self._enc_feat_map,
                                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2) # 1,32,2,34,34  ->>1,32,12,34,34
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        
        if return_var:
            return mu, log_var
        else:
            return mu # 1,16,12,34,34

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :],
                                   feat_cache=self._feat_map,
                                   feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2) # may add tensor offload
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


class WanVideoVAE(nn.Module):

    def __init__(self, z_dim=16):
        super().__init__()

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE_(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8


    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x


    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask


    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values


    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        # import ipdb;ipdb.set_trace()
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device #"cuda"

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        # import ipdb;ipdb.set_trace()
        for h, h_, w, w_ in tasks:
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)

            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values


    def single_encode(self, video, device, return_var=False):
        # video = video.to(device)
        x = self.model.encode(video, self.scale, return_var=return_var)
        return x


    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)


    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # import ipdb;ipdb.set_trace()
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled: #true
                tile_size = (tile_size[0] * 8, tile_size[1] * 8) #272,272
                tile_stride = (tile_stride[0] * 8, tile_stride[1] * 8) #144,128
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                # import ipdb;ipdb.set_trace()
                hidden_state = self.single_encode(video.to('cuda'), device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states


    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos


    @staticmethod
    def state_dict_converter():
        return WanVideoVAEStateDictConverter()


class WanVideoVAEStateDictConverter:

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        state_dict_ = {}
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        for name in state_dict:
            state_dict_['model.' + name] = state_dict[name]
        return state_dict_

class WanVAELatentLoss(nn.Module):
    def __init__(self, 
                 kl_weight=0.1,
                 recon_type='l2',
                 feature_weight=0.5,
                 wanvae_mean=None,
                 wanvae_std=None,
                 device="cuda"):
        """
        Args:
            kl_weight: KL散度权重系数
            recon_type: 重建损失类型 ('l1', 'l2', 'smooth_l1')
            feature_weight: 特征相似度权重
            wanvae_mean: WanVAE潜在空间的均值（从您的代码中提取）
            wanvae_std: WanVAE潜在空间的标准差
        """
        super().__init__()
        self.device = device
        self.kl_weight = kl_weight
        self.recon_type = recon_type
        self.feature_weight = feature_weight

        
        # 注册WanVAE的统计量为buffer
        self.register_buffer('wanvae_mean', torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ], dtype=torch.float).to(self.device))
        
        self.register_buffer('wanvae_std', torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ], dtype=torch.float).to(self.device))

    def forward(self, upsample_lents, target_latents):
        """
        Args:
            upsample_lents: 上采样后的潜在变量 (B,D,T,H,W)
            target_latents: WanVAE生成的参考潜在变量 (B,D,T,H,W)
        Returns:
            dict: 包含各项损失的字典
        """
        # 1. 潜在空间重建损失
        # import ipdb;ipdb.set_trace()
        recon_loss = self._compute_recon_loss(upsample_lents, target_latents)
        
        # 2. KL散度（针对WanVAE潜在空间特性调整）
        # kl_loss = self._wanvae_kl_divergence(upsample_lents)
        
        # 3. 特征分布匹配损失
        # dist_loss = self._distribution_match_loss(upsample_lents
        
        # total_loss = recon_loss + self.kl_weight * kl_loss + \
        #             self.feature_weight * dist_loss
        total_loss = recon_loss
        
        return {
            'total_loss': total_loss,
            # 'recon_loss': recon_loss,
            # 'kl_loss': kl_loss,
            # 'dist_loss': dist_loss
        }
    

    def _compute_recon_loss(self, pred, target):
        """计算重建损失"""
        if self.recon_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.recon_type == 'l2':
            return F.mse_loss(pred, target)
        else:  # smooth_l1
            return F.smooth_l1_loss(pred, target)

    def _wanvae_kl_divergence(self, z):
        """针对WanVAE潜在空间特性的KL散度计算"""
        # 计算每个潜在维度的KL散度
        # import ipdb;ipdb.set_trace()
        var = self.wanvae_std.pow(2).to(self.device)
        kl_per_dim = 0.5 * (
            (z - self.wanvae_mean[None,:,None,None,None]).pow(2) / var[None,:,None,None,None] +
            torch.log(var[None,:,None,None,None]) - 1
        )
        return kl_per_dim.mean()

    def _distribution_match_loss(self, z):
        """匹配WanVAE潜在空间分布特性"""
        # 计算每个维度的统计量差异
        z_mean = z.mean(dim=[0,2,3,4])  # 沿batch和空间维度平均
        z_std = z.std(dim=[0,2,3,4])
        
        mean_loss = F.mse_loss(z_mean, self.wanvae_mean.to(self.device))
        std_loss = F.mse_loss(z_std, self.wanvae_std.to(self.device))
        
        return 0.5 * (mean_loss + std_loss)

    @staticmethod
    def _spatial_temporal_smoothness(z):
        """时空平滑性约束"""
        # 时间维度平滑
        temp_diff = F.l1_loss(z[:,:,1:], z[:,:,:-1])
        
        # 空间维度平滑
        spatial_grad = torch.abs(z[:,:,:,:,1:] - z[:,:,:,:,:-1]) + \
                      torch.abs(z[:,:,:,1:,:] - z[:,:,:,:-1,:])
        spatial_loss = spatial_grad.mean()
        
        return 0.3 * temp_diff + 0.2 * spatial_loss


class ResBlock3D(nn.Module):
    """3D残差块（显著增加参数量）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + residual

    

class EnhancedResidualUpsample3d_v2(nn.Module):
    """
    残差上采样网络
    输入: (B, 16, T/16, H/64, W/64)
    输出: (B, 16, T/4, H/8, W/8)
    """
    
    def __init__(self, dim=16, hidden_dim=96):
        super().__init__()
        # for decoder2
        # 输入投影层 - 扩展通道数
        self.input_proj = nn.Conv3d(dim, hidden_dim, 1)
        
        # 第一个上采样块
        self.up1 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1),
            nn.ConvTranspose3d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=(1, 4, 4),    # 时间轴核大小=1（不扩展）
                stride=(1, 2, 2),         # 时间轴步长=1（不变）
                padding=(0, 1, 1)         # 时间轴填充=0
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        # 中间处理层
        self.middle1 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        self.middle2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        # 第二个上采样块
        self.up2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        # 第三个上采样块
        self.up3 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=(0,1,1)),
            # nn.ConvTranspose3d(
            #     in_channels=hidden_dim,
            #     out_channels=hidden_dim,
            #     kernel_size=(1, 4, 4),    # 时间轴核大小=1（不扩展）
            #     stride=(1, 2, 2),         # 时间轴步长=1（不变）
            #     padding=(0, 1, 1)         # 时间轴填充=0
            # ),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        # 输出投影层 - 恢复通道数
        self.output_proj = nn.Conv3d(hidden_dim, dim, 1)
        
        # 跳跃连接的投影层
        # self.skip_proj = nn.ConvTranspose3d(dim, dim, 8, stride=(4, 8, 8), padding=2)
        self.skip_proj = nn.ConvTranspose3d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(6, 12, 12),  # 核大小 = stride + 2*padding
            stride=(4, 8, 8),
            padding=(1, 2, 2)         # 各维度填充
        )
        
        # 中间层的残差连接
        self.middle_skip1 = nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.middle_skip2 = nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1)
        
        # for decoder2
        # self.skip_proj_d2 = nn.ConvTranspose3d(hidden_dim, dim, 4, stride=(4, 4, 4), padding=1)
        self.skip_proj_d2 = nn.ConvTranspose3d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=(6, 8, 8),  # 核大小 = stride + 2*padding
            stride=(4, 4, 4),
            padding=(1, 2, 2)         # 各维度填充
        )
        #第一次上采样
        self.up1_d2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        # 中间处理层
        self.middle1_d2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        #残差链接
        self.middle_skip1_d2 = nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1)
        
        #第二次上采样
        self.up2_d2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        # 中间处理层
        self.middle2_d2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        #残差链接
        self.middle_skip2_d2 = nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1)
        self.output_proj_d2 = nn.Conv3d(hidden_dim, dim, 1)

        
        
        # for decoder3
        self.skip_proj_d3 = nn.ConvTranspose3d(hidden_dim, dim, 4, stride=(2, 2, 2), padding=1)
        # 上采样块
        self.up_d3 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        # 中间处理层
        self.middle1_d3 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        #残差链接
        self.middle_skip1_d3 = nn.ConvTranspose3d(hidden_dim, hidden_dim, 4, stride=(2, 2, 2), padding=1)
        self.output_proj_d3 = nn.Conv3d(hidden_dim, dim, 1)
        
    def decoder1(self,x):
        # 保存输入用于跳跃连接
        skip = self.skip_proj(x)  # (B, 16, T/4, H/8, W/8) (8,16,12,56,104)
        
        # 主路径
        x = self.input_proj(x)  # (B, 64, T/16, H/64, W/64)
        
        # 第一次上采样
        x1 = self.up1(x)  # (B, 64, T/16, H/16, W/16) (8,64,3,14,26)
        
        # 中间处理 + 残差连接
        middle_skip1 = self.middle_skip1(x)  # (B, 64, T/8, H/16, W/16)
        x1_ = self.middle1(x1) + middle_skip1  # (B, 64, T/8, H/16, W/16)
        
        # 第二次上采样
        x2 = self.up2(x1_)  # (B, 64, T/4, H/8, W/8) (8,64,12,28,52)

        # 中间处理 + 残差连接
        middle_skip2 = self.middle_skip2(x1_)  # (B, 64, T/8, H/16, W/16)
        x2_ = self.middle1(x2) + middle_skip2  # (B, 64, T/8, H/16, W/16)
        
        # 第二次上采样
        x3 = self.up3(x2_)  # (B, 64, T/4, H/8, W/8) (8,64,13,56,104)
        # 输出投影
        # import ipdb;ipdb.set_trace()
        output = self.output_proj(x3)  # (B, 16, T/4, H/8, W/8)

        
        # 最终残差连接
        output = output + skip
        return output
    def decoder2(self,x):
        # import ipdb;ipdb.set_trace()
        skip_d2 = self.skip_proj_d2(x)
        # 第一次上采样
        x1_d2 = self.up1_d2(x)  # (B, 64, T/16, H/16, W/16) (8,64,3,14,26)
        # 中间处理 + 残差连接
        middle_skip1_d2 = self.middle_skip1_d2(x)  # (B, 64, T/8, H/16, W/16)
        x1_d2_ = self.middle1_d2(x1_d2) + middle_skip1_d2  # (B, 64, T/8, H/16, W/16)
        
        # 第二次上采样
        x2_d2 = self.up2_d2(x1_d2_)  # (B, 64, T/16, H/16, W/16) (8,64,3,14,26)
        # 中间处理 + 残差连接
        middle_skip2_d2 = self.middle_skip2_d2(x1_d2_)  # (B, 64, T/8, H/16, W/16)
        x2_d2_ = self.middle2_d2(x2_d2) + middle_skip2_d2  # (B, 64, T/8, H/16, W/16)
        output = self.output_proj_d2(x2_d2_) 
        output = output+skip_d2
        return output
    
    
    def decoder3(self,x):
        # 保存输入用于跳跃连接
        skip_d3 = self.skip_proj_d3(x)  # (B, 16, T/4, H/8, W/8) (8,16,12,56,104)
        
        # 第一次上采样
        x1_d3 = self.up_d3(x)  # (B, 64, T/16, H/16, W/16) (8,64,3,14,26)
        
        # 中间处理 + 残差连接
        middle_skip1_d3 = self.middle_skip1_d3(x)  # (B, 64, T/8, H/16, W/16)
        x1_d3_ = self.middle1_d3(x1_d3) + middle_skip1_d3  # (B, 64, T/8, H/16, W/16)
        output = self.output_proj_d3(x1_d3_) 
        output = output + skip_d3
        
        return output
    
    def forward(self, latent1, latent2, latent3):
        # import ipdb;ipdb.set_trace()
        output1 = self.decoder1(latent1) # x8 8,16,3,7,13 ->8,16,12,56,104
        output2 = self.decoder2(latent2) # x4
        output3 = self.decoder3(latent3) # x2
        
        return output1, output2, output3
    

class EnhancedResidualDownsample3d_v2(nn.Module):

    """
    增强版带残差连接的降采样网络
    输入: (B, 16, T/4, H/8, W/8)
    输出: (B, 16, T/16, H/64, W/64)
    """
    
    def __init__(self, dim=16, hidden_dim=96):
        super().__init__()
        
        # 输入投影层 - 扩展通道数
        self.input_proj = nn.Conv3d(dim, hidden_dim, 1)
        
        # 第一个降采样块 (增加更多层)
        self.down1 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        # 中间处理层
        self.middle1 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        # 中间处理层
        self.middle2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        # 第二个降采样块
        self.down2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )

        # 第三个个降采样块
        self.down3 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        # 输出投影层 - 恢复通道数
        self.output_proj = nn.Conv3d(hidden_dim, dim, 1)
        
        # 跳跃连接的投影层
        self.skip_proj = nn.Conv3d(dim, dim, 1, stride=(4, 8, 8))
        
        # 中间层的残差连接
        self.middle_skip1 = nn.Conv3d(hidden_dim, hidden_dim, 1, stride=(2, 2, 2))
        self.middle_skip2 = nn.Conv3d(hidden_dim, hidden_dim, 1, stride=(2, 2, 2))
        
        
    def forward(self, x):
        # 保存输入用于跳跃连接
        # import ipdb;ipdb.set_trace() # x(8,16,12,56,104)
        skip = self.skip_proj(x)  # (B, 16, T/16, H/64, W/64)  (8,16,3,14,26) (8,16,3,7,13)
        
        # 主路径
        x = self.input_proj(x)  # (B, 64, T/4, H/8, W/8) (8,64,12,56,104) (8,64,12,56,104)
        
        # 第一次降采样 降2倍
        x1 = self.down1(x)  # (B, 64, T/8, H/16, W/16) (8,64,6,28,52)    (8,64,6,28,52)
        
        # 中间处理 + 残差连接
        middle_skip1 = self.middle_skip1(x)  # (B, 64, T/8, H/16, W/16) (8,64,6,28,52) (8,64,6,28,52)
        x1_ = self.middle1(x1) + middle_skip1  # (B, 64, T/8, H/16, W/16) (8,64,6,28,52) (8,64,6,28,52)
        
        # 第二次降采样 降2倍
        x2 = self.down2(x1_)  # (B, 64, T/16, H/64, W/64) (8,64,3,14,26)  (8,64,3,14,26)
        
        # 中间处理 + 残差连接
        middle_skip2 = self.middle_skip2(x1_)  # (B, 64, T/8, H/16, W/16) (8,64,6,28,52)
        x2_ = self.middle2(x2) + middle_skip2  # (B, 64, T/8, H/16, W/16) (8,64,6,28,52)
        # import ipdb;ipdb.set_trace()
        #第三次降采样 
        x3 = self.down3(x2_)  # (B, 64, T/16, H/64, W/64) (8,64,3,14,26)
        # 输出投影
        output = self.output_proj(x3)  # (B, 16, T/16, H/64, W/64) (8,16,3,14,26)
        
        # 最终残差连接
        output = output + skip
        
        return output, x2_, x1_ #(8,16,3,7,13); (8,64,6,28,52)








class MyTransformerBlock3D(nn.Module):
    """
    Pre-Norm Transformer:
        x = x + MHA(Norm(x))
        x = x + FFN(Norm(x))
    """
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.0, eps=1e-6):
        super().__init__()
        self.attn  = SelfAttention(dim, heads, eps)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, freqs):
        # x: (B, C, T, H, W)  ->  (B, THW, C)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                # (B, THW, C)
        x = x + self.attn(x, freqs)
        x = x + self.ffn(self.norm2(x))
        x = x.transpose(1, 2).view(B, C, T, H, W)       # 还原
        return x

class Upsample3d_Coder_v3(nn.Module):
    def __init__(self, hidden_dim=96, num_heads=4, eps=1e-6, depth=2, stride=(1,2,2)):
        super().__init__()
        mlp_ratio = 4.0
        dropout = 0.1
        self.blocks = nn.ModuleList([
            MyTransformerBlock3D(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.up = nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
            padding=1, output_padding=tuple(s-1 for s in stride))
        
    def forward(self, x, freqs):
        for blk in self.blocks:
            x = blk(x, freqs)
        x = self.up(x)
        return x

class Downsample3d_Coder_v3(nn.Module):
    def __init__(self, hidden_dim=96, num_heads=4, eps=1e-6, depth=2, stride=(1,2,2)):
        super().__init__()
        mlp_ratio = 4.0
        dropout = 0.1
        self.blocks = nn.ModuleList([
            MyTransformerBlock3D(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.down = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1)
        
    def forward(self, x, freqs):
        for blk in self.blocks:
            x = blk(x, freqs)
        x = self.down(x)
        return x



class EnhancedResidualUpsample3d_v3(nn.Module):
    """
    残差上采样网络
    输入: (B, 16, T/8, H/64, W/64)
    输出: (B, 16, T/4, H/8, W/8)
    """
    
    def __init__(self, dim=16, out_dim=16, hidden_dim=96, num_heads=4, eps=1e-6, depth=[2,2,2]):
        super().__init__()
        self.input_proj1 = nn.Conv3d(dim, hidden_dim, 1)
        self.input_proj2 = nn.Conv3d(dim, hidden_dim, 1)
        self.input_proj3 = nn.Conv3d(dim, hidden_dim, 1)

        self.freqs = precompute_freqs_cis_3d(hidden_dim // num_heads)
        
        # 三个分辨率级
        self.decoder1 = nn.ModuleList([
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[0], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(2,2,2)),
        ])

        self.decoder2 = nn.ModuleList([
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(2,2,2)),
        ])

        self.decoder3 = nn.ModuleList([Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(2,2,2))])

        self.output_proj1 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj2 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj3 = nn.Conv3d(hidden_dim, out_dim, 1)
        
    def forward(self, x1, x2, x3):
        x1 = self.input_proj1(x1)
        for blk in self.decoder1:
            b, c, f, h, w = x1.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x1.device)
            x1 = blk(x1, freqs)
        x1_out = self.output_proj1(x1)


        x2 = self.input_proj2(x2)
        for blk in self.decoder2:
            b, c, f, h, w = x2.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x2.device)
            x2 = blk(x2, freqs)
        x2_out = self.output_proj2(x2)

        x3 = self.input_proj2(x3)
        for blk in self.decoder3:
            b, c, f, h, w = x3.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x3.device)
            x3 = blk(x3, freqs)
        x3_out = self.output_proj3(x3)
        return x1_out, x2_out, x3_out

class EnhancedResidualDownsample3d_v3(nn.Module):

    """
    增强版带残差连接、且带self-attn的降采样网络
    输入: (B, 16, T/4, H/8, W/8)
    输出: (B, 16, T/8, H/64, W/64)
    """
    
    def __init__(self, dim=16, out_dim=16, hidden_dim=96, num_heads=4, eps=1e-6, depth=[2,2,2]):
        super().__init__()
        
        # 输入投影层 - 扩展通道数
        self.input_proj = nn.Conv3d(dim, hidden_dim, 1)
        self.freqs = precompute_freqs_cis_3d(hidden_dim // num_heads)
        
        # 三个分辨率级
        self.encoder1 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[0], stride=(2,2,2))
        self.encoder2 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2))
        self.encoder3 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(1,2,2))

        self.output_proj1 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj2 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj3 = nn.Conv3d(hidden_dim, out_dim, 1)
        

    def forward(self, x):
        x = self.input_proj(x)

        b, c, f, h, w = x.shape
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        x = self.encoder1(x, freqs)
        x1_out = self.output_proj1(x)

        b, c, f, h, w = x.shape
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        x = self.encoder2(x, freqs)
        x2_out = self.output_proj2(x)

        b, c, f, h, w = x.shape
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        x = self.encoder3(x, freqs)
        x3_out = self.output_proj3(x)
        
        return x1_out, x2_out, x3_out
    



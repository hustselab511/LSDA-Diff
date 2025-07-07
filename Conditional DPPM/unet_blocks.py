import torch.nn as nn
import torch
from torch.nn import Module
from torch import einsum
import torch.nn.functional as F

from VIT_encoder.utils import default, exists

from einops import rearrange
import math

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

    
def Upsample(dim, dim_out=None, pad=0):
    return nn.Sequential(
        nn.Sequential(nn.ConvTranspose1d(dim, default(dim_out, dim), 4, stride=2, padding=1, output_padding=pad),
                      batchnorm(default(dim_out, dim)),
                      nn.GELU())
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(nn.Conv1d(dim, default(dim_out, dim), 4, stride=2, padding=1),
                         batchnorm(default(dim_out, dim)),
                         nn.GELU())

def normalization(channels, groups=32):
    return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)

def batchnorm(channels):
    return nn.BatchNorm1d(channels)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim)) 

    def forward(self, x):
        return F.normalize(x, dim = 2) * self.g * (x.shape[2] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # self.norm = normalization(dim)
        self.norm = batchnorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class ResBlock(Module):
    def __init__(self, dim, dim_out, dropout=0., kernel_size=3):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel_size, padding=kernel_size//2) # same padding

        self.norm = batchnorm(dim_out)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        
        x = self.proj(x)
        x = self.norm(x)
        
        if scale_shift == None:
            x = self.dropout(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)

        return x

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = ResBlock(dim, dim_out)
        self.block2 = ResBlock(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb) # (b, dim_out*2)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1) # ((b, dim_out, 1), (b, dim_out, 1))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        h = self.block1(x, scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            context_dim=None,
            dropout=0.,
            prenorm=True
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.ref_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_k_x = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_x = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, reference=None):
        h = self.heads

        if reference is not None:
            reference = self.ref_norm(reference)
            k = self.to_k(reference) 
            v = self.to_v(reference) 
        else: 
            reference = x
            k = self.to_k_x(x)
            v = self.to_v_x(x)

        q = self.to_q(x)  
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v)) 
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale 
        attn = sim.softmax(dim=-1)
        
        # dropouts
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.heads = heads

        self.attn = CrossAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, context_dim=384)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x, reference=None):

        x = rearrange(x, 'b c n -> b n c')
        x = self.norm(x)
        
        if reference is not None:
            x = self.attn(x, reference=reference) + x
        else:
            x_attn, _ = self.self_attn(x, x, x)
            x = x_attn + x
        
        x = self.ff(x) + x

        x = rearrange(x, 'b n c -> b c n')
        return x

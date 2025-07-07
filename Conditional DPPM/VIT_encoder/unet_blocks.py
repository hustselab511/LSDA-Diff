import torch
import torch.nn as nn
from torch.nn import functional as F

from torch import einsum
from einops import rearrange


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels*2, out_channels)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #x = self.conv2(x)

        return x
    
class SegmentUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        
        x1 = self.up(x1)
        x = self.conv(x1)

        return x


class SelfAttention(nn.Module):
    def __init__(self, h_size, num_heads=8):
        super(SelfAttention, self).__init__()
        self.h_size = h_size
        self.mha = nn.MultiheadAttention(h_size, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value
    
class SAWrapper(nn.Module):
    def __init__(self, h_size):
        super(SAWrapper, self).__init__()
        self.sa = nn.Sequential(*[SelfAttention(h_size) for _ in range(1)])
        self.h_size = h_size

    def forward(self, x):
        x = self.sa(x.swapaxes(1, 2))
        return x.swapaxes(2, 1)


class CrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            context_dim=768
    ):
        super().__init__()
        context_dim = context_dim

        self.heads = heads
        self.dim_head = dim / heads
        self.scale = self.dim_head ** -0.5
        inner_dim = dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, reference=None):
        h = self.heads

        if reference is not None:
            k = self.to_k(reference) 
            v = self.to_v(reference) 

        q = self.to_q(x)  
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v)) 
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale 
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        # self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention = CrossAttention(dim=embed_dim, heads=num_heads)
        self.ln = nn.LayerNorm([embed_dim], elementwise_affine=True)
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([embed_dim], elementwise_affine=True),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, c):
        x_ln = self.ln(x.permute(0, 2, 1))
        c_ln = self.ln(c.permute(0, 2, 1))
        attention_value = self.cross_attention(x_ln, c_ln)
        attention_value = attention_value + x_ln
        attention_value = self.ff_cross(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)
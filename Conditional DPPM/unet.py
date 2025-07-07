from functools import partial

from torch import nn
from torch.nn import Module, ModuleList

from utils import default
from unet_blocks import *

# unet_model

class Unet1D(Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=12,
            seq_length=1024,
            dropout=0.,
            self_condition=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            sinusoidal_pos_emb_theta=10000,
            attn_heads=8,
            attn_dim_head=64,
            attn_dropout=0.  
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.seq_length = seq_length

        init_dim = default(init_dim, dim)
        
        self.init_conv = nn.Conv1d(self.channels, init_dim, 1)

        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(self.seq_length))
                                                for _ in range(self.channels))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)
        attn_block = partial(MultiHeadCrossAttention, heads=attn_heads, dim_head=attn_dim_head, dropout=attn_dropout)  

        self.downs = ModuleList([])
        self.ups = ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                Residual(attn_block(dim=dim_in, dim_head=min(64 * (ind + 1), 64))),
                Downsample(dim_in, dim_out) if not is_last else nn.Sequential(nn.Conv1d(dim_in, dim_out, 3, padding=1),
                                                                              nn.BatchNorm1d(dim_out),
                                                                              nn.GELU())
            ]))

        # middle layers
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(attn_block(dim=mid_dim))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            pad = 0

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                Residual(attn_block(dim=dim_out, dim_head=min(64 * (num_resolutions - ind), 64))),
                Upsample(dim_out, dim_in, pad=pad) if not is_last else nn.Sequential(nn.Conv1d(dim_out, dim_in, 3, padding=1),
                                                                            nn.BatchNorm1d(dim_in),
                                                                            nn.GELU())
            ]))

        self.out_dim = out_dim
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)
        

    def forward(self, x, time, x_self_cond=None, reference=None, is_train=True, use_reference_prob=0.85):
        
        if is_train and torch.rand(1).item() > use_reference_prob:
            reference = None
            
        x[:, 0:1, :] = x_self_cond
        
        x = self.init_conv(x) 
        residual = x

        t = self.time_mlp(time)

        h = []

        for block, attn, downsample in self.downs:
            x = block(x, t)

            x = attn(x, reference)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, reference)
        x = self.mid_block2(x, t)

        for block, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, t)

            x = attn(x, reference)

            x = upsample(x)

        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x, t)

        x = self.final_conv(x)
        return x
    
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from models.encoder.vit import TransformerBlock

class ST_MEM_Decoder(nn.Module):
    def __init__(self,
                 num_patches,
                 patch_size,
                 num_leads,
                 embed_dim,
                 decoder_embed_dim,
                 decoder_depth,
                 decoder_num_heads,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self._repr_dict = {'num_patches': num_patches,
                          'patch_size': patch_size,
                          'num_leads': num_leads,
                          'embed_dim': embed_dim,
                          'decoder_embed_dim': decoder_embed_dim,
                          'decoder_depth': decoder_depth,
                          'decoder_num_heads': decoder_num_heads,
                          'mlp_ratio': mlp_ratio,
                          'qkv_bias': qkv_bias,
                          'norm_layer': str(norm_layer)}
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_leads = num_leads
        
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 2, decoder_embed_dim),
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(input_dim=decoder_embed_dim,
                            output_dim=decoder_embed_dim,
                            hidden_dim=decoder_embed_dim * mlp_ratio,
                            heads=decoder_num_heads,
                            dim_head=64,
                            qkv_bias=qkv_bias)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)
    
    def forward(self, x, ids_restore=None, use_mask=True):
        x = self.to_decoder_embedding(x)

        if use_mask and ids_restore is not None:
            x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads)
            b, _, n_masked_with_sep, d = x.shape
            n = ids_restore.shape[2]

            mask_embeddings = self.mask_embedding.unsqueeze(1)
            mask_embeddings = mask_embeddings.repeat(b, self.num_leads, n + 2 - n_masked_with_sep, 1)
            
            # Unshuffle without SEP embedding
            x_wo_sep = torch.cat([x[:, :, 1:-1, :], mask_embeddings], dim=2)
            x_wo_sep = torch.gather(x_wo_sep, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, d))
        else:
            x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads)
            b, _, n, d = x.shape
            x_wo_sep = x
            # x_wo_sep = x[:, :, 1:-1, :]

        x_wo_sep = x_wo_sep + self.decoder_pos_embed[:, 1:x_wo_sep.shape[2]+1, :].unsqueeze(1)
        left_sep = x[:, :, :1, :] + self.decoder_pos_embed[:, :1, :].unsqueeze(1)
        right_sep = x[:, :, -1:, :] + self.decoder_pos_embed[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x_wo_sep, right_sep], dim=2)
        
        # lead-wise decoding
        x_decoded = []
        for i in range(self.num_leads):
            x_lead = x[:, i, :, :]
            for block in self.decoder_blocks:
                x_lead = block(x_lead)
            x_lead = self.decoder_norm(x_lead)
            x_lead = self.decoder_head(x_lead)
            x_decoded.append(x_lead[:, 1:-1, :])  # remove SEP embedding
        x = torch.stack(x_decoded, dim=1)
        return x
    
    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str
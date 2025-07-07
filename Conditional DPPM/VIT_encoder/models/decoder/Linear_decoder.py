import torch.nn as nn

class LinearDecoder(nn.Module):
    def __init__(self, patch_size, seq_len, num_leads,dropout=0.):
        super(LinearDecoder, self).__init__()
        
        self.num_leads = num_leads
        
        self.decoder = nn.ModuleList([nn.Sequential(nn.Linear(patch_size, seq_len),
                                                    nn.GELU(),
                                                    nn.Dropout(dropout)) 
                                        for _ in range(num_leads)])
        
    def forward(self, x, use_mask=True):
        x_decoded = []
        
        for i in range(self.num_leads):
            if use_mask:
                patch = torch.cat([self.decoder[i](x[:, i, :, :])], dim=1)
                x_decoded.append(patch) 
            else:
                x_decoded.append(self.decoder[i](x[:, i, :]))
        
        x = torch.stack(x_decoded, dim=1)
        return x
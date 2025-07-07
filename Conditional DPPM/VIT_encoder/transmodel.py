import torch
import torch.nn as nn

from einops import rearrange
from unet_blocks import *


class Transfer(nn.Module):
    def __init__(self, channels, out_dim):
        super().__init__()
        self.in_channels = channels
        self.out_dim = out_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.inc_x = DoubleConv(self.in_channels, 64)

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 1024)
        
        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)

        self.sa1_x = SAWrapper(128)
        self.sa2_x = SAWrapper(256)
        self.sa3_x = SAWrapper(512)
        self.sa4_x = SAWrapper(1024)
        self.sa5_x = SAWrapper(1024)
        
        self.outc_x = nn.Conv1d(32, self.out_dim, kernel_size=1)

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b (c n) p -> (b n) c p', c=self.in_channels)
        
        # Level 1
        x1 = self.inc_x(x)
        x2 = self.down1_x(x1)

        # Level 2
        x2 = self.sa1_x(x2)
        x3 = self.down2_x(x2)

        # Level 3
        x3 = self.sa2_x(x3)

        x4 = self.down3_x(x3)
        
        # Level 4
        x4 = self.sa3_x(x4)
        x5 = self.down4_x(x4)
        
        # Level 5
        x5 = self.sa4_x(x5)
        x6 = self.down5_x(x5)
        
        # Level 6
        x6 = self.sa5_x(x6)
        
        # Upward path
        x = self.up1_x(x6, x5)

        x = self.up2_x(x, x4)

        x = self.up3_x(x, x3)

        x = self.up4_x(x, x2)

        x = self.up5_x(x, x1)

        output = self.outc_x(x)
        output = rearrange(output, '(b n) c p -> b (c n) p', b = b)
    
        return output

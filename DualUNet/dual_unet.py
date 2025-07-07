import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_blocks import *

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=12, seq_length=1024):
        super().__init__()
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.out_dim = out_channels
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
        
        self.outc_x = nn.Conrandd(32, self.out_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.inc_x(x)
        x2 = self.down1_x(x1)
        
        x2 = self.sa1_x(x2)
        x3 = self.down2_x(x2)

        x3 = self.sa2_x(x3)
        x4 = self.down3_x(x3)
        
        x4 = self.sa3_x(x4)
        x5 = self.down4_x(x4)
        
        x5 = self.sa4_x(x5)
        x6 = self.down5_x(x5)
        
        x6 = self.sa5_x(x6)
        
        # up_list
        x = self.up1_x(x6, x5) # [128, 512, 64]
        
        x = self.up2_x(x, x4)

        x = self.up3_x(x, x3)

        x = self.up4_x(x, x2)

        x = self.up5_x(x, x1)

        output = self.outc_x(x)
    
        return output, x6

class DualUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=12, seq_length=1024):
        super().__init__()
        
        self.lead_I_model = UNet(in_channels, out_channels, seq_length)
        self.lead_rand_model = UNet(in_channels, out_channels, seq_length)
    
    def forward(self, lead_I, lead_rand=None):
        lead_I_out, lead_I_features = self.lead_I_model(lead_I)
        
        if self.training:
            if lead_rand is None:
                raise ValueError("lead_rand must be provided during training")
                
            lead_rand_out, lead_rand_features = self.lead_rand_model(lead_rand)
            
            return {
                'lead_I_output': lead_I_out,
                'lead_rand_output': lead_rand_out,
                'lead_I_features': lead_I_features,
                'lead_rand_features': lead_rand_features
            }
        else:
            return {
                'lead_I_output': lead_I_out,
                'lead_I_features': lead_I_features
            }
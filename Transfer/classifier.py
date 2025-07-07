import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, channels)
        
    def forward(self, x):
        y = x
        z = self.global_max_pool(x).squeeze(-1)
        z = F.elu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))

        z = z.unsqueeze(-1)

        x = x * z

        return x + y

class MultiDropout(nn.Module):
    def __init__(self, drop_rates):
        super(MultiDropout, self).__init__()
        self.drop_rates = drop_rates
    
    def forward(self, x):
        result = 0
        for rate in self.drop_rates:
            result += F.dropout(x, p=rate, training=self.training) / len(self.drop_rates)
        return result

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool1d(pool_size, stride=pool_size, padding=0)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.elu2 = nn.ELU()
        
        self.pool2 = nn.MaxPool1d(pool_size, stride=pool_size, padding=0)
        self.norm = nn.GroupNorm(out_channels//8, out_channels)
        self.attention = AttentionBlock(out_channels)
    
    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.elu1(x0)
        x0 = self.pool1(x0)

        x1 = self.conv2(x0)
        x1 = self.elu2(x1)

        x = x1 + x0

        x = self.pool2(x)
        x = self.norm(x)
        x = self.attention(x)
        
        return x

class Classifier(nn.Module):
    def __init__(self, length=15000, dim=12, classnum=9, kernel_size=5, pool_size=2, filnum=[32, 64, 128, 256]):
        super(Classifier, self).__init__()
        
        self.resblock0 = ResBlock(dim, filnum[0], kernel_size, pool_size)
        self.resblock1 = ResBlock(filnum[0], filnum[1], kernel_size, pool_size)
        self.resblock2 = ResBlock(filnum[1], filnum[2], kernel_size, pool_size)
        self.resblock3 = ResBlock(filnum[2], filnum[3], kernel_size, pool_size)

        self.resblock40 = ResBlock(filnum[0], filnum[-1], kernel_size, pool_size**4)
        self.resblock41 = ResBlock(filnum[1], filnum[-1], kernel_size, pool_size**3)
        self.resblock42 = ResBlock(filnum[2], filnum[-1], kernel_size, pool_size**2)
        self.resblock43 = ResBlock(filnum[3], filnum[-1], kernel_size, pool_size)
 
        self.norm = nn.GroupNorm(filnum[-1]//8, filnum[-1])
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.multidropout = MultiDropout([0.0, 0.2, 0.4, 0.8])
        self.fc = nn.Linear(filnum[-1], classnum)
    
    def forward(self, x):

        x0 = self.resblock0(x)
        x1 = self.resblock1(x0)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)

        x40 = self.resblock40(x0)
        x41 = self.resblock41(x1)
        x42 = self.resblock42(x2)
        x43 = self.resblock43(x3)

        x4 = x40 + x41 + x42 + x43
        x4 = self.norm(x4)

        y0 = self.globalavgpool(x4).squeeze(-1)

        y0 = self.multidropout(y0)

        y = self.fc(y0)

        y = torch.sigmoid(y)
        
        return y

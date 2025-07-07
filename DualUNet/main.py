import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import os
import time
import random
from tqdm import tqdm

from datautils import Dataset_ECG
from dual_unet import DualUNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--length', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--dataset', type=str, default='PTB_XL')
parser.add_argument('--flag', type=str, default='val')
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--lambda_I', type=float, default=1.0)
parser.add_argument('--lambda_rand', type=float, default=1.0)
parser.add_argument('--lambda_feat', type=float, default=50.0)

args = parser.parse_args()

train_set = Dataset_ECG(root_path=args.data_path, flag='train', dataset=args.dataset)  
test_set = Dataset_ECG(root_path=args.data_path, flag=args.flag, dataset=args.dataset)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = DualUNet(in_channels=1, out_channels=12, seq_length=args.length)
 
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
criterion = nn.MSELoss()
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'rand', 'V2', 'V3', 'V4', 'V5', 'V6']

if not os.path.exists('logs'):
    os.makedirs('logs')
    
if not os.path.exists('model_dict'):
    os.makedirs('model_dict')

train_loss_list = []
epoch_times = []
best_val_loss = 1e10

for epoch in range(args.epochs):
    start_time = time.time() 
    train_loss = 0.
    val_loss = 0.
    model.train()
    
    second_lead_idx = random.choice(list(range(1, 12)))  
    second_lead_name = leads[second_lead_idx]
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train] Using leads I and {second_lead_name}", leave=False)
    for idx, (inputs, targets) in enumerate(train_loader_tqdm): 
        targets = targets.to(args.device)

        lead_I = inputs[:, 0:1, :].to(args.device)
        lead_rand = inputs[:, second_lead_idx:second_lead_idx+1, :].to(args.device)
        
        optimizer.zero_grad()
        outputs = model(lead_I, lead_rand) 

        lead_I_output = outputs['lead_I_output']
        lead_rand_output = outputs['lead_rand_output']
        lead_I_features = outputs['lead_I_features']
        lead_rand_features = outputs['lead_rand_features']
        
        lead_I_loss = criterion(lead_I_output, targets)
        lead_rand_loss = criterion(lead_rand_output, targets)

        align_loss = criterion(lead_I_features, lead_rand_features)

        feature_loss = criterion(lead_I_features, lead_rand_features)

        loss = args.lambda_I * lead_I_loss + args.lambda_rand * lead_rand_loss + args.lambda_feat * align_loss
       
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loader_tqdm.set_postfix(
            train_loss=f"{train_loss / (idx + 1):.4f}",
            I_loss=f"{lead_I_loss.item():.4f}",
            rand_loss=f"{lead_rand_loss.item():.4f}",
            feat_loss=f"{feature_loss.item():.4f}"
        )

    train_loss = train_loss / len(train_loader)
    train_loss_list.append(train_loss)
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_times.append(epoch_time)
    loader_size = len(test_loader)

    if (epoch + 1) % args.val_interval == 0:
        model.eval()
        val_loss = 0.0
        lead_I_losses = [0.0] * 12
        lead_I_pccs = [0.0] * 12
        
        with torch.no_grad():
            val_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False)
            for idx, (inputs, targets) in enumerate(val_loader_tqdm):
                targets = targets.to(args.device)
                
                lead_I = inputs[:, 0:1, :].to(args.device)
                
                outputs = model(lead_I)
                
                lead_I_output = outputs['lead_I_output']
                
                lead_I_loss = criterion(lead_I_output, targets)
                val_loss += lead_I_loss.item()
                
                for ch in range(12):
                    ch_I_loss = criterion(lead_I_output[:, ch], targets[:, ch])
                    lead_I_losses[ch] += ch_I_loss.item()
                    
                    ch_I_pcc = cal_pearson(lead_I_output[:, ch], targets[:, ch]).mean()
                    lead_I_pccs[ch] += ch_I_pcc.item()
                
                val_loader_tqdm.set_postfix(val_loss=f"{val_loss / (idx + 1):.4f}")

            val_loss = val_loss / loader_size
            lead_I_losses = [loss / loader_size for loss in lead_I_losses]
            lead_I_pccs = [pcc / loader_size for pcc in lead_I_pccs]
            
            avg_I_pcc = sum(lead_I_pccs) / len(lead_I_pccs)
            
            with open(f'logs/dualunet_val_{args.lambda_feat}.txt', 'a') as f:
                f.write('-' * 35 + '\n')
                f.write(f'Epoch: {epoch + 1}\n')
                f.write(f'Using leads I and {second_lead_name} for training\n')
                f.write(f'Average loss: {val_loss:.4f}\n')
                f.write(f'Average Lead I PCC: {avg_I_pcc:.4f}\n')
                
                f.write('\nLead I Model Performance:\n')
                for lead, loss, pcc in zip(leads, lead_I_losses, lead_I_pccs):
                    f.write(f'{lead}: Loss={loss:.4f}, PCC={pcc:.4f}\n')
                
                f.write('-' * 35 + '\n')
            
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), './model_dict/best_dual_unet.pth')
                
with open(f'logs/dualunet_val_{args.lambda_feat}.txt', 'a') as f:
    f.write('-' * 35 + '\n')
    f.write(f'Best loss: {best_val_loss:.5f}\n')
    f.write('-' * 35 + '\n')
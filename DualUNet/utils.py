import torch
import numpy as np
import os

def cal_pearson(x, y):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    y_mean = torch.mean(y, dim=-1, keepdim=True)
    
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(torch.sum(x_centered**2, dim=-1)) * torch.sqrt(torch.sum(y_centered**2, dim=-1))
    
    r = numerator / (denominator + 1e-8)
    return r


def evaluate_pcc(model, device, loader, save_path=None, return_inputs=False):
    model.eval()
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    all_pccs_I = []
    all_pccs_V1 = []
    all_inputs = []
    all_outputs_I = []
    all_outputs_V1 = []
    all_targets = []
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            targets = targets.to(device)
            lead_I = inputs[:, 0:1, :].to(device)
            lead_V1 = inputs[:, 6:7, :].to(device)
            
            outputs = model(lead_I, lead_V1)
            lead_I_output = outputs['lead_I_output']
            lead_V1_output = outputs['lead_V1_output']
            
            # Calculate PCC for each channel
            for ch in range(12):
                ch_I_pcc = cal_pearson(lead_I_output[:, ch], targets[:, ch])
                ch_V1_pcc = cal_pearson(lead_V1_output[:, ch], targets[:, ch])
                
                all_pccs_I.append(ch_I_pcc.cpu().numpy())
                all_pccs_V1.append(ch_V1_pcc.cpu().numpy())
            
            if return_inputs:
                all_inputs.append(inputs.cpu().numpy())
                all_outputs_I.append(lead_I_output.cpu().numpy())
                all_outputs_V1.append(lead_V1_output.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    avg_pccs_I = [np.mean(np.concatenate(all_pccs_I)[i::12]) for i in range(12)]
    avg_pccs_V1 = [np.mean(np.concatenate(all_pccs_V1)[i::12]) for i in range(12)]
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Lead I Model Performance:\n")
            for lead, pcc in zip(leads, avg_pccs_I):
                f.write(f"{lead}: {pcc:.4f}\n")
                
            f.write("\nLead V1 Model Performance:\n")
            for lead, pcc in zip(leads, avg_pccs_V1):
                f.write(f"{lead}: {pcc:.4f}\n")
                
            f.write(f"\nAverage Lead I PCC: {np.mean(avg_pccs_I):.4f}\n")
            f.write(f"Average Lead V1 PCC: {np.mean(avg_pccs_V1):.4f}\n")
    
    if return_inputs:
        return {
            'inputs': np.concatenate(all_inputs),
            'outputs_I': np.concatenate(all_outputs_I),
            'outputs_V1': np.concatenate(all_outputs_V1),
            'targets': np.concatenate(all_targets),
            'pccs_I': avg_pccs_I,
            'pccs_V1': avg_pccs_V1
        }
    
    return {
        'pccs_I': avg_pccs_I,
        'pccs_V1': avg_pccs_V1
    }
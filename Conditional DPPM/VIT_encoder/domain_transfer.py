import argparse
import os

import torch
import torch.nn as nn
import yaml
from einops import rearrange

from datautils import Dataset_ECG_VIT
from torch.utils.data import DataLoader
from utils import load_config, z_score_normalize, load_encoder, load_pre_encoded
from transmodel import Transfer

def create_ref(config, device, model, pre_encoded, transfer_model):
    
    dataset = Dataset_ECG_VIT(root_path=config['root_path'], flag=config['flag'], dataset=config['dataset'], seq_length=config['model']['seq_len'])    
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    references = []
    
    with torch.no_grad():
        for (seqs, _) in data_loader:
            seqs = seqs.type(torch.FloatTensor).to(device, non_blocking=True)
            
            samples = pre_encoded(seqs)['lead_I_output']
            samples = z_score_normalize(samples)

            sample_embed = model(samples)
            sample_embed = transfer_model(sample_embed)
            
            sample_embed = rearrange(sample_embed, 'b (c n) p -> b c n p', c=12)  
            sample_embed = torch.mean(sample_embed, dim=2)
        
            references.append(sample_embed)

        references = torch.cat(references, dim=0)

        os.makedirs(config['ref_path'], exist_ok=True)
        torch.save(references, os.path.join(config['ref_path'], f'{config["dataset"]}_{config["flag"]}_ref.pt'))


def main(config):
    device = torch.device(config['device'])
    
    model = load_encoder(config, device)
    pre_encoded = load_pre_encoded(config, device)

    transmodel = Transfer(12, 12)
    transmodel.load_state_dict(torch.load('./best_transfer.pth'))
    transmodel.to(device)
    transmodel.eval()
    
    create_ref(config, device, model, pre_encoded, transmodel)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str, default='ref', choices=['embedding', 'ref'])
    parser.add_argument('--flag', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--dataset', type=str, default='PTB_XL', choices=['PTB_XL', 'CPSC', 'Georgia', 'PTB'])
    
    # path configurations
    parser.add_argument('--config_path', type=str, default="./configs/downstream/st_mem.yaml", help='YAML config file path')
    parser.add_argument('--root_path', type=str, default="", help='Path of dataset')
    parser.add_argument('--embedding_path', type=str, default="./database", help='Path to save embeddings')
    parser.add_argument('--ref_path', type=str, default="./database", help='Path to save references')
    
    args = parser.parse_args()
    
    main(config=load_config(args=args))
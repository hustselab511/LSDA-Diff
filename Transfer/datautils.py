import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

def load_from_npy(file_path):
    data = np.load(file_path)
    
    data_np = data['predict']
    labels_np = data['labels'] if 'labels' in data else None
    length = data['beat_lengths']
    
    signal = torch.tensor(data_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.float32) if labels_np is not None else None
    length = torch.tensor(length, dtype=torch.float32)
    
    return signal, labels, length

class Dataset_ECG_VIT(Dataset):
    def __init__(self, root_path, flag, lead='I', dataset='PTB_XL', status='train', ref_path=None, seq_length=1024):

        self.root_path = root_path
        self.ref_path = ref_path
        self.seq_length = seq_length
        self.status = status

        self.flag = flag
        if flag == 'train':
            self.data_path = str(Path(dataset) / 'train_data.npz')
        elif flag == 'val':
            self.data_path = str(Path(dataset) / 'val_data.npz')
        elif flag == 'test':
            self.data_path = str(Path(dataset) / 'test_data.npz')
        
        self.lead_names = {
            'I': 0, 'II': 1, 'III': 2,
            'aVR': 3, 'aVL': 4, 'aVF': 5,
            'V1': 6, 'V2': 7, 'V3': 8,
            'V4': 9, 'V5': 10, 'V6': 11
        }
        
        self.all, self.labels, self.length = load_from_npy(os.path.join(self.root_path, self.data_path))
        self.seq = self.all[:, [self.lead_names[lead]], :]
        
        if self.ref_path is not None:
            self.reference = torch.load(os.path.join(self.ref_path, f'{dataset}_{self.flag}_ref.pt'), map_location='cpu')
        else:
            self.reference = None    
            
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        if self.status == 'sample':
            return self.seq[idx], self.all[idx], self.labels[idx], self.length[idx]
        else:
            return self.seq[idx], self.all[idx]
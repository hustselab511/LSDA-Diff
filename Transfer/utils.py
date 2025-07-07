import math
import os
import torch
from pathlib import Path
import yaml

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def path(root_dir, sub_dir):
    os.makedirs(Path(root_dir) / sub_dir, exist_ok=True)
    return Path(root_dir) / sub_dir

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# loading functions

def load_config(args) -> dict:
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    for k, v in vars(args).items():
        if v and k in config:
            config[k] = v

    return config


def load_encoder(config, device):
    import models.encoder as encoder
    
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] != "scratch":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model)
         
    model.to(device)
    model.eval()
    
    return model


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def z_score_normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / (var + 1.e-6)**.5

def normalize(x):
    min_vals = x.min(dim=1, keepdim=True)[0]
    min_vals = min_vals.min(dim=2, keepdim=True)[0]
    
    max_vals = x.max(dim=1, keepdim=True)[0]
    max_vals = max_vals.max(dim=2, keepdim=True)[0]

    x = (x - min_vals) / (max_vals - min_vals)
    
    return x

def unnormalize(x, min_vals, max_vals):
    return x * (max_vals - min_vals) + min_vals

# metric functions

def calculate_pcc(x, y):
    vx = x - torch.mean(x, dim=-1, keepdim=True)
    vy = y - torch.mean(y, dim=-1, keepdim=True)
    
    pcc = torch.sum(vx * vy, dim=-1) / (torch.sqrt(torch.sum(vx ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)) + 1e-8)
    return pcc

def calculate_r2(pred, target):
        target_mean = torch.mean(target, dim=-1, keepdim=True)
        ss_tot = torch.sum((target - target_mean) ** 2, dim=-1)
        ss_res = torch.sum((target - pred) ** 2, dim=-1)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2
    
def relabel_data(labels):
    relabeled = []
    for label in labels:
        if torch.equal(label, torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)):
            relabeled.append(0)
        else:
            relabeled.append(1)
    return torch.tensor(relabeled, dtype=torch.long)

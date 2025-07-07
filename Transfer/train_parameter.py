import argparse
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from tqdm import tqdm
from einops import rearrange

from transmodel import Transfer
from classifier import Classifier
from utils import *

def relabel_data(labels):
    relabeled = []
    for label in labels:
        if torch.equal(label, torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)):
            relabeled.append(0)
        else:
            relabeled.append(1)
    return torch.tensor(relabeled, dtype=torch.long)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', type=int, default=30, help='epoch number of training')
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--config_path', default="./configs/downstream/st_mem.yaml", type=str)
    parser.add_argument('--lambda_class', type=float, default=0.25)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args)
    encode = load_encoder(config, device)

    train_data = torch.load('/root/autodl-tmp/DDPM/DualUNet/samples/samples_PTB_XL_dualunet_train.pt')
    val_data = torch.load('/root/autodl-tmp/DDPM/DualUNet/samples/samples_PTB_XL_dualunet_val.pt')
    
    inputs_train = train_data['samples']
    targets_train = train_data['target']
    inputs_val = val_data['samples']
    targets_val = val_data['target']
    
    train_label = train_data['labels']
    test_label = val_data['labels']
    
    inputs_data = torch.cat([inputs_train, inputs_val], dim=0)
    targets_data = torch.cat([targets_train, targets_val], dim=0)
    labels_data = torch.cat([train_label, test_label], dim=0)
    
    relabeled_data = relabel_data(labels_data)
    
    encoded_inputs = []
    encoded_targets = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs_data), args.batch_size), desc="Encoding data"):
            batch_inputs = inputs_data[i:i+args.batch_size].to(device)
            batch_targets = targets_data[i:i+args.batch_size].to(device)
            
            batch_inputs = z_score_normalize(batch_inputs)
            batch_targets = z_score_normalize(batch_targets)
            
            encoded_input = encode(batch_inputs)
            encoded_target = encode(batch_targets)
            
            # encoded_input = rearrange(encoded_input, 'b (c n) p -> b c (n p)', c=12)
            # encoded_target = rearrange(encoded_target, 'b (c n) p -> b c (n p)', c=12)
            
            encoded_inputs.append(encoded_input.cpu())
            encoded_targets.append(encoded_target.cpu())  
              
    encoded_inputs = torch.cat(encoded_inputs, dim=0)
    encoded_targets = torch.cat(encoded_targets, dim=0)
    feature_dim = encoded_inputs.shape[-1]
    
    dataset = TensorDataset(encoded_inputs, encoded_targets, relabeled_data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Transfer(seq_length=feature_dim, channels=12, out_dim=12).to(device)
    classifier = Classifier(length=384, dim=12, classnum=2)
    classifier.load_state_dict(torch.load('model_dict/best_classifier.pth'))
    classifier.to(device)
    classifier.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('model_dict'):
        os.makedirs('model_dict')

    best_loss = float('inf')
    for epoch in range(args.train_epoch):
        epoch_loss = 0.0
        epoch_mse = 0.0

        model.train()

        data_loader_tqdm = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.train_epoch}", leave=False)
        for batch_idx, (inputs, targets, labels) in enumerate(data_loader_tqdm):
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            mse_loss = mse_criterion(outputs, targets)
            
            outputs_reshaped = rearrange(outputs, 'b (c n) p -> b c (n p)', c=12)
            classification = classifier(outputs_reshaped)
            classification_loss = classification_criterion(classification, labels)

            total_loss = mse_loss + args.lambda_class * classification_loss
            total_loss = mse_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_mse += mse_loss.item()

            data_loader_tqdm.set_postfix(
                mse_loss=f"{mse_loss.item():.4f}",
                classification_loss=f"{classification_loss.item():.4f}"
            )

        avg_loss = epoch_loss / len(data_loader)
        avg_mse = epoch_mse / len(data_loader)

        # print(f"Epoch {epoch + 1}, MSE: {avg_mse:.4f}, Classification Loss: {avg_classification_loss:.4f}")

        with open(f'logs/training_{args.lambda_class}.txt', 'a') as f:
            f.write(f'Epoch: {epoch + 1}\n')
            f.write(f'Average MSE Loss: {avg_mse:.4f}\n')
            f.write('\n')

        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save(model.state_dict(), f'model_dict/best_transfer_{args.lambda_class}.pth')
            
    with open(f'logs/training_{args.lambda_class}.txt', 'a') as f:
            f.write(f'Best Loss: {best_loss:.4f}\n')
            f.write('\n')
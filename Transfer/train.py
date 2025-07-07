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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', type=int, default=100, help='epoch number of training')
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--config_path', default="./configs/downstream/st_mem.yaml", type=str)
    parser.add_argument('--lambda_class', type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args)
    encode = load_encoder(config, device)

    train_data = torch.load('./samples_PTB_XL_dualunet_train.pt')
    val_data = torch.load('./samples_PTB_XL_dualunet_val.pt')
    
    inputs_train = train_data['samples']
    targets_train = train_data['target']
    train_labels = train_data['labels']
    
    inputs_val = val_data['samples']
    targets_val = val_data['target']
    val_labels = val_data['labels']
    
    relabeled_train = relabel_data(train_labels)
    relabeled_val = relabel_data(val_labels)
    
    def encode_data(inputs, targets):
        encoded_inputs = []
        encoded_targets = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), args.batch_size), desc="Encoding data"):
                batch_inputs = inputs[i:i+args.batch_size].to(device)
                batch_targets = targets[i:i+args.batch_size].to(device)
                
                batch_inputs = z_score_normalize(batch_inputs)
                batch_targets = z_score_normalize(batch_targets)
                
                encoded_input = encode(batch_inputs)
                encoded_target = encode(batch_targets)
                
                encoded_inputs.append(encoded_input.cpu())
                encoded_targets.append(encoded_target.cpu())
                
        return torch.cat(encoded_inputs, dim=0), torch.cat(encoded_targets, dim=0)
    
    encoded_inputs_train, encoded_targets_train = encode_data(inputs_train, targets_train)
    encoded_inputs_val, encoded_targets_val = encode_data(inputs_val, targets_val)
    
    train_dataset = TensorDataset(encoded_inputs_train, encoded_targets_train, relabeled_train)
    val_dataset = TensorDataset(encoded_inputs_val, encoded_targets_val, relabeled_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = Transfer(channels=12, out_dim=12).to(device)
    classifier = Classifier(length=384, dim=12, classnum=2).to(device)
    classifier.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('model_dict'):
        os.makedirs('model_dict')

    best_val_loss = float('inf')
    
    for epoch in range(args.train_epoch):
        model.train()
        train_loss = 0.0
        train_mse = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.train_epoch}", leave=False)
        for batch_idx, (inputs, targets, labels) in enumerate(train_loader_tqdm):
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            mse_loss = mse_criterion(outputs, targets)
            
            classification = classifier(outputs)
            classification_loss = classification_criterion(classification, labels)

            total_loss = mse_loss + args.lambda_class * classification_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_mse += mse_loss.item()

            train_loader_tqdm.set_postfix(
                train_mse=f"{mse_loss.item():.4f}",
                train_class=f"{classification_loss.item():.4f}"
            )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mse = train_mse / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for inputs, targets, labels in val_loader:
                inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device)
                
                outputs = model(inputs)
                mse_loss = mse_criterion(outputs, targets)
                
                classification = classifier(outputs)
                classification_loss = classification_criterion(classification, labels)
                
                total_loss = mse_loss + args.lambda_class * classification_loss
                
                val_loss += total_loss.item()
                val_mse += mse_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        with open(f'logs/training_{args.lambda_class}.txt', 'a') as f:
            f.write(f'Epoch: {epoch + 1}\n')
            f.write(f'Train MSE Loss: {avg_train_mse:.4f}\n')
            f.write(f'Val MSE Loss: {avg_val_mse:.4f}\n')
            f.write(f'Train Total Loss: {avg_train_loss:.4f}\n')
            f.write(f'Val Total Loss: {avg_val_loss:.4f}\n')
            f.write('\n')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_dict/best_transfer.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
    with open(f'logs/training_{args.lambda_class}.txt', 'a') as f:
        f.write(f'Best Validation Loss: {best_val_loss:.4f}\n')
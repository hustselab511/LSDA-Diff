import argparse
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from einops import rearrange

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
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--config_path', default="./configs/downstream/st_mem.yaml", type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args)
    encode = load_encoder(config, device)

    train_data = torch.load('./samples_PTB_XL_dualunet_train.pt')
    val_data = torch.load('./samples_PTB_XL_dualunet_val.pt')
    
    inputs_train = train_data['target']
    inputs_val = val_data['target']
    
    train_labels = train_data['labels']
    val_labels = val_data['labels']
    
    relabeled_train = relabel_data(train_labels)
    relabeled_val = relabel_data(val_labels)
    
    def encode_data(inputs):
        encoded_inputs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), args.batch_size), desc="Encoding data"):
                batch_inputs = inputs[i:i+args.batch_size].to(device)
                batch_inputs = z_score_normalize(batch_inputs)
                encoded_input = encode(batch_inputs)
                encoded_input = rearrange(encoded_input, 'b (c n) p -> b c (n p)', c=12)
                encoded_inputs.append(encoded_input.cpu())
        return torch.cat(encoded_inputs, dim=0)
    
    encoded_train = encode_data(inputs_train)
    encoded_val = encode_data(inputs_val)
    
    train_dataset = TensorDataset(encoded_train, relabeled_train)
    val_dataset = TensorDataset(encoded_val, relabeled_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    classifier = Classifier(length=384, dim=12, classnum=2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('model_dict'):
        os.makedirs('model_dict')

    best_val_loss = float('inf')
    
    for epoch in range(args.train_epoch):
        classifier.train()
        train_loss = 0.0
        train_predictions = []
        train_labels_list = []

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.train_epoch}", leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_loader_tqdm):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_predictions.extend(predictions.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        
        classifier.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels_list = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        train_f1 = f1_score(train_labels_list, train_predictions)
        train_accuracy = accuracy_score(train_labels_list, train_predictions)
        train_precision = precision_score(train_labels_list, train_predictions)
        train_recall = recall_score(train_labels_list, train_predictions)
        
        val_f1 = f1_score(val_labels_list, val_predictions)
        val_accuracy = accuracy_score(val_labels_list, val_predictions)
        val_precision = precision_score(val_labels_list, val_predictions)
        val_recall = recall_score(val_labels_list, val_predictions)

        print(f"Epoch {epoch + 1}")
        print(f"Train - Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_accuracy:.4f}")
        print(f"Val - Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_accuracy:.4f}")

        with open('logs/classifier_training.txt', 'a') as f:
            f.write(f'Epoch: {epoch + 1}\n')
            f.write(f'Train Loss: {avg_train_loss:.4f}\n')
            f.write(f'Train F1: {train_f1:.4f}\n')
            f.write(f'Train Accuracy: {train_accuracy:.4f}\n')
            f.write(f'Train Precision: {train_precision:.4f}\n')
            f.write(f'Train Recall: {train_recall:.4f}\n')
            f.write(f'Val Loss: {avg_val_loss:.4f}\n')
            f.write(f'Val F1: {val_f1:.4f}\n')
            f.write(f'Val Accuracy: {val_accuracy:.4f}\n')
            f.write(f'Val Precision: {val_precision:.4f}\n')
            f.write(f'Val Recall: {val_recall:.4f}\n')
            f.write('\n')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(classifier.state_dict(), 'model_dict/best_classifier.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
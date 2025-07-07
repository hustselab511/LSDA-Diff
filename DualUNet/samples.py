import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, ConcatDataset
from datautils import Dataset_ECG
from dual_unet import DualUNet

def sample_from_model():
    parser = argparse.ArgumentParser(description='DualUNet Sampling')
    parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
    parser.add_argument('--length', type=int, default=1024, help='length of signal')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--model', type=str, default='DualUNet')
    parser.add_argument('--dataset', type=str, default='PTB_XL', help='dataset name')
    parser.add_argument('--flag', type=str, default='val')
    parser.add_argument('--model_path', type=str, default='model_dict/best_dual_unet.pth')
    parser.add_argument('--output_path', type=str, default='samples', help='output directory')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_path, exist_ok=True)

    test_set = Dataset_ECG(root_path=args.data_path, flag=args.flag,  dataset=args.dataset, status='sample')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = DualUNet(in_channels=1, out_channels=12, seq_length=args.length)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        all_samples = []
        all_targets = []
        all_labels = []
        all_lengths = []
        
        pbar = tqdm(test_loader, desc="Sampling", leave=True)
        for inputs, targets, labels, length in pbar:
            targets = targets.to(device)
            
            lead_I = inputs[:, 0:1, :].to(device)
            
            outputs = model(lead_I)
            generated_samples = outputs['lead_I_output']

            all_samples.append(generated_samples.cpu())
            all_targets.append(targets.cpu())
            all_labels.append(labels.cpu())
            all_lengths.append(length.cpu())

        samples = torch.cat(all_samples, dim=0)
        targets = torch.cat(all_targets, dim=0)
        labels = torch.cat(all_labels, dim=0)
        lengths = torch.cat(all_lengths, dim=0)

        output = {
            'samples': samples,
            'target': targets,
            'lengths': lengths,
            'labels': labels
        }

        output_file = os.path.join(args.output_path, f'samples_{args.dataset}_dualunet_{args.flag}.pt')
        torch.save(output, output_file)

if __name__ == '__main__':
    sample_from_model()
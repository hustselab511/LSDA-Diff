import argparse

import torch
import torch.nn as nn

from trainer import Trainer1D

from unet import Unet1D
from diffusion_model import GaussianDiffusion1D
from VIT_encoder.datautils import Dataset_ECG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument("--dataset", type=str, default='PTB_XL', choices=["PTB_XL", "CPSC", "PTB", "Georgia"])
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--ref_path", type=str, default="./VIT_encoder/database")
    
    # Trainer
    parser.add_argument("--train_steps", type=int, default=150)
    parser.add_argument("--sample_interval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lead", type=str, default='I', help='Choose which lead as condition')
    parser.add_argument("--gradient_accumulate_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--result_path", type=str, default='./results')
    
    # UNet
    parser.add_argument("--in_c", type=int, default=12)
    parser.add_argument("--out_dim", type=int, default=12)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--init_dim", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--condition", type=bool, default=True)
    parser.add_argument("--is_random_pe", type=bool, default=False)
    
    # GaussianDiffusion
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--objective", type=str, default="pred_noise", choices=["pred_noise", "pred_x0", "pred_v"])
    parser.add_argument("--beta", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--normalize", type=bool, default=False)

    # Evaluation
    criterion_dict = {
        'MSELoss': nn.MSELoss(),
        'L1Loss': nn.L1Loss()
    }
    parser.add_argument("--status", type=str, default="train", choices=["train", "test", "sample"])
    parser.add_argument("--resume", type=int, default=0, help='Resume model from checkpoint')
    parser.add_argument("--criterion", type=str, default='MSELoss', choices=criterion_dict.keys())
    parser.add_argument("--seed", type=int, default=3407, help='Random seed for reproducibility')

    args = parser.parse_args()


    unet = Unet1D(dim=args.dim, 
                  init_dim=args.init_dim, 
                  out_dim=args.out_dim,
                  channels=args.in_c,
                  seq_length=args.length,
                  dropout=args.dropout,
                  self_condition=args.condition,
                  random_fourier_features=args.is_random_pe
                  )
    
    model = GaussianDiffusion1D(model=unet,
                                timesteps=args.timesteps,
                                sampling_timesteps=args.sampling_timesteps,
                                objective=args.objective,
                                beta_schedule=args.beta,
                                auto_normalize=args.normalize
                                )
    
    train_set = Dataset_ECG(root_path=args.root_path, flag='train', lead=args.lead, status=args.status,
                                seq_length=args.length, dataset='PTB_XL', ref_path=args.ref_path)
    val_set = Dataset_ECG(root_path=args.root_path, flag='val', lead=args.lead, status=args.status,
                            seq_length=args.length, dataset='PTB_XL', ref_path=args.ref_path)
    test_set = Dataset_ECG(root_path=args.root_path, flag='test', lead=args.lead, status=args.status,
                            seq_length=args.length, dataset=args.dataset, ref_path=args.ref_path)
    
    trainer = Trainer1D(diffusion_model=model, 
                        dataset=args.dataset,
                        train_set=train_set,
                        val_set=val_set,
                        test_set=test_set, 
                        lead=args.lead,
                        batch_size=args.batch_size,
                        gradient_accumulate_every=args.gradient_accumulate_every,
                        train_num_steps=args.train_steps,
                        train_lr=args.lr,
                        criterion=criterion_dict[args.criterion],
                        save_and_sample_every=args.sample_interval
                        )

    if args.status == 'train':
        trainer.train()
    elif args.status == 'test':
        trainer.load(args.resume, status=args.status)
        trainer.evaluate(trainer.test, criterion=trainer.criterion)
    else:
        trainer.load(args.resume, status=args.status)
        trainer.sample(trainer.test)
from pathlib import Path
from tqdm import tqdm
import datetime
import time
import json
from multiprocessing import cpu_count

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from unet_blocks import *

from accelerate import Accelerator
from ema_pytorch import EMA

from VIT_encoder.utils import path, calculate_pcc

class Trainer1D(object):
    def __init__(
            self,
            diffusion_model,
            criterion,
            dataset,
            train_set: Dataset,
            val_set: Dataset,
            test_set: Dataset,
            lead='I',
            *,
            batch_size=64,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100,
            is_training=True,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.999),
            save_and_sample_every=10,
            results_folder='./results',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        self.dataset = dataset
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, 
                                  pin_memory=True, num_workers=max(1, cpu_count() - 1))
        self.train_loader = train_loader

        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, 
                                pin_memory=True, num_workers=max(1, cpu_count() - 1))
        self.val = val_loader
        
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, 
                                 pin_memory=True, num_workers=max(1, cpu_count() - 1))
        self.test = test_loader

        # condition
        self.lead = lead

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # learning rate scheduler
        self.warmup_steps = 5
        if self.warmup_steps > 0:
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    return 0.1 + 0.9 * (float(current_step) / float(max(1, self.warmup_steps)))
                return 1.0
            
            self.scheduler = LambdaLR(self.opt, lr_lambda)
        else:
            self.scheduler = LambdaLR(self.opt, lambda _: 1.0)

        # criterion
        self.criterion = criterion

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # results folder
        self.results_folder = Path(results_folder) / self.dataset
        self.results_folder.mkdir(exist_ok=True)

        self.model_dict_folder = path('./results/PTB_XL', 'model_dicts')
        self.samples_folder = path(self.results_folder, 'samples')
        self.evaluation_folder = path(self.results_folder, 'evaluation')

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # model_status
        self.is_training = is_training

    @property
    def device(self):
        return self.accelerator.device

    def save_model(self, milestone=0, best=False):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        if not best:
            torch.save(data, str(self.model_dict_folder / f'model_{milestone}.pt'))
        else:
            torch.save(data, str(self.model_dict_folder / f'best_model_{self.lead}.pt'))

    def sample(self, dataloader):
        accelerator = self.accelerator
        device = accelerator.device
        self.ema.ema_model.eval()
        
        with torch.no_grad():
            all_samples = []
            targets = []
            labels = []
            lengths = []

            pbar = tqdm(dataloader, desc="Sampling", leave=True)
            for cond, target, ref, label, length in pbar:
                cond, target, ref, length = cond.to(device), target.to(device), ref.to(device), length.to(device)

                sample = self.ema.ema_model.sample(
                    condition=cond, 
                    reference=ref
                )
                all_samples.append(sample)
                targets.append(target)
                labels.append(label)
                lengths.append(length)
                
            self.save_samples(torch.cat(all_samples, dim=0), torch.cat(targets, dim=0), torch.cat(labels, dim=0), torch.cat(lengths, dim=0))

    def save_samples(self, samples, targets, labels, lengths=None):
        if not self.accelerator.is_local_main_process:
            return

        output = {
            'samples': samples,
            'target': targets,
            'lengths': lengths,
            'labels': labels
        }

        torch.save(output, str(self.samples_folder / f'samples_{self.lead}.pt'))

    def load(self, check_point, status='training'):
        accelerator = self.accelerator
        device = accelerator.device
        if check_point:
            data = torch.load(str(self.model_dict_folder / f'model_{check_point}.pt'), map_location=device, weights_only=True)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])
        else:
            data = torch.load(str(self.model_dict_folder / f'best_model_{self.lead}.pt'), map_location=device, weights_only=True)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

        if status == 'test':
            self.is_training = False

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def evaluate(self, dataloader, criterion):
        accelerator = self.accelerator
        device = accelerator.device
        self.ema.ema_model.eval()
        
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        with torch.no_grad():
            eva_loss = 0.
            channel_losses = torch.zeros(len(leads), device=device)
            lead_pcc = torch.zeros(len(leads), device=device)
            inference_time = 0.
            
            sample_mses = []
            sample_pccs = []

            pbar = tqdm(dataloader, desc="Evaluating")
            for i, (cond, target, ref) in enumerate(pbar):
                cond, target, ref = cond.to(device), target.to(device), ref.to(device)
                
                start_time = time.time()
                sample = self.ema.ema_model.sample(condition=cond, reference=ref)
                end_time = time.time()
                inference_time += end_time - start_time
                
                loss = criterion(sample, target)

                eva_loss += loss.item()
                for b in range(sample.size(0)):
                    sample_mse = torch.mean((sample[b] - target[b]) ** 2).item()
                    sample_mses.append(sample_mse)
                    sample_pcc = torch.mean(torch.stack([calculate_pcc(sample[b, ch, :], target[b, ch, :]) for ch in range(12)]))
                    sample_pccs.append(sample_pcc.item())

                for ch in range(len(leads)):
                    ch_loss = criterion(sample[:, ch:ch+1], target[:, ch:ch+1])
                    channel_losses[ch] += ch_loss.item()
                    
                    pcc_values = calculate_pcc(sample[:, ch], target[:, ch])
                    lead_pcc[ch] += torch.mean(pcc_values).item()
                
                pbar.set_description(f'val_loss: {eva_loss / (pbar.n + 1):.4f}')
                
            eva_loss /= len(dataloader)
            channel_losses /= len(dataloader)
            lead_pcc /= len(dataloader)
            avg_pcc = torch.mean(lead_pcc).item()
            avg_inference_time = inference_time / len(dataloader)
            
            if self.is_training:
                with open(self.evaluation_folder / f'val_loss_{self.lead}.txt', 'a') as f:
                    f.write('-' * 35 + '\n')
                    f.write(f'Steps: {self.step + 1}\n')
                    f.write(f'Average loss: {eva_loss:.4f}\n')
                    f.write(f'Average PCC: {avg_pcc:.4f}\n')
                    f.write(f'Average inference time per batch: {avg_inference_time:.4f} seconds\n')
                    for lead, ch_loss, ch_pcc in zip(leads, channel_losses, lead_pcc):
                        f.write(f'{lead}: Loss={ch_loss:.4f}, PCC={ch_pcc:.4f}\n') 
                    f.write('-' * 35 + '\n')
                    return eva_loss

            else:
                with open(self.evaluation_folder / f'test_format_{self.lead}.txt', 'a') as f:
                    f.write('-' * 35 + '\n')
                    f.write(f'Time: {datetime.datetime.now()}\n')
                    f.write(f'Average loss: {eva_loss:.4f}\n')
                    f.write(f'Average PCC: {avg_pcc:.4f}\n')
                    for lead, ch_loss, ch_pcc in zip(leads, channel_losses, lead_pcc):
                        f.write(f'{lead}: Loss={ch_loss:.4f}, PCC={ch_pcc:.4f}\n') 
                    f.write('-' * 35 + '\n')
                
                # mse_data = {'Ours': sample_mses}
                # with open('/root/autodl-tmp/DDPM/Statistic_MSE/Ours.json', 'w') as f:
                #     json.dump(mse_data, f)
                    
                # pcc_data = {'Ours': sample_pccs}
                # with open('/root/autodl-tmp/DDPM/Statistic_PCC/Ours.json', 'w') as f:
                #     json.dump(pcc_data, f)
                    
                return eva_loss
                    

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        loss_list = []
        
        batch_accum_count = 0
        running_loss = 0.0
        num_batches = len(self.train_loader)
        best_eva_loss = float('inf')
        milestone = 0
        
        with tqdm(total=self.train_num_steps, initial=self.step, disable=not accelerator.is_main_process) as pbar:
            for _ in range(self.train_num_steps):
                if self.step >= self.train_num_steps:
                    break
                    
                self.model.train()
                
                with tqdm(total=num_batches, desc=f"Step {self.step+1}/{self.train_num_steps}", leave=False) as batch_pbar:
                    for cond, target, ref in self.train_loader:
                        cond, target, ref = cond.to(device), target.to(device), ref.to(device)
                        
                        with self.accelerator.autocast():
                            loss = self.model(target, cond, ref)
                            loss = loss / self.gradient_accumulate_every
                            running_loss += loss.item()
                        
                        self.accelerator.backward(loss)
                        
                        batch_accum_count += 1
                        
                        if batch_accum_count % self.gradient_accumulate_every == 0:
                            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            
                            self.opt.step()
                            self.opt.zero_grad()
                            
                            loss_list.append(running_loss)
                            pbar.set_description(f'train_loss: {running_loss:.4f}')
                            running_loss = 0.0
                            
                            if accelerator.is_main_process:
                                self.ema.update()
                        
                        batch_pbar.update(1)
                        
                if (self.step + 1) >= 30 and (self.step + 1) % self.save_and_sample_every == 0:
                # if (self.step + 1) % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()
                    
                    with torch.no_grad():
                        val_loss = self.evaluate(self.val, self.criterion)
                        
                        if val_loss < best_eva_loss:
                            best_eva_loss = val_loss
                            self.save_model(best=True)

                self.step += 1
                self.scheduler.step()
                
                pbar.update(1)
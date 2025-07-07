import torch
from thop import profile, clever_format
from dual_unet import DualUNet

def count_model_stats():
    model = DualUNet(in_channels=1, out_channels=12, seq_length=1024)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    model.eval()
    
    lead_I = torch.randn(1, 1, 1024)
    
    macs, _ = profile(model, inputs=(lead_I,), verbose=False)
    flops = macs * 2
    
    flops_str = clever_format([flops], "%.3f")
    print(f"Inference FLOPs: {flops_str}")

if __name__ == "__main__":
    count_model_stats()
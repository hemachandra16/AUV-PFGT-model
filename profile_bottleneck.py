import time
import torch
from data.dataset import UIEBDataset
from torch.utils.data import DataLoader
from models.build import build_model
from models.loss import PFGTLoss
from torch.amp import autocast, GradScaler
import yaml
from pathlib import Path
import os

def profile_step():
    device = torch.device("cuda")
    model = build_model(device=device)
    criterion = PFGTLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    # Create dataset
    # Look at train.yaml for root dir
    config_path = "configs/train.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ds_cfg = cfg.get("dataset", {})
    ds = UIEBDataset(
        root_dir=ds_cfg.get("root_dir"),
        raw_dir=ds_cfg.get("raw_dir"),
        reference_dir=ds_cfg.get("reference_dir"),
        image_size=ds_cfg.get("image_size", 256)
    )
    dl = DataLoader(ds, batch_size=4, num_workers=4, pin_memory=True, drop_last=True)
    
    dl_iter = iter(dl)
    
    # Warmup
    for _ in range(2):
        inputs, targets = next(dl_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)["total_loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Profile
    torch.cuda.synchronize()
    
    times = {'data': 0, 'fwd': 0, 'bwd': 0, 'opt': 0}
    num_steps = 5
    
    for i in range(num_steps):
        t0 = time.time()
        inputs, targets = next(dl_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()
        
        with autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)["total_loss"]
        torch.cuda.synchronize()
        t2 = time.time()
        
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t3 = time.time()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t4 = time.time()
        
        times['data'] += t1 - t0
        times['fwd'] += t2 - t1
        times['bwd'] += t3 - t2
        times['opt'] += t4 - t3

    print(f"Average times over {num_steps} steps:")
    print(f"Data loading: {times['data']/num_steps:.4f} s")
    print(f"Forward pass: {times['fwd']/num_steps:.4f} s")
    print(f"Backward pass: {times['bwd']/num_steps:.4f} s")
    print(f"Optimizer:    {times['opt']/num_steps:.4f} s")
    total = sum(times.values())/num_steps
    print(f"Total step:   {total:.4f} s")

if __name__ == "__main__":
    profile_step()

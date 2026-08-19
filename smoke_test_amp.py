import torch
from models.model import PFGTUIEModel
from torch.amp import autocast, GradScaler

def run_smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on {device}...")
    model = PFGTUIEModel().to(device)
    
    print("Creating dummy input...")
    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    dummy_target = torch.randn(2, 3, 256, 256, device=device)
    
    scaler = GradScaler(device=device, enabled=(device=="cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    print("Running forward pass with autocast...")
    with autocast(device_type=device, enabled=(device=="cuda")):
        output = model(dummy_input)
        loss = torch.nn.functional.l1_loss(output, dummy_target)
        
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item()}")
    
    print("Running backward pass...")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print("Smoke test with AMP passed! No errors.")

if __name__ == "__main__":
    run_smoke_test()

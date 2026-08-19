import torch
from models.model import PFGTUIEModel

def run_smoke_test():
    print("Initializing model...")
    model = PFGTUIEModel().cuda()
    print("Model moved to GPU.")
    
    print("Creating dummy input...")
    dummy_input = torch.randn(2, 3, 256, 256, device="cuda")
    
    print("Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 3, 256, 256), f"Expected shape (2, 3, 256, 256), got {output.shape}"
    print("Smoke test passed! No CUDA errors.")

if __name__ == "__main__":
    run_smoke_test()

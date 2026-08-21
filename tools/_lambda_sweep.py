"""Pick lambda_dc by measuring where L_dc's gradient dominates AT the module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from models.loss import PFGTLoss
from utils.checkpoint import load_checkpoint

dev='cuda'
_, val = get_splits(augment_train=False)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()
pp = list(m.global_correction.predictor.parameters())
crit = PFGTLoss(lambda_l1=1.0,lambda_ssim=0.5,lambda_perceptual=0.1,lambda_frequency=0.0,lambda_dc=1.0).to(dev)

x,y = next(iter(DataLoader(val,batch_size=8,shuffle=False,num_workers=0)))
x,y = x.to(dev), y.to(dev)
out = m(x); L = crit(out,y)

def gnorm(loss):
    m.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    return float(sum((p.grad.detach()**2).sum() for p in pp if p.grad is not None)**0.5)

g_dc_unit = gnorm(L["dc_loss"])          # gradient from L_dc at weight 1.0
g_old     = gnorm(1.0*L["l1_loss"] + 0.5*L["ssim_loss"] + 0.1*L["perceptual_loss"])
m.zero_grad(set_to_none=True)

print(f"||grad|| at global_correction.predictor")
print(f"  from L_dc at weight 1.0            : {g_dc_unit:.6e}")
print(f"  from L1 + SSIM + perceptual        : {g_old:.6e}\n")
print(f"{'lambda_dc':>10}{'ratio vs others':>18}{'% of objective':>17}")
print('-'*46)
dc_raw=float(L["dc_loss"]); base_tot=1.0*float(L["l1_loss"])+0.5*float(L["ssim_loss"])+0.1*float(L["perceptual_loss"])
for lam in (0.3,0.5,0.7,1.0,1.5,2.0):
    r=lam*g_dc_unit/g_old
    w=lam*dc_raw
    print(f"{lam:>10.1f}{r:>17.2f}x{w/(base_tot+w)*100:>16.1f}%")
print(f"\nparity (ratio 1.0) at lambda_dc = {g_old/g_dc_unit:.2f}")

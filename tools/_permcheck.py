"""Independently verify the claim that the transformer block is permutation-equivariant.

If true, the block has no notion of WHERE a token sits or who its neighbours are. That would
make the high-frequency branch (whose documented job in docs/architecture.md Module 4 is
edges / texture / oriented detail) architecturally incapable of that job, since edges are
defined by spatial arrangement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.transformer_block import TransformerBlock

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, N, C = 2, 64, 128          # 64 tokens = an 8x8 grid
blk = TransformerBlock(embed_dim=C, num_heads=4).to(dev).eval()

x  = torch.randn(B, N, C, device=dev)
pf = torch.randn(B, 64, 16, 16, device=dev)

perm = torch.randperm(N, device=dev)
g = int(N ** 0.5)
# permute the physics grid the SAME way, so the comparison is fair
pf_flat = torch.nn.functional.adaptive_avg_pool2d(pf, (g, g)).flatten(2)   # (B,64,N)
pf_perm = pf_flat[:, :, perm].reshape(B, 64, g, g)

with torch.no_grad():
    y_plain = blk(x, torch.nn.functional.adaptive_avg_pool2d(pf, (g, g)))
    y_perm  = blk(x[:, perm, :], pf_perm)

diff = (y_perm - y_plain[:, perm, :]).abs().max().item()
scale = y_plain.abs().max().item()
print(f"max |block(P(x), P(pf)) - P(block(x, pf))| = {diff:.3e}")
print(f"output scale max|y|                        = {scale:.3f}")
print(f"relative                                   = {diff/scale:.3e}")
print(f"-> permutation-equivariant: {diff/scale < 1e-5}")

# Adjacent-swap: does the block distinguish neighbours at all?
sw = torch.arange(N, device=dev); sw[0], sw[1] = 1, 0
pf_sw = pf_flat[:, :, sw].reshape(B, 64, g, g)
with torch.no_grad():
    y_sw = blk(x[:, sw, :], pf_sw)
d2 = (y_sw - y_plain[:, sw, :]).abs().max().item()
print(f"\nadjacent-token swap diff = {d2:.3e}  -> no locality: {d2/scale < 1e-5}")

# Confirm no positional encoding exists anywhere
import subprocess
r = subprocess.run(['grep','-rniE','pos_embed|positional|position_emb|PositionalEncoding|pos_enc',
                    'models','train.py'], capture_output=True, text=True)
print(f"\npositional-encoding references in models/ + train.py: "
      f"{len([l for l in r.stdout.splitlines() if l.strip()])}")

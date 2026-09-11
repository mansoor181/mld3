"""Where does each corpus keep its positional correlation, near or far?

SMILES has grammar (ring-closure digits, matching brackets) that couples distant positions.
Regulatory DNA has local motifs and little long-range syntax. If that is the reason the mixture
pays off on molecules and not on DNA, the data's pairwise MI should fall off very differently
with separation.
"""
from pathlib import Path
import sys, torch
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
from eval.eval_text import domain_of, loader_module

def pmi(x, V, pairs):
    N = x.shape[0]; tot = 0.0
    for i, j in pairs:
        a, b = x[:, i], x[:, j]
        J = torch.zeros(V, V, device=x.device)
        J.index_put_((a, b), torch.ones(N, device=x.device), accumulate=True)
        J /= J.sum()
        pi = J.sum(1, keepdim=True); pj = J.sum(0, keepdim=True)
        nz = J > 0
        tot += float((J[nz] * (J[nz].log() - (pi @ pj)[nz].clamp_min(1e-12).log())).sum())
    return tot / max(len(pairs), 1)

dev = torch.device("cuda")
for ds, V, L in [("qm9", 40, 32), ("zinc250k", 72, 74), ("deepstarr", 6, 249)]:
    loader, _ = loader_module(domain_of(ds)).make_loader(ds, "val", 256, shuffle=False, num_workers=2)
    xs = []
    for xb in loader:
        xb = xb[0] if isinstance(xb, (list, tuple)) else xb
        xs.append(xb)
        if sum(r.shape[0] for r in xs) >= 4096: break
    x = torch.cat(xs)[:4096].to(dev)
    out = []
    for d in (1, 2, 5, 10, 30):
        if d >= L: out.append(None); continue
        pairs = [(i, i + d) for i in range(0, L - d)]
        out.append(pmi(x, V, pairs))
    print("%-10s L=%-4d " % (ds, L) + "  ".join(
        "sep%-3d=%s" % (d, ("%.5f" % v) if v is not None else "  -  ") for d, v in zip((1,2,5,10,30), out)))

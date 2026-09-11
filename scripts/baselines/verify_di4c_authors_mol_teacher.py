"""Confirm the converted molecule teacher loads into the authors' DiTOrig with no gaps.

self.teacher[0].load_state_dict(..., strict=False) in multi_round_sdtt would hide a total
mismatch, so we reproduce the same call here and assert that nothing is missing and nothing
is left over.
"""
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
sys.path.insert(0, str(paths.BASELINES_DIR / "di4c" / "sdtt" / "src"))
from omegaconf import OmegaConf
from sdtt.models.loading_utils import get_backbone

ds, vocab, length = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = OmegaConf.create({
    "model": {"name": "small", "type": "ddit-orig", "hidden_size": 768, "cond_dim": 128,
              "length": length, "n_blocks": 12, "n_heads": 12, "scale_by_sigma": True,
              "dropout": 0.1, "tie_word_embeddings": False},
    "time_conditioning": True,
})
net = get_backbone(cfg, vocab_size=vocab)
ck = torch.load(str(paths.RESULTS_ROOT / ("di4c_authors_mol/teachers/%s_mdlm_teacher.ckpt" % ds)),
                map_location="cpu", weights_only=True)["state_dict"]
ck = {k.replace("backbone.", ""): v for k, v in ck.items()}
res = net.load_state_dict(ck, strict=False)
print("== %s (vocab %d, length %d)" % (ds, vocab, length))
print("   model params : %d" % len(net.state_dict()))
print("   ckpt tensors : %d" % len(ck))
print("   MISSING keys : %d %s" % (len(res.missing_keys), res.missing_keys[:4]))
print("   UNEXPECTED   : %d %s" % (len(res.unexpected_keys), res.unexpected_keys[:4]))
ok = not res.missing_keys and not res.unexpected_keys
print("   VERDICT      : %s" % ("CLEAN LOAD" if ok else "MISMATCH"))

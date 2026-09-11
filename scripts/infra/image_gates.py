"""Kill gates for the ImageNet-256 image arm, run before any trainer code is written.

The token dump we train on (`Jiayi-Pan/imagenet-maskgit-256-vqgan`) and the MaskGIT teacher we
distil from (`llvictorll/Maskgit-pytorch`) were published by different people. Codebook indices are
arbitrary labels, so if the two were produced by different VQGANs then everything downstream loads
cleanly, trains without complaint, and generates noise. Nothing in the training loop can detect it.
These gates detect it in minutes rather than after a GPU-day.

  gate1  decode 16 val rows through the VQGAN and write a PNG grid. Photographs or stop.
  gate2  classify 1000 decoded rows with InceptionV3 and compare against the stored label column.
         This separates a codebook mismatch (gate 1 already failed) from a label permutation.
  gate3  load the MaskGIT checkpoint at heads=16 and at heads=8 and see which one denoises.
         nn.MultiheadAttention stores in_proj_weight as [3d, d] regardless of head count, so the
         wrong value passes load_state_dict in silence and computes garbage.
  gate4  teacher NELBO on val rows at 50 percent masking. Chance is ln(1024) = 6.93 nats.

Usage:
  python3.10 scripts/image_gates.py --gate 1
  python3.10 scripts/image_gates.py --gate all
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.utils as vutils
from omegaconf import OmegaConf
from torchvision.models import Inception_V3_Weights, inception_v3

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

import paths
from utils.vendor_import import load_file_module

# The Taming VQGAN comes from the vendored ReDi image tree, appended rather than prepended so
# that its top-level modules cannot shadow ours.
sys.path.append(paths.REDI_IMAGE_DIR)
from Network.Taming.models.vqgan import VQModel

ROOT = paths.IMAGE_ROOT
MASKGIT_DIR = ROOT / "maskgit"
HF_DIR = ROOT / "imagenet256" / "_hf"
OUT = ROOT / "gates"
DI4C_MG = Path(paths.MASKGIT_DIR)

CODEBOOK, MASK_ID, GRID = 1024, 1024, 16


def _dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_val_tokens(n: int, spread: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens [n,16,16] int64, labels [n] int64) from the val parquet shard.

    The 50k val rows are sorted by class, fifty to a class, so the head of the file is one
    single class. Reading the head is what we want for a quick eyeball, because sixteen images
    of the same thing make a codebook mismatch obvious. Any measurement against the label
    column has to set spread and stride the whole file instead, or it scores one class and
    reports a number that means nothing.
    """

    f = sorted(glob.glob(str(HF_DIR / "**" / "val-*.parquet"), recursive=True))
    assert f, f"no val parquet under {HF_DIR}"
    pf = pq.ParquetFile(f[0])
    if spread:
        tbl = pf.read().take(torch.arange(0, pf.metadata.num_rows,
                                          pf.metadata.num_rows // n)[:n].tolist()).to_pylist()
    else:
        tbl = pf.read_row_group(0).slice(0, n).to_pylist()
    tok = torch.tensor([r["token"] for r in tbl], dtype=torch.int64)
    lab = torch.tensor([r["label"] for r in tbl], dtype=torch.int64)
    assert tok.shape[1:] == (GRID, GRID), tok.shape
    assert int(tok.min()) >= 0 and int(tok.max()) < CODEBOOK, (int(tok.min()), int(tok.max()))
    return tok, lab


def build_vqgan(device: torch.device):
    cfg = OmegaConf.load(str(MASKGIT_DIR / "model.yaml"))
    vq = VQModel(**cfg.model.params)
    sd = torch.load(str(MASKGIT_DIR / "last.ckpt"), map_location="cpu")["state_dict"]
    # The checkpoint carries the discriminator and the LPIPS network, neither of which VQModel
    # defines, hence strict=False is correct here. We still report what was missing, because a
    # missing ENCODER or DECODER key would mean the config and the checkpoint disagree.
    miss, unexp = vq.load_state_dict(sd, strict=False)
    hard = [k for k in miss if k.startswith(("encoder.", "decoder.", "quantize.",
                                             "quant_conv.", "post_quant_conv."))]
    assert not hard, f"VQGAN is missing core weights: {hard[:8]}"
    print(f"[vq] loaded, {len(miss)} missing (all loss-side), {len(unexp)} unexpected")
    return vq.eval().to(device).requires_grad_(False)


@torch.no_grad()
def decode_uint8(vq, tok: torch.Tensor, device: torch.device) -> torch.Tensor:
    """[B,16,16] int64 codes -> [B,3,256,256] uint8, the chain used at sample_and_eval.py:167."""
    img = vq.decode_code(tok.to(device).clamp(0, CODEBOOK - 1))
    img = img.float().clamp(-1.0, 1.0) * 0.5 + 0.5
    return (img * 255).round().clamp(0, 255).to(torch.uint8).cpu()


# ----------------------------------------------------------------------------- gate 1
def gate1(n: int = 16) -> None:

    device = _dev()
    tok, lab = load_val_tokens(n)
    vq = build_vqgan(device)
    imgs = decode_uint8(vq, tok, device)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "gate1_decoded_val.png"
    vutils.save_image(imgs.float() / 255.0, str(p), nrow=4, padding=2)
    print(f"[gate1] wrote {p}")
    print(f"[gate1] labels {lab.tolist()}")
    # A decoded photograph has strong spatial structure, so neighbouring pixels correlate. Uniform
    # codebook noise does not. This is a numeric companion to looking at the grid, not a substitute.
    x = imgs.float() / 255.0
    tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean().item()
    print(f"[gate1] mean vertical |grad| = {tv:.4f}  (photographs ~0.02-0.06, noise >0.15)")


# ----------------------------------------------------------------------------- gate 2
def gate2(n: int = 1000, batch: int = 50) -> None:

    device = _dev()
    tok, lab = load_val_tokens(n, spread=True)
    print(f"[gate2] {len(set(lab.tolist()))} distinct classes in the sample")
    vq = build_vqgan(device)
    net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    top1 = top5 = 0
    with torch.no_grad():
        for i in range(0, n, batch):
            im = decode_uint8(vq, tok[i:i + batch], device).to(device).float() / 255.0
            im = torch.nn.functional.interpolate(im, size=(299, 299), mode="bilinear",
                                                 align_corners=False)
            pred = net((im - mean) / std).topk(5, dim=-1).indices.cpu()
            y = lab[i:i + batch, None]
            top1 += int((pred[:, :1] == y).any(1).sum())
            top5 += int((pred == y).any(1).sum())
    print(f"[gate2] n={n}  top1={top1 / n:.1%}  top5={top5 / n:.1%}")
    print("[gate2] PASS" if top5 / n > 0.50 else "[gate2] FAIL, tokens and labels disagree")


# ----------------------------------------------------------------------------- gate 3 helpers
def build_maskgit(heads: int, device: torch.device, strict_report: bool = True):
    # By file path, not via sys.path. The ReDi image tree carries a rival top-level `Network`
    # package, and once its VQGAN has been imported, which the FID code does, a plain
    # `from Network.transformer import ...` hands back ReDi's MaskTransformer instead. That one
    # takes a required `randmask` argument, so the failure is loud here but would not have to be.
    MaskTransformer = load_file_module(
        "_di4c_maskgit_transformer", DI4C_MG / "Network" / "transformer.py").MaskTransformer

    net = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=CODEBOOK, depth=24,
                          heads=heads, mlp_dim=3072, dropout=0.0, nclass=1000)
    ck = torch.load(str(MASKGIT_DIR / "MaskGIT_ImageNet_256.pth"), map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    miss, unexp = net.load_state_dict(sd, strict=False)
    if strict_report:
        print(f"[maskgit] heads={heads}  missing={len(miss)} unexpected={len(unexp)}")
        if miss:
            print(f"[maskgit]   missing e.g. {list(miss)[:5]}")
        if unexp:
            print(f"[maskgit]   unexpected e.g. {list(unexp)[:5]}")
    return net.eval().to(device).requires_grad_(False)


@torch.no_grad()
def denoise_score(net, tok: torch.Tensor, device: torch.device, keep: float = 0.5,
                  seed: int = 0, lab: torch.Tensor | None = None) -> tuple[float, float]:
    """Mask a fraction of positions and score the teacher on the held-out truth.

    Passing lab conditions the teacher on the true class, and leaving it None drops the label
    to the null class 2025. MaskGIT was trained class-conditionally with the null class held
    out at ten percent, so the two numbers differ a lot and the gap is what decides whether
    the image arm can run its teacher unconditionally.

    Returns (nats per masked token, top-1 accuracy on masked positions). Chance is ln(1024)=6.93.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = tok.shape[0]
    flat = tok.view(B, -1).to(device)
    m = (torch.rand(flat.shape, generator=g) > keep).to(device)          # True = masked
    x = torch.where(m, torch.full_like(flat, MASK_ID), flat)
    y = (torch.zeros(B, dtype=torch.long, device=device) if lab is None
         else lab.to(device).long())
    drop = torch.full((B,), lab is None, dtype=torch.bool, device=device)
    logits = net(x.view(B, GRID, GRID), y=y, drop_label=drop).float()    # [B,256,1025]
    lp = torch.log_softmax(logits, dim=-1)
    tgt = lp.gather(-1, flat[..., None]).squeeze(-1)                     # [B,256]
    nats = float(-(tgt[m]).mean())
    acc = float((logits.argmax(-1)[m] == flat[m]).float().mean())
    return nats, acc


def gate3(n: int = 32) -> None:
    device = _dev()
    tok, _ = load_val_tokens(n)
    print(f"[gate3] chance = ln(1024) = {np.log(CODEBOOK):.3f} nats, acc = {1/CODEBOOK:.4f}")
    for h in (16, 8, 12):
        try:
            net = build_maskgit(h, device)
            nats, acc = denoise_score(net, tok, device)
            print(f"[gate3] heads={h:2d}  nats/masked-token={nats:6.3f}  top1={acc:.3f}")
            del net
            torch.cuda.empty_cache()
        except Exception as e:                                    # noqa: BLE001
            print(f"[gate3] heads={h}: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- gate 4
def gate4(n: int = 512, heads: int = 16, batch: int = 64) -> None:
    """Teacher NELBO across masking rates, with and without the true class label.

    The sweep over keep is what separates a broken correspondence from a teacher that is simply
    working hard. A correct teacher gets steadily better as more context survives, and a broken
    one sits flat at chance no matter how much of the image it is shown.
    """
    device = _dev()
    tok, lab = load_val_tokens(n, spread=True)
    net = build_maskgit(heads, device, strict_report=False)
    print(f"[gate4] n={n} heads={heads}, chance {np.log(CODEBOOK):.3f} nats / {1/CODEBOOK:.4f} top1")
    best = 9.9
    for keep in (0.1, 0.3, 0.5, 0.7, 0.9):
        row = []
        for cond in (False, True):
            vals, accs = [], []
            for i in range(0, n, batch):
                v, a = denoise_score(net, tok[i:i + batch], device, keep=keep, seed=i,
                                     lab=lab[i:i + batch] if cond else None)
                vals.append(v)
                accs.append(a)
            row.append((float(np.mean(vals)), float(np.mean(accs))))
        best = min(best, row[0][0])
        print(f"[gate4] keep={keep:.1f}  null-label {row[0][0]:5.3f} nats / {row[0][1]:.3f} top1"
              f"   true-label {row[1][0]:5.3f} nats / {row[1][1]:.3f} top1"
              f"   gap {row[0][0] - row[1][0]:+.3f}")
    print("[gate4] PASS" if best < 6.0 else "[gate4] FAIL, codebook correspondence is wrong")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="1")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    todo = ["1", "2", "3", "4"] if args.gate == "all" else [args.gate]
    for g in todo:
        print(f"\n{'=' * 20} gate {g} {'=' * 20}")
        fn = {"1": gate1, "2": gate2, "3": gate3, "4": gate4}[g]
        fn(**({"n": args.n} if args.n else {}))


if __name__ == "__main__":
    main()

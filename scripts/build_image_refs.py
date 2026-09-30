"""Build the two Inception feature caches the ImageNet-256 arm scores against.

`val_real` holds features of the 50,000 real ImageNet validation images and is the standard FID
reference. `val_recon` holds features of VQGAN reconstructions of the same 50,000 validation
token rows and gives the tokenizer floor, which is the best FID any model working in token
space could possibly reach. Reporting the floor alongside our numbers stops a reader from
taking a tokenizer ceiling for a modelling failure.

We cache the features rather than the FID moments because `MultiInceptionMetrics.compute`
concatenates the real features unconditionally at `inception_metrics.py:406` even when moments
are supplied, and because precision and recall need the individual features in any case. Each
cache is 50000 x 2048 float32, which is 410 MB.

Usage:
  python scripts/build_image_refs.py                # both caches
  python scripts/build_image_refs.py --which recon  # just the tokenizer floor
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import glob
import io

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

import data
import paths
from metrics import image_metrics as IM


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
import argparse
import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(REPO_DIR))
from metrics import image_metrics as IM              # noqa: E402

ROOT = paths.IMAGE_ROOT
PIXELS = ROOT / "imagenet_val_px"
RESOLUTION = 256


def _center_crop_256(img) -> np.ndarray:
    """Resize the short side to 256 and take the central 256x256, as RGB uint8 HWC.

    This is the standard ImageNet generation protocol. The corpus is already resized so that
    the short side is 256, but a handful of rows are not, and a silent off-size image would
    otherwise reach Inception at the wrong scale.
    """

    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) != RESOLUTION:
        scale = RESOLUTION / min(w, h)
        img = img.resize((max(RESOLUTION, round(w * scale)),
                          max(RESOLUTION, round(h * scale))), Image.BICUBIC)
        w, h = img.size
    left, top = (w - RESOLUTION) // 2, (h - RESOLUTION) // 2
    return np.asarray(img.crop((left, top, left + RESOLUTION, top + RESOLUTION)), dtype=np.uint8)


def build_real(metric, device: torch.device, batch: int = 64, limit: int = 0) -> np.ndarray:

    files = sorted(glob.glob(str(PIXELS / "**" / "val-*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no val parquet under {PIXELS}")
    feats, buf, seen = [], [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for rec in pf.read_row_group(g, columns=["image"]).column("image").to_pylist():
                buf.append(_center_crop_256(Image.open(io.BytesIO(rec["bytes"]))))
                if len(buf) == batch:
                    feats.append(_feat(metric, buf, device))
                    seen += len(buf)
                    buf = []
                    if seen % 5000 == 0:
                        print(f"[refs] real {seen}", flush=True)
                    if limit and seen >= limit:
                        return np.concatenate(feats)
    if buf:
        feats.append(_feat(metric, buf, device))
        seen += len(buf)
    print(f"[refs] real {seen} images")
    return np.concatenate(feats)


def _feat(metric, images: list[np.ndarray], device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().to(device)
    return IM.inception_features(metric, x).float().cpu().numpy()


def build_recon(metric, device: torch.device, batch: int = 64, limit: int = 0) -> np.ndarray:
    sys.path.insert(0, str(REPO_DIR))

    ds, meta = data.load("imagenet256", "val")
    n = limit or len(ds)
    tok = ds.x[:n]
    vq = IM.build_vqgan(device)
    feats = []
    for i, block in enumerate(IM.tokens_to_uint8(tok, vq, device, chunk=batch,
                                                 grid=meta["grid"])):
        feats.append(IM.inception_features(metric, block).float().cpu().numpy())
        if (i + 1) * batch % 5000 < batch:
            print(f"[refs] recon {(i + 1) * batch}", flush=True)
    print(f"[refs] recon {n} images")
    return np.concatenate(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "real", "recon"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IM.REF_DIR.mkdir(parents=True, exist_ok=True)
    metric = IM.build_metric(device)
    todo = ["real", "recon"] if args.which == "all" else [args.which]
    for name in todo:
        out = IM.reference_path(f"val_{name}")
        if out.exists():
            print(f"[refs] have {out}")
            continue
        fn = build_real if name == "real" else build_recon
        feats = fn(metric, device, batch=args.batch, limit=args.limit)
        np.savez(out, features=feats.astype(np.float32))
        print(f"[refs] wrote {out}  {feats.shape}  {out.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()

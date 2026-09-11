"""Fetch the MaskGIT teacher, its VQGAN and the pre-tokenized ImageNet corpus for the image arm.

Everything lands under MLDF_IMAGE_ROOT. The download is roughly 5 GB, and
/software turned out to be a read-only ceph mount, so data2 is the only writable volume on this box
with room for the 2.09 GB teacher, the 958 MB VQGAN and the token dump.

We deliberately skip MaskGIT_ImageNet_512.pth, which the vendored download_models.py also pulls. Our
arm is 256x256 only and the 512 checkpoint is another 2 GB we would never open.

Usage:
  python3.10 scripts/fetch_image_arm.py            # fetch everything that is missing
  python3.10 scripts/fetch_image_arm.py --what vq  # just the VQGAN
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
from huggingface_hub import hf_hub_download, snapshot_download

import argparse
import os
from pathlib import Path

ROOT = paths.IMAGE_ROOT
MASKGIT_REPO = "llvictorll/Maskgit-pytorch"
TOKENS_REPO = "Jiayi-Pan/imagenet-maskgit-256-vqgan"

# The HF cache and the destination sit on the same filesystem so that hf_hub_download can hardlink
# the blob into place instead of copying it. A cache on root would both fill root and double the
# space used, because a cross-device link silently degrades to a copy.
# HF_HOME comes from the environment (.env) so the download lands in the shared cache.
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_HOME"] + "/datasets")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def fetch_maskgit() -> None:

    dest = ROOT / "maskgit"
    dest.mkdir(parents=True, exist_ok=True)
    for fn in ("pretrained_maskgit/VQGAN/last.ckpt",
               "pretrained_maskgit/VQGAN/model.yaml",
               "pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth"):
        out = dest / Path(fn).name
        if out.exists():
            print(f"[fetch] have {out} ({out.stat().st_size / 2**20:.0f} MB)")
            continue
        print(f"[fetch] {MASKGIT_REPO}:{fn}")
        p = hf_hub_download(repo_id=MASKGIT_REPO, filename=fn, local_dir=str(dest / "_raw"))
        Path(p).replace(out)
        print(f"[fetch] wrote {out} ({out.stat().st_size / 2**20:.0f} MB)")


def fetch_tokens() -> None:

    dest = ROOT / "imagenet256" / "_hf"
    if dest.exists() and any(dest.rglob("*.parquet")):
        print(f"[fetch] have token dump at {dest}")
        return
    print(f"[fetch] {TOKENS_REPO}")
    snapshot_download(repo_id=TOKENS_REPO, repo_type="dataset", local_dir=str(dest))
    n = sum(1 for _ in dest.rglob("*.parquet"))
    print(f"[fetch] wrote {n} parquet shards under {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all", "maskgit", "vq", "tokens"])
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.what in ("all", "maskgit", "vq"):
        fetch_maskgit()
    if args.what in ("all", "tokens"):
        fetch_tokens()
    print(f"[fetch] done, root={ROOT}")


if __name__ == "__main__":
    main()

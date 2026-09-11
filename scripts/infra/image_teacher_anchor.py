"""Score the frozen MaskGIT teacher under our own samplers, which anchors the image arm.

Every student is distilled from this teacher, hence the teacher's own FID at a given budget is
the ceiling the arm is measured against. A student that beat it would mean the adapter is wrong
rather than that distillation improved on its source, which is why this runs before any GPU day
is spent on training.

The number produced here is deliberately not MaskGIT's published FID. We decode with the plain
analytic absorbing reverse and draw every position independently, while MaskGIT's own decoder
adds confidence ranking, Gumbel noise and classifier-free guidance, all of which are worth many
FID points. The comparison the paper makes is student against student against this anchor, all
measured by this code on this reference set.

Usage:
  python3.10 scripts/image_teacher_anchor.py                      # NFE grid, 10k samples
  python3.10 scripts/image_teacher_anchor.py --nfe 8 --n 2000     # a quick single point
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
from metrics import image_metrics as IM
from models.teacher_adapters import build_maskgit_teacher
from sampling.sampler import sample_lkf_analytic, sample_lkf

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(REPO_DIR))
from metrics import image_metrics as IM                # noqa: E402
from models.teacher_adapters import build_maskgit_teacher  # noqa: E402
from sampling.sampler import sample_lkf_analytic, sample_lkf  # noqa: E402

ROOT = paths.IMAGE_ROOT
CKPT = ROOT / "maskgit" / "MaskGIT_ImageNet_256.pth"
SEQ_LEN = 256


@torch.no_grad()
def draw(teacher, nfe: int, n: int, batch: int, device, sampler: str) -> torch.Tensor:
    fn = sample_lkf_analytic if sampler == "analytic" else sample_lkf
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        # One class per row, held for the whole trajectory. Under `held` the teacher is the class
        # marginal, which is only an unconditional model if a rollout keeps a single class.
        if getattr(teacher, "label_mode", "null") == "held":
            teacher.resample_labels(b, device)
        out.append(fn(teacher, nfe, b, SEQ_LEN, device).cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfe", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--sampler", default="analytic", choices=["analytic", "ancestral"])
    ap.add_argument("--ref", default="val_real")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--label_mode", default="null", choices=["null", "random", "held"])
    ap.add_argument("--out", default=str(ROOT / "results" / "image_distill" /
                                         "teacher_anchor" / "eval.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, _ = build_maskgit_teacher(str(CKPT), device, seq_len=SEQ_LEN, bf16=args.bf16,
                                       label_mode=args.label_mode)
    vq = IM.build_vqgan(device)
    metric = IM.build_metric(device)
    ref = IM.load_reference(args.ref, device)
    print(f"[anchor] reference {args.ref} {tuple(ref.shape)}  sampler={args.sampler}  "
          f"n={args.n}", flush=True)

    rows = {}
    for nfe in args.nfe:
        t0 = time.time()
        toks = draw(teacher, nfe, args.n, args.batch, device, args.sampler)
        t1 = time.time()
        res = IM.score(metric, IM.tokens_to_uint8(toks, vq, device, chunk=args.chunk), ref)
        res["unique_frac"] = float(len(torch.unique(toks, dim=0)) / len(toks))
        rows[str(nfe)] = res
        print(f"[anchor] nfe={nfe:>3}  fid={res['fid']:7.2f}  is={res['inception_score']:6.2f}  "
              f"prec={res['precision']:.3f}  rec={res['recall']:.3f}  "
              f"uniq={res['unique_frac']:.3f}  sample {t1 - t0:.0f}s  "
              f"score {time.time() - t1:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cell": "teacher_anchor", "dataset": "imagenet256",
                               "sampler": args.sampler, "n_samples": args.n,
                               "label_mode": args.label_mode,
                               "reference": args.ref, "nfe": rows}, indent=2))
    print(f"[anchor] wrote {out}")


if __name__ == "__main__":
    main()

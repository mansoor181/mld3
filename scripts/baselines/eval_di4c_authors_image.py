"""Score a Di4C student trained by the authors' own MaskGIT code on our image judge.

The authors ship an evaluation of their own in `Metrics/sample_and_eval.py`, and it builds its
reference from a different pass over the validation split than ours does. A number produced there
cannot be placed in a table beside ours, so we keep their sampler and replace only the judge.

Everything upstream of the metric is theirs. We call `MaskGIT.sample` unchanged, which means the
arccos schedule, the confidence gate, the linearly annealed Gumbel noise, the softmax temperature
and the classifier-free guidance ramp are all the published ones. The images it returns come out
of the same VQGAN decoder our own cells decode through, hence the only thing this file adds is the
rescale from the decoder's [-1, 1] range to uint8, which is copied verbatim from
`metrics/image_metrics.tokens_to_uint8`.

Two differences from our own cells are real and are reported rather than papered over. Their
student is class conditional and ours is not, so we draw a label uniformly over the 1000 classes
for every sample, which makes the generated class marginal match the validation set's. Their
student also carries classifier-free guidance, which ours has no analogue for, and we sweep the
guidance weight.

Setting the guidance weight to zero does not make the model unconditional. The released sampler
takes its `w == 0` branch to `self.vit(code, labels, drop_label=~drop)` with `drop` all true, hence
`drop_label` is all false and the true class token is still fed to the network. That setting removes
the classifier-free amplification and keeps the conditioning. The `--uncond` flag is the one that
matches our students, and it forces `drop_label` true so that the network reads the null class token
the authors' own drop-label rate of 0.1 trained it to accept.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths

TREE = Path(paths.MASKGIT_DIR)
sys.path.append(str(TREE))
from Trainer.vit import MaskGIT
from metrics import image_metrics as IMG


def build_args(ckpt_dir: str, vqgan: str, teacher: str, device):
    """Rebuild the argument namespace `MaskGIT.__init__` expects.

    `debug` is set so that the constructor skips `get_data`, because scoring never touches the
    training corpus and building the loader would read the whole token blob for nothing.
    """
    a = argparse.Namespace(
        data="imagenet_tokens", data_folder="", vqgan_folder=vqgan, vit_folder=ckpt_dir,
        writer_log="", sched_mode="arccos", grad_cum=1, channel=3, num_workers=0, step=4,
        seed=42, epoch=301, img_size=256, bsize=2, mask_value=1024, lr=1e-5, cfg_w=3,
        r_temp=4.5, sm_temp=1.0, drop_label=0.1, test_only=False, resume=True, debug=True,
        teacher_vit=teacher, is_student=True, randomize="linear", latent_bsize=32,
        r_delta=0.05, lfg=False, teacher_steps=8, alpha_t="sigmoid", alpha_const=0.1,
        max_iter=0, log_iter=10000, save_iter=2000,
        device=device, iter=0, global_epoch=0, is_master=True, is_multi_gpus=False,
    )
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="directory holding checkpoints/current.pth, with a trailing slash")
    ap.add_argument("--vqgan", default=str(paths.IMAGE_ROOT / "maskgit") + "/")
    ap.add_argument("--teacher", default=str(paths.IMAGE_ROOT / "maskgit" / "MaskGIT_ImageNet_256.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", default="1,2,3,4,5,6,7,8,16,32")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--cfg_w", type=float, default=3.0)
    ap.add_argument("--sm_temp", type=float, default=1.0)
    ap.add_argument("--image_ref", default="val_real")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uncond", action="store_true",
                    help="drop the class token so the model runs unconditionally, as ours do")
    args = ap.parse_args()
    if args.uncond and args.cfg_w != 0.0:
        ap.error("--uncond requires --cfg_w 0, because guidance needs a class to guide toward")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    os.chdir(TREE)
    model = MaskGIT(build_args(args.ckpt_dir, args.vqgan, args.teacher, device))
    step_at = int(model.args.iter)
    print(f"[di4c-eval] loaded a student trained to {step_at} steps", flush=True)

    if args.uncond:
        # Force the null class token on every call. We wrap the network rather than edit their
        # sampler, so the schedule, the confidence gate and the noise stay exactly as released and
        # the only thing that changes is what the class embedding reads.
        inner = model.vit.forward

        def uncond_forward(img_token, y=None, drop_label=None, **kw):
            drop = torch.ones(img_token.size(0), dtype=torch.bool, device=img_token.device)
            return inner(img_token, y, drop, **kw)

        model.vit.forward = uncond_forward
        print("[di4c-eval] class token dropped, the model is running unconditionally", flush=True)

    metric = IMG.build_metric(device)
    ref = IMG.load_reference(args.image_ref, device)
    print(f"[di4c-eval] reference {args.image_ref}: {tuple(ref.shape)}", flush=True)

    rows = []
    for nfe in [int(v) for v in args.nfe.split(",")]:
        t0 = time.time()

        def batches():
            got = 0
            while got < args.samples:
                b = min(args.batch, args.samples - got)
                labels = torch.randint(0, 1000, (b,), device=device)
                x, _, _ = model.sample(nb_sample=b, labels=labels, sm_temp=args.sm_temp,
                                       w=args.cfg_w, randomize="linear", r_temp=4.5,
                                       sched_mode="arccos", step=nfe, teacher=None)
                got += b
                # The same rescale `tokens_to_uint8` applies, so an image scored here is
                # bit-identical to one our own cells put in front of the judge.
                img = x.float().clamp(-1.0, 1.0) * 0.5 + 0.5
                yield (img * 255).round().clamp(0, 255).to(torch.uint8)

        res = IMG.score(metric, batches(), ref)
        dt = time.time() - t0
        res["sec_per_sample"] = dt / args.samples
        print(f"[di4c-eval] nfe={nfe:>4}  fid={res['fid']:.2f}  is={res['inception_score']:.2f}  "
              f"prec={res['precision']:.3f}  rec={res['recall']:.3f}  "
              f"{res['sec_per_sample']*1e3:.1f} ms/samp", flush=True)
        for k, v in res.items():
            rows.append({"metric": k, "nfe": nfe, "value": float(v)})

    payload = {
        "metrics": rows,
        "meta": {
            "cell": "di4c_authors_maskgit",
            "step": step_at,
            "samples": args.samples,
            "cfg_w": args.cfg_w,
            "sampler": "authors_maskgit_confidence",
            "image_ref": args.image_ref,
            "seed": args.seed,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[di4c-eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

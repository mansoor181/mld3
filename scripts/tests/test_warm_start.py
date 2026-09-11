"""The warm-started student reproduces its MDLM teacher on every single-component route.

The molecule and DNA students are trained for a tenth of the teacher's budget, which is only
sensible if they start from the teacher rather than from scratch. This test builds a small
MDLM DiT with random weights, copies it into an M-branch Latent-Kernel student through
`models.warm_start.warm_start_from_mdlm`, and checks that the two produce the same logits
for every branch. Any mistake in the adaLN packing order, the time embedding layout or the
block split shows up here as a large discrepancy rather than as a silently worse student.

Run:
    python -m tests.test_warm_start
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import omegaconf

from models.latent_kernel import LKFConfig, LatentKernelFlow
from models.teacher_adapters import _load_dit_class
from models.warm_start import warm_start_from_mdlm

DIM, DEPTH, HEADS, COND, VOCAB, SEQ, LAT_L = 64, 4, 4, 32, 11, 12, 2


def build_teacher(dit_source: str):
    DIT = _load_dit_class(dit_source)
    cfg = omegaconf.OmegaConf.create({
        "model": {"hidden_size": DIM, "cond_dim": COND, "length": SEQ, "n_blocks": DEPTH,
                  "n_heads": HEADS, "scale_by_sigma": True, "dropout": 0.0},
        "algo": {"causal_attention": False},
    })
    net = DIT(cfg, vocab_size=VOCAB)
    # The reference zero-inits the output projection and every adaLN map, which would make the
    # test pass on a trunk that ignored them entirely. We randomise them instead.
    for p in net.parameters():
        if p.dim() >= 1:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
    net.eval()
    return net


def build_student(M: int) -> LatentKernelFlow:
    cfg = LKFConfig(
        vocab_size=VOCAB, seq_len=SEQ, dim=DIM, depth=DEPTH, heads=HEADS, mlp_mult=4,
        latent_M=M, latent_last_L=LAT_L, time_embed_dim=256, tie_embeddings=False,
        trunk_arch="mdlm_dit", cond_dim=COND, dropout=0.0,
        cond_extra_silu=False, time_conditioning=False,
        mask_id=VOCAB - 1, use_mask_prior=True, objective="x0",
    )
    model = LatentKernelFlow(cfg)
    model.eval()
    return model


def main() -> None:
    torch.manual_seed(0)
    ok = True
    # MDLM's own DiT calls a Triton rotary kernel that has no CPU backend, so the test runs on
    # the GPU whenever one is visible and falls back to the CPU-safe PairFlow tree otherwise.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = ("mdlm", "pairflow") if dev.type == "cuda" else ("pairflow",)
    # Both trunks run their blocks under bfloat16 autocast on the GPU, which caps the agreement
    # we can ask for there at roughly the bfloat16 resolution of the logits.
    tol = 5e-2 if dev.type == "cuda" else 2e-4
    ran = 0
    for source, M in ((s, m) for s in sources for m in (3, 4, 8)):
        try:
            net = build_teacher(source).to(dev)
        except Exception as exc:                      # pragma: no cover - optional tree
            print(f"[warm-start-test] skipping {source}: {exc}")
            continue
        ran += 1
        student = build_student(M).to(dev)
        warm_start_from_mdlm(student, net, k_noise=0.0, verbose=False)

        x = torch.randint(0, VOCAB, (5, SEQ), device=dev)
        sigma = torch.zeros(5, device=dev)
        with torch.no_grad():
            ref = net(x, sigma).float()               # [B, L, V]
            t = torch.rand(5, device=dev)
            logits, _ = student.trunk(x, t, torch.ones_like(t))   # [B, M, L, V]

        for k in range(M):
            d = (logits[:, k] - ref).abs().max().item()
            status = "OK " if d < tol else "FAIL"
            if d >= tol:
                ok = False
            print(f"[warm-start-test] {source} M={M} branch {k}: max|student - teacher| = {d:.3e} {status}")

        # Every branch selected through k_select must give the same answer as the full sweep,
        # because the routed path is the one a committed rollout actually pays for. We check
        # all M branches and the router logits, since a committed rollout reads both.
        with torch.no_grad():
            _, router_swept = student.trunk(x, t, torch.ones_like(t))
            for k in range(M):
                sel = torch.full((5,), k, dtype=torch.long, device=dev)
                routed, router_sel = student.trunk(x, t, torch.ones_like(t), k_select=sel)
                d = (routed[:, 0] - logits[:, k]).abs().max().item()
                print(f"[warm-start-test] {source} M={M} routed-vs-swept branch {k}: {d:.3e} "
                      f"{'OK ' if d < 1e-5 else 'FAIL'}")
                if d >= 1e-5:
                    ok = False
                if router_sel is not None and router_swept is not None:
                    dr = (router_sel - router_swept).abs().max().item()
                    print(f"[warm-start-test] {source} M={M} router logits branch {k}: {dr:.3e} "
                          f"{'OK ' if dr < 1e-5 else 'FAIL'}")
                    if dr >= 1e-5:
                        ok = False

    if ran == 0:
        ok = False
        print("[warm-start-test] no teacher tree could be built")
    print("[warm-start-test] PASS" if ok else "[warm-start-test] FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

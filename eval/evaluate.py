"""Sample a trained student across a budget grid and score it.

    python -m eval.evaluate --ckpt run/ckpt_0050000.pt --out run/eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data
from models.mld3 import MLD3, MLD3Config, mask_forward_noise, mask_two_time_transition
from sampling import sampler as S

SAMPLERS = {"spread": S.spread_consensus, "commit": S.commit_k, "bestofm": S.best_of_m}


def load_model(path: str, device, use_ema: bool = True):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MLD3(MLD3Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    if use_ema and ck.get("ema"):
        model.load_state_dict({**model.state_dict(), **ck["ema"]["shadow"]})
    return model.eval(), ck["dataset"], ck["meta"]


@torch.no_grad()
def transition_information(model, loader, device, K: int, n_batches: int = 8) -> float:
    """Eq. 10: mean H(w) - H(r) over held-out steps, with r the posterior over the component.

    The pair (x_t, x_s) is drawn from the data rather than from the student's own kernel, so
    this estimates the transition information under the data distribution.
    """
    cfg = model.cfg
    total, n = 0.0, 0
    for i, x_1 in enumerate(loader):
        if i >= n_batches:
            break
        x_1 = x_1.to(device)
        B = x_1.shape[0]
        j = torch.randint(0, K, (B,), device=device)
        t, s = j / K, (j + 1) / K
        x_t = mask_forward_noise(x_1, t, cfg.mask_id)
        x_s = mask_two_time_transition(x_t, x_1, t, s, cfg.mask_id)

        logits, router = model.trunk(x_t, t, s)
        log_w = F.log_softmax(router, dim=-1)
        # Score the revealed tokens under components renormalised over the clean vocabulary.
        log_p = F.log_softmax(logits.index_fill(
            -1, torch.tensor([cfg.mask_id], device=device), -1e4), dim=-1)
        idx = x_s[:, None, :, None].expand(-1, cfg.latent_M, -1, 1)
        gathered = log_p.gather(-1, idx).squeeze(-1)
        revealed = ((x_t == cfg.mask_id) & (x_s != cfg.mask_id)).float()[:, None]
        ll = (gathered * revealed).sum(-1)

        posterior = F.softmax(log_w + ll, dim=-1)
        h_prior = -(log_w.exp() * log_w).sum(-1)
        h_posterior = -(posterior * posterior.clamp_min(1e-30).log()).sum(-1)
        total += float((h_prior - h_posterior).sum())
        n += B
    return total / max(n, 1)


def score(domain: str, dataset: str, samples: np.ndarray, reference: np.ndarray,
          meta: dict, device) -> dict:
    if domain == "text":
        from metrics import text_metrics as T
        texts = data.decode(dataset, samples, meta)
        return {"gen_ppl": T.generative_ppl(texts, device),
                "unigram_entropy": T.unigram_entropy(samples, meta["model_vocab"]),
                "unigram_kl": T.unigram_kl(samples, reference, meta["model_vocab"])}
    if domain == "mol":
        from metrics import mol_metrics
        return mol_metrics.evaluate([data.decode(dataset, samples, meta)], dataset)
    if domain == "dna":
        from metrics import dna_metrics
        return dna_metrics.evaluate(data.decode(dataset, samples, meta),
                                    data.decode(dataset, reference, meta), device=str(device))
    from metrics import image_metrics
    return image_metrics.evaluate(samples, meta, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampler", default="spread", choices=list(SAMPLERS))
    ap.add_argument("--components", type=int, default=2, help="components evaluated per step")
    ap.add_argument("--nfe", default="1,2,4,8,16,32")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset, meta = load_model(args.ckpt, device, not args.no_ema)
    domain, seq_len = meta["domain"], meta["seq_len"]
    loader, _ = data.make_loader(dataset, "val", args.batch, shuffle=False, num_workers=2)

    reference = np.concatenate([b.numpy() for b, _ in zip(loader, range(64))])[:args.samples]
    sample = SAMPLERS[args.sampler]
    kwargs = {"n_components": args.components} if args.sampler == "spread" else {}

    results = {"checkpoint": args.ckpt, "dataset": dataset, "sampler": args.sampler,
               "transition_information": transition_information(model, loader, device, K=4),
               "by_nfe": []}
    for nfe in (int(k) for k in args.nfe.split(",")):
        chunks, blocks = [], 0
        while sum(len(c) for c in chunks) < args.samples:
            n = min(args.batch, args.samples - sum(len(c) for c in chunks))
            x, blocks = sample(model, K=nfe, batch_size=n, seq_len=seq_len,
                               device=device, **kwargs)
            chunks.append(x.cpu().numpy())
        samples = np.concatenate(chunks)[:args.samples]
        row = {"nfe": nfe, "blocks_per_sample": blocks,
               **score(domain, dataset, samples, reference, meta, device)}
        print(row)
        results["by_nfe"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

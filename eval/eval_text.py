"""Evaluation harness for a trained checkpoint, across every corpus family.

Given an LKF text checkpoint it computes the two metric families the language papers report:

  (A) Likelihood cross-check (Table T3): the standard absorbing-diffusion NELBO of the held-out
      val set, using LKF's k-marginalized one-step denoiser p_θ(x_1^i | x_t) at s=1. For a linear
      keep-schedule κ_t=t the per-token bound is
          NELBO/tok = E_{t∼U(0,1)} E_{x_t} [ (1/(1-t)) Σ_{i: x_t^i=MASK} −log p_θ(x_1^i|x_t) ] / L,
      the masked-indicator (prob 1−t) cancels the 1/(1−t) weight so the estimator is finite.
      Reported as ELBO-PPL = exp(NELBO/tok) and, for char-level text8, BPC = NELBO/tok / ln2.

  (B) Generative-PPL vs NFE (Tables T1/T2): unconditional K-step samples decoded back to text and
      scored by gpt2-large, swept over NFE ∈ {1,2,4,8,16,32,...}, plus unigram/bigram entropy and
      unigram-KL to the val marginals, and the LKF-specific I(k) diagnostic. Extended/systems
      metrics (wall-clock/sample, throughput tok/s, NFE, params) are recorded per NFE.

Writes a normalized eval.json: {"metrics":[{"metric":..,"nfe":..,"value":..}], "meta":{...}}.

Usage (from the repository root):
    python -m eval.eval_text --ckpt <ckpt.pt> --out <dir>/eval.json \
        [--nfe 1,2,4,8,16,32] [--gen_samples 512] [--elbo_batches 40] [--gen_ppl_model gpt2-large]
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import text as textdata
from data import mols as moldata
from data import dna as dnadata
from data import images as imgdata
from models.latent_kernel import (
    LatentKernelFlow, LKFConfig, mask_forward_noise, mask_two_time_transition,
)
from models.teacher_adapters import build_mdlm_teacher
from sampling.sampler import sample_lkf, sample_lkf_analytic, best_of_m_sample
from metrics import metrics as M
from metrics import dna_metrics as DM
from metrics import image_metrics as IMG
from metrics import mol_metrics as MM
from sampling import policies as P


def domain_of(ds_name: str) -> str:
    """Which of the four corpora families a dataset name belongs to."""
    if moldata.is_mol_dataset(ds_name):
        return "mol"
    if dnadata.is_dna_dataset(ds_name):
        return "dna"
    if imgdata.is_image_dataset(ds_name):
        return "image"
    return "text"


def loader_module(domain: str):
    return {"mol": moldata, "dna": dnadata, "image": imgdata}.get(domain, textdata)


def load_model(ckpt_path: str, device, use_ema: bool = True):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = LKFConfig(**ck["cfg"])
    model = LatentKernelFlow(cfg).to(device)
    model.load_state_dict(ck["model"])
    # Prefer EMA weights when the checkpoint carries them (MDLM evaluates its EMA copy). Scoring the
    # raw weights instead is the control that tells us how much of a cell's quality is averaging.
    if not use_ema:
        print("[eval] using RAW weights (EMA shadow ignored)")
    elif ck.get("ema") is not None:
        shadow = ck["ema"]["shadow"]
        sd = model.state_dict()
        n_over = 0
        for n, v in shadow.items():
            if n in sd:
                sd[n] = v.to(device)
                n_over += 1
        model.load_state_dict(sd)
        print(f"[eval] using EMA weights (decay={ck['ema']['decay']}, "
              f"num_updates={ck['ema']['num_updates']}, {n_over} tensors)")
    model.eval()
    ds = ck.get("dataset")
    meta = ck.get("meta", {})
    return model, cfg, ds, meta, int(ck.get("step", 0))


@torch.no_grad()
def elbo_ppl(model, cfg, val_loader, device, n_batches: int, t_draws: int = 4,
             ongrid_K: int | None = None):
    """Absorbing-diffusion NELBO per token via the k-marginalized one-step denoiser.

    Passing ongrid_K restricts the noise level to the K times a distillation student was
    actually trained on, namely t in {0, 1/K, ..., (K-1)/K}. Reading the two numbers side by
    side separates a student that models the data badly from one that models it well only
    where it was supervised, and the second failure is the one a grid-only objective causes.
    """
    mask_id = cfg.mask_id
    tot_nll, tot_tok = 0.0, 0
    for bi, xb in enumerate(val_loader):
        if bi >= n_batches:
            break
        x1 = xb.to(device)
        B, L = x1.shape
        for _ in range(t_draws):
            if ongrid_K:
                j = torch.randint(0, ongrid_K, (B,), device=device)
                t = (j.float() / ongrid_K).clamp(1e-3, 1 - 1e-3)
            else:
                t = torch.rand(B, device=device).clamp(1e-3, 1 - 1e-3)
            keep = torch.rand(B, L, device=device) < t[:, None]
            x_t = torch.where(keep, x1, torch.full_like(x1, mask_id))
            s = torch.ones(B, device=device)
            logits, router_logits = model.trunk(x_t, t, s)          # [B,M,L,V], [B,M]
            w = F.softmax(router_logits, dim=-1)                    # [B,M]
            p = F.softmax(logits, dim=-1)                           # [B,M,L,V]
            p_marg = (w[:, :, None, None] * p).sum(1)               # [B,L,V] marginal denoiser
            lp = torch.log(p_marg.clamp_min(1e-30))
            true_lp = lp.gather(-1, x1[:, :, None]).squeeze(-1)     # [B,L]
            masked = (x_t == mask_id).float()                       # [B,L]
            # PAD-aware ckpts (cfg.pad_id >= 0): exclude PAD positions from the NELBO and the
            # token count, matching the training normalization (MDLM attention_mask semantics).
            pad_id = getattr(cfg, "pad_id", -1)
            if pad_id >= 0:
                valid = (x1 != pad_id).float()
                masked = masked * valid
                n_tok = int(valid.sum())
            else:
                n_tok = B * L
            weight = (1.0 / (1.0 - t)).clamp_max(1e4)               # [B]
            # per-sequence weighted CE over masked positions
            seq_nll = (weight[:, None] * (-true_lp) * masked).sum(1)  # [B]
            tot_nll += float(seq_nll.sum())
            tot_tok += n_tok
    nelbo_tok = tot_nll / max(tot_tok, 1)
    return nelbo_tok


@torch.no_grad()
def transition_mi(model, cfg, val_loader, device, n_batches: int = 4):
    """Mean I(k; X_s | x_t) diagnostic across a few (t,s) draws on real x_t."""
    mi_tot, n = 0.0, 0
    for bi, xb in enumerate(val_loader):
        if bi >= n_batches:
            break
        x1 = xb.to(device)
        B = x1.shape[0]
        t = torch.rand(B, device=device) * (1 - cfg.dt_min)
        s = (t + cfg.dt_min + torch.rand(B, device=device) * (1 - t - cfg.dt_min).clamp_min(0)).clamp(max=1.0)
        x_t = mask_forward_noise(x1, t, cfg.mask_id)
        x_s = mask_two_time_transition(x_t, x1, t, s, cfg.mask_id)
        _, log_w, per = model.mixture_logprob(x_t, x_s, t, s)
        w = log_w.exp()
        H_prior = -(w * log_w).sum(-1)
        post = F.softmax(log_w + per, dim=-1)
        H_post = -(post * post.clamp_min(1e-30).log()).sum(-1)
        mi_tot += float((H_prior - H_post).sum()); n += B
    return mi_tot / max(n, 1)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_kind = getattr(args, "teacher_kind", None)
    if teacher_kind:
        # Cross-family teacher reference: build the native adapter and score it through the
        # same absorbing analytic sampler the students use. The adapter has no two-time
        # mixture head, so we skip the transition-information diagnostic below.
        sl = args.teacher_seq_len
        if teacher_kind == "mdlm":
            model, tstate = build_mdlm_teacher(
                args.teacher_path, device, seq_len=sl,
                vocab_size=args.teacher_vocab_size, mask_id=args.teacher_mask_id,
                dit_source=args.teacher_dit_source,
                time_conditioning=not args.teacher_no_time_conditioning)
        else:
            raise SystemExit(f"unknown --teacher_kind {teacher_kind}")
        cfg = model.cfg
        ds_name = args.dataset or "wikitext103"
        # The molecule and DNA teachers run on their own small vocabularies, and describing
        # them as GPT-2 would put a wrong tokenizer name in the metadata of the eval file.
        if args.teacher_vocab_size is None:
            meta = {"tokenizer": "gpt2", "vocab_size": 50257}
        else:
            meta = {"tokenizer": domain_of(ds_name), "vocab_size": args.teacher_vocab_size}
        step = tstate.get("step", -1)
    else:
        model, cfg, ds_name, meta, step = load_model(args.ckpt, device, use_ema=not args.no_ema)
    ds_name = args.dataset or ds_name
    tokenizer = meta.get("tokenizer")
    vocab_data = meta.get("vocab_size", cfg.vocab_size - 1)
    seq_len = cfg.seq_len
    n_params = sum(p.numel() for p in model.parameters())
    is_char = (tokenizer == "char")
    print(f"[eval_text] ds={ds_name} tok={tokenizer} step={step} params={n_params/1e6:.1f}M L={seq_len}")

    domain = domain_of(ds_name)
    val_loader, _ = loader_module(domain).make_loader(
        ds_name, "val", args.gen_ppl_batch, shuffle=False, num_workers=2)

    metrics = []
    def add(metric, value, nfe=None):
        metrics.append({"metric": metric, "nfe": nfe, "value": float(value)})

    # ---- (A) likelihood ----
    nelbo = elbo_ppl(model, cfg, val_loader, device, args.elbo_batches)
    add("nll_per_tok", nelbo)
    add("elbo_ppl", math.exp(min(nelbo, 20)))
    if is_char:
        add("bpc", nelbo / math.log(2))
    if not teacher_kind:
        add("mi_k_transition", transition_mi(model, cfg, val_loader, device))
    print(f"[eval_text] NELBO/tok={nelbo:.4f}  elbo_ppl={math.exp(min(nelbo,20)):.2f}"
          + (f"  bpc={nelbo/math.log(2):.4f}" if is_char else ""))
    # The on-grid NELBO, reported next to the dense one, is how we tell a student that models
    # the data badly everywhere from a student that models it well only at the K noise levels
    # its objective supervised. The gap between the two is the quantity the dense time sampler is
    # meant to close, and we want it on every run rather than only on the ones we suspect.
    if args.ongrid_K > 0:
        nelbo_on = elbo_ppl(model, cfg, val_loader, device, args.elbo_batches,
                            ongrid_K=args.ongrid_K)
        add("nll_per_tok_ongrid", nelbo_on)
        add("nll_per_tok_offgrid_gap", nelbo - nelbo_on)
        print(f"[eval_text] on-grid(K={args.ongrid_K}) NELBO/tok={nelbo_on:.4f}  "
              f"gap={nelbo - nelbo_on:+.4f}")

    # reference val samples (token ids) for entropy / unigram-KL
    ref = []
    for xb in val_loader:
        ref.append(xb.numpy())
        if sum(a.shape[0] for a in ref) >= max(args.gen_samples, 1024):
            break
    ref = np.concatenate(ref, 0)[: max(args.gen_samples, 1024), :seq_len]

    # ---- (B) gen-PPL vs NFE ----
    sampler_fn = sample_lkf_analytic if args.sampler == "analytic" else sample_lkf
    # commit_k: draw k once from the router at step 0 and hold it for the whole trajectory.
    # Matches the sequence-level marginalization used at training time; recommended M>1
    # default. Supported by both the ancestral and analytic samplers.
    use_commit_k = bool(args.commit_k) and cfg.latent_M > 1
    # best_of_m: run M parallel freeze_k=m rollouts, score with the model's own sequence-
    # level mixture NELBO (T=score_draws shared MC draws), keep argmin. Headline M>1
    # policy; only meaningful for M > 1.
    # The pool size defaults to the component count. Overriding it is what the M=1 paired control
    # of the decoding panel needs, since that control gives a factorized student the same number
    # of candidates a mixture gets and lets the reranker choose among them.
    bom_pool = args.best_of_m_pool or cfg.latent_M
    use_bom = bool(args.best_of_m) and bom_pool > 1
    # consensus: search over the component sequence inside a single chain. At every step the one
    # full call already enumerates all M continuations, so we draw one proposal per component and
    # keep the one the router-weighted marginal scores highest. Costs K*C(M) and returns a single
    # sample per slot, so unlike best_of_m it never reranks finished sequences.
    use_consensus = bool(args.consensus) and cfg.latent_M > 1
    if use_bom or use_consensus:
        sampler_kw = {}
    else:
        sampler_kw = {"commit_k": True} if use_commit_k else {}
    print(f"[eval_text] sampler={args.sampler} commit_k={use_commit_k} "
          f"best_of_m={use_bom} consensus={use_consensus} (M={cfg.latent_M})")
    # The VQGAN and the Inception metric are built once for the whole sweep. Reloading a 958 MB
    # decoder at every point of the NFE grid would cost a quarter hour per cell for nothing.
    img_vq = img_metric = img_ref = None
    if domain == "image" and not args.no_gen_ppl:
        img_vq = IMG.build_vqgan(device)
        img_metric = IMG.build_metric(device)
        img_ref = IMG.load_reference(args.image_ref, device)
        print(f"[eval_text] image reference {args.image_ref}: {tuple(img_ref.shape)}")
    nfes = [int(x) for x in args.nfe.split(",") if x]
    for nfe in nfes:
        t0 = time.time()
        outs = []
        got = 0
        bs = args.sample_batch
        while got < args.gen_samples:
            b = min(bs, args.gen_samples - got)
            if use_consensus:
                x, _ = P.consensus_chain(model, K=nfe, batch_size=b, seq_len=seq_len,
                                          device=device, soft=args.consensus_soft,
                                          analytic=(args.sampler == "analytic"))
            elif use_bom:
                x = best_of_m_sample(model, K=nfe, batch_size=b, seq_len=seq_len,
                                     device=device, M=bom_pool,
                                     score_draws=args.best_of_m_draws,
                                     sampler=args.sampler,
                                     selector=args.best_of_m_selector)
            else:
                x = sampler_fn(model, K=nfe, batch_size=b, seq_len=seq_len, device=device,
                               **sampler_kw)
            outs.append(x.cpu().numpy()); got += b
        samp = np.concatenate(outs, 0)
        dt = time.time() - t0
        sec_per_sample = dt / max(len(samp), 1)
        tok_per_s = len(samp) * seq_len / max(dt, 1e-6)

        add("unigram_entropy", M.unigram_entropy(samp, cfg.vocab_size), nfe)
        add("bigram_entropy", M.bigram_entropy(samp, cfg.vocab_size), nfe)
        add("unigram_kl", M.unigram_kl(samp, ref, cfg.vocab_size), nfe)
        add("unique_frac", M.unique_fraction(samp), nfe)
        add("sec_per_sample", sec_per_sample, nfe)
        add("throughput_tok_s", tok_per_s, nfe)

        if args.no_gen_ppl:
            print(f"[eval_text] nfe={nfe:>4}  (quality judge skipped)  "
                  f"{sec_per_sample*1e3:.1f} ms/samp")
        elif domain == "mol":
            # The molecule judge is the RDKit parser, and validity, uniqueness and novelty
            # follow PairFlow's own conventions so that the two tables line up.
            # PairFlow reports the mean over independent trials rather than one pooled draw,
            # because uniqueness and novelty both depend on the size of the set they are
            # measured on. We cut the generated pool into trials of the same size they use.
            seqs = moldata.decode(ds_name, samp)
            size = max(int(args.mol_trial_size), 1)
            trials = [seqs[i:i + size] for i in range(0, len(seqs), size)] or [seqs]
            res = MM.evaluate(trials, ds_name)
            valid_smiles = res.pop("valid_smiles", [])
            for key in ("valid_pct", "unique_pct", "novel_pct",
                        "unique_canon_pct", "novel_canon_pct"):
                add(key, res[key], nfe)
            print(f"[eval_text] nfe={nfe:>4}  valid={res['valid_pct']*100:.2f}%  "
                  f"unique={res['unique_pct']*100:.2f}%  novel={res['novel_pct']*100:.2f}%  "
                  f"{sec_per_sample*1e3:.1f} ms/samp")
            # The Frechet ChemNet distance and the MOSES panel cost a forward pass over a
            # chemistry network and a quadratic fingerprint comparison, and the appendix table
            # reports them at a single budget, so we compute them at that budget only.
            if nfe == args.mol_dist_nfe and valid_smiles:
                panel = MM.distributional_report(valid_smiles, ds_name, device=str(device))
                for key, val in panel.items():
                    if val is not None:
                        add(key, float(val), nfe)
                shown = {k: v for k, v in panel.items()
                         if k in ("fcd", "scaf", "snn", "intdiv1", "filters") and v is not None}
                print("[eval_text] distributional panel  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in shown.items()))
        elif domain == "dna":
            # The DNA judge is the DeepSTARR regressor, which scores function rather than
            # syntax, and the real reference set is the held-out split we already loaded.
            gen = dnadata.decode(ds_name, samp)
            real = dnadata.decode(ds_name, ref[:len(gen)])
            res = DM.evaluate(gen, real, device=str(device))
            res.update(DM.motif_counts(gen))
            for key, val in res.items():
                if key not in ("n_gen", "n_real"):
                    add(key, val, nfe)
            print(f"[eval_text] nfe={nfe:>4}  fbd={res['fbd']:.3f}  "
                  f"w1_dev={res['w1_dev']:.3f}  pi90={res['pi90_dev']:.3f}  "
                  f"{sec_per_sample*1e3:.1f} ms/samp")
        elif domain == "image":
            # The image judge decodes the sampled token grids back to pixels through the VQGAN
            # the corpus was tokenized with, and scores them against cached Inception features
            # of the real validation set. Decoding is streamed, so the ten thousand images at a
            # single NFE point never exist at once.
            res = IMG.score(img_metric,
                            IMG.tokens_to_uint8(samp, img_vq, device, chunk=args.image_chunk),
                            img_ref)
            for key, val in res.items():
                add(key, val, nfe)
            print(f"[eval_text] nfe={nfe:>4}  fid={res['fid']:.2f}  "
                  f"is={res['inception_score']:.2f}  prec={res['precision']:.3f}  "
                  f"rec={res['recall']:.3f}  {sec_per_sample*1e3:.1f} ms/samp")
        else:
            texts = textdata.decode(ds_name, samp)
            texts = [tt for tt in texts if len(tt.strip()) > 0]
            if texts:
                gp = M.gen_ppl_gpt2(texts, model_name=args.gen_ppl_model, device=device,
                                    batch_size=8)
                add("gen_ppl", gp["gen_ppl"], nfe)
                print(f"[eval_text] nfe={nfe:>4}  gen_ppl={gp['gen_ppl']:.2f}  "
                      f"uni_kl={metrics[-6]['value']:.4f}  {sec_per_sample*1e3:.1f} ms/samp")

    add("n_params_M", n_params / 1e6)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics,
               "meta": {"dataset": ds_name, "domain": domain,
                        "tokenizer": tokenizer, "step": step,
                        "seq_len": seq_len, "latent_M": cfg.latent_M, "sampler": args.sampler,
                        "commit_k": use_commit_k, "best_of_m": use_bom,
                        "consensus": use_consensus,
                        "consensus_soft": bool(args.consensus_soft) if use_consensus else None,
                        "best_of_m_draws": args.best_of_m_draws if use_bom else None,
                        "best_of_m_pool": bom_pool if use_bom else None,
                        "best_of_m_selector": args.best_of_m_selector if use_bom else None,
                        "weights": "raw" if args.no_ema else "ema",
                        "n_params": n_params, "ckpt": str(args.ckpt)}}
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[eval_text] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no_ema", action="store_true",
                    help="score the raw training weights instead of the EMA shadow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher_kind", default=None, choices=["mdlm"],
                    help="evaluate a cross-family teacher through its native adapter instead "
                         "of an LKF student ckpt")
    ap.add_argument("--teacher_path", default=None,
                    help="teacher checkpoint")
    ap.add_argument("--teacher_seq_len", type=int, default=128)
    ap.add_argument("--teacher_vocab_size", type=int, default=None,
                    help="model vocabulary of the teacher, left unset for the GPT-2 text "
                         "teachers and given explicitly for the molecule and DNA ones")
    ap.add_argument("--teacher_mask_id", type=int, default=None)
    ap.add_argument("--teacher_dit_source", default="mdlm", choices=["mdlm", "pairflow"],
                    help="which fork's DiT definition the teacher weights were written by")
    ap.add_argument("--teacher_no_time_conditioning", action="store_true",
                    help="set for the PairFlow molecule and DNA teachers, which are trained "
                         "without time conditioning")
    ap.add_argument("--dataset", default=None, help="override dataset name (else from ckpt)")
    ap.add_argument("--nfe", default="1,2,4,8,16,32")
    ap.add_argument("--gen_samples", type=int, default=512)
    ap.add_argument("--sample_batch", type=int, default=128)
    ap.add_argument("--gen_ppl_batch", type=int, default=64)
    ap.add_argument("--elbo_batches", type=int, default=40)
    ap.add_argument("--ongrid_K", type=int, default=0,
                    help="also report the NELBO restricted to t in {0,1/K,...,(K-1)/K}, "
                         "the noise levels a K-step distillation student was trained on")
    ap.add_argument("--gen_ppl_model", default="gpt2-large")
    ap.add_argument("--sampler", default="ancestral", choices=["ancestral", "analytic"],
                    help="ancestral = sample x_s from Ψ_{t→s} (learned reveal rate); "
                         "analytic = MDLM-equivalent absorbing reverse (schedule-enforced reveal)")
    ap.add_argument("--commit_k", action=argparse.BooleanOptionalAction, default=True,
                    help="draw k once at step 0 and hold for the trajectory (default ON; "
                         "--no-commit_k for per-step re-routing). No-op at M=1 and when "
                         "--best_of_m is also set.")
    ap.add_argument("--best_of_m", action=argparse.BooleanOptionalAction, default=False,
                    help="Best-of-M rollout selector: run M parallel freeze_k=m rollouts "
                         "and keep argmin sequence-level NELBO. No-op at M=1. Overrides "
                         "--commit_k when both are set.")
    ap.add_argument("--consensus", action=argparse.BooleanOptionalAction, default=False,
                    help="Consensus commit: at every step draw one proposal per component and "
                         "keep the one the router-weighted marginal scores highest at the masked "
                         "positions. Single chain, costs K*C(M). No-op at M=1. Overrides "
                         "--commit_k and --best_of_m when set.")
    ap.add_argument("--consensus_soft", action="store_true",
                    help="draw the winning component from a softmax of the per-masked-token "
                         "score instead of taking the argmax.")
    ap.add_argument("--best_of_m_selector", default="nelbo", choices=["nelbo", "evidence"],
                    help="how best-of-M picks the winner among the M committed rollouts. "
                         "'nelbo' scores each candidate with the mixture NELBO and costs "
                         "M*draws extra full calls; 'evidence' keeps the candidate whose "
                         "component gave the highest log-probability to the tokens it revealed "
                         "and is free.")
    ap.add_argument("--best_of_m_draws", type=int, default=4,
                    help="shared MC (t, mask) draws used to score best-of-M candidates")
    ap.add_argument("--best_of_m_pool", type=int, default=None,
                    help="candidates per slot for best-of-M (defaults to the component count; "
                         "set it above 1 on an M=1 student to run the paired reranking control)")
    ap.add_argument("--no_gen_ppl", action="store_true")
    ap.add_argument("--mol_trial_size", type=int, default=1024,
                    help="molecule metrics are averaged over independent trials of this size, "
                         "which is what PairFlow reports")
    ap.add_argument("--mol_dist_nfe", type=int, default=8,
                    help="the single budget at which the Frechet ChemNet distance and the "
                         "MOSES panel are computed")
    ap.add_argument("--image_ref", default="val_real",
                    help="which cached Inception reference the image arm scores against. "
                         "val_real is the standard FID reference and val_recon is the "
                         "tokenizer floor")
    ap.add_argument("--image_chunk", type=int, default=64,
                    help="how many token grids the VQGAN decodes at once")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

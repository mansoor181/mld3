"""Decoding-policy ablation for a trained mixture-flow (MLDF) checkpoint.

For every sampling budget K on the grid we draw one candidate pool per slot and score that
same pool with every pool-based selector, so the candidates are held fixed and the only thing
that varies across a column is how the winner is chosen. The single-chain policies need their
own rollouts and are run alongside at the same budget. Every policy also reports what it
actually costs in transformer-block passes, the unit the cost model in the paper is written in,
because a call that asks for all M components costs C(M) = (D - L_lat) + M L_lat while a call
that names one component costs only D, and a fair comparison of decoding policies has to price
that difference.

Writes {"policies": [{policy, nfe, gen_ppl, unigram_entropy, ..., blocks_per_sample}], "meta":{}}.

Usage (from the repository root):
    python -m eval.eval_policies --ckpt <ckpt.pt> --out <dir>/eval_policies.json \
        [--nfe 1,2,4,8,16,32] [--gen_samples 512] [--sample_batch 32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data import text as textdata
from eval.eval_text import domain_of, load_model, loader_module
from metrics import metrics as M
from sampling import policies as P
from metrics import mol_metrics as MM


def run(args):
    # The sweep runs about twenty policies at every budget, and the candidate pool alone spends M
    # complete rollouts per slot before the NELBO selector adds its own full calls on top of them.
    # A run that only needs three columns must therefore consult this filter before the generation
    # call rather than at reporting time, because the generation is where the saving is. Names are
    # matched in full, since a substring test on "commit-k" would also select every gated variant.
    want_all = "all" in args.policies
    wanted = set(args.policies)

    def want(*names) -> bool:
        return want_all or any(n in wanted for n in names)

    # Each of the two pooled blocks below produces several policies from one shared set of
    # candidates, so a block is gated on whether any of the names it can emit was asked for.
    pool_names = ["best-of-M, running evidence", "random component (no selection)",
                  "best-of-M, mixture NELBO", "successive halving, running evidence"] \
        + [f"best-of-{r}, running evidence" for r in args.subset]
    gated_pool_names = [f"best-of-M over gated chains (rho={args.pool_rho})",
                        f"gated chain, no selection (rho={args.pool_rho})",
                        f"successive halving over gated chains (rho={args.pool_rho})"]
    # A misspelled name would otherwise cost a complete run and leave an empty file behind. The
    # check depends only on the arguments, so it happens before the checkpoint is even loaded and
    # a typo costs a second rather than an hour.
    known = set(pool_names) | set(gated_pool_names) \
        | {"commit-k", "router-marginal chain", "consensus commit", "consensus commit (soft)"} \
        | {f"gated commit-k, self arbiter (rho={r})" for r in args.rho} \
        | {f"gated commit-k, mixture arbiter (rho={r})" for r in args.rho}
    unknown = wanted - known - {"all"}
    if unknown:
        raise SystemExit(f"[policies] unknown policy name(s) {sorted(unknown)}\n"
                         f"[policies] known names are {sorted(known)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ds_name, meta, step = load_model(args.ckpt, device)
    ds_name = args.dataset or ds_name
    seq_len = cfg.seq_len
    Mm = cfg.latent_M
    S, Lk, D = P.block_costs(cfg)
    full = P.full_call_cost(cfg)
    # The confidence gate imposes a reveal schedule of its own, and stacking the analytic
    # schedule underneath it would decide the same question twice, so the gated policies keep
    # the ancestral kernel at either setting and only the free policies follow --sampler.
    AN = (args.sampler == "analytic")
    print(f"[policies] ds={ds_name} step={step} M={Mm} depth={cfg.depth} "
          f"latent_last_L={cfg.latent_last_L}  full call={full} blocks, "
          f"single-branch call={D} blocks")

    # The molecule corpora reach this harness through the same loader interface the text corpora
    # use, so the reference pass below is unchanged and only the judge at the bottom differs. The
    # DNA and image domains have no judge here at all, and scoring a nucleotide sequence through
    # the GPT-2 perplexity of the text branch would fill a file with numbers that mean nothing,
    # so we stop instead of writing it.
    domain = domain_of(ds_name)
    if domain in ("dna", "image"):
        raise SystemExit(f"[policies] no judge is implemented for the {domain} domain")
    data_mod = loader_module(domain)
    val_loader, _ = data_mod.make_loader(ds_name, "val", args.gen_ppl_batch, shuffle=False,
                                         num_workers=2)
    ref = []
    for xb in val_loader:
        ref.append(xb.numpy())
        if sum(a.shape[0] for a in ref) >= max(args.gen_samples, 1024):
            break
    ref = np.concatenate(ref, 0)[: max(args.gen_samples, 1024), :seq_len]

    rows = []
    nfes = [int(x) for x in args.nfe.split(",") if x]
    for K in nfes:
        bank: dict[str, list] = {}
        cost: dict[str, float] = {}
        wall: dict[str, float] = {}

        def stash(name, x, blocks, dt):
            # A policy the run never asked for does not reach the bank, so the reporting loop
            # needs no filter of its own and a gate missed above can only cost time rather than
            # produce a row nobody wanted.
            if not want(name):
                return
            bank.setdefault(name, []).append(x.cpu().numpy())
            cost[name] = blocks
            wall[name] = wall.get(name, 0.0) + dt

        got = 0
        while got < args.gen_samples:
            b = min(args.sample_batch, args.gen_samples - got)
            ar = torch.arange(b, device=device)

            # ---- single-chain policies ----
            if want("commit-k"):
                t0 = time.time()
                x, c = P.commit_k_chain(model, K, b, seq_len, device, analytic=AN)
                stash("commit-k", x, c, time.time() - t0)

            # The self-arbiter gate never leaves the committed branch, so it runs unchanged at
            # M=1 and is the control that isolates confidence-ordered unmasking on its own.
            for rho in args.rho:
                name = f"gated commit-k, self arbiter (rho={rho})"
                if not want(name):
                    continue
                t0 = time.time()
                x, c = P.gated_commit_chain(model, K, b, seq_len, device, rho=rho,
                                            arbiter="self")
                stash(name, x, c, time.time() - t0)

            if Mm > 1:
                if want("router-marginal chain"):
                    t0 = time.time()
                    x, c = P.marginal_chain(model, K, b, seq_len, device, analytic=AN)
                    stash("router-marginal chain", x, c, time.time() - t0)

                if want("consensus commit"):
                    t0 = time.time()
                    x, c = P.consensus_chain(model, K, b, seq_len, device, soft=False,
                                             analytic=AN)
                    stash("consensus commit", x, c, time.time() - t0)

                if want("consensus commit (soft)"):
                    t0 = time.time()
                    x, c = P.consensus_chain(model, K, b, seq_len, device, soft=True,
                                             analytic=AN)
                    stash("consensus commit (soft)", x, c, time.time() - t0)

                for rho in args.rho:
                    name = f"gated commit-k, mixture arbiter (rho={rho})"
                    if not want(name):
                        continue
                    t0 = time.time()
                    x, c = P.gated_commit_chain(model, K, b, seq_len, device, rho=rho,
                                                arbiter="mixture")
                    stash(name, x, c, time.time() - t0)

            if Mm > 1 and want(*pool_names):
                # ---- one shared candidate pool, many selectors ----
                t0 = time.time()
                cands, ev, ev_hist, w0, pool_cost = P.rollout_pool(model, K, b, seq_len, device,
                                                                   analytic=AN)
                dt_pool = time.time() - t0

                stash("best-of-M, running evidence", P.pick(cands, ev.argmax(0)),
                      pool_cost, dt_pool)

                # A random component needs a single chain, so it is priced as one chain even
                # though we read it off the pool to keep the candidates identical.
                one_chain = S + Lk + (K - 1) * D
                rnd = torch.randint(0, Mm, (b,), device=device)
                stash("random component (no selection)", P.pick(cands, rnd),
                      one_chain, dt_pool / Mm)

                # This selector is the one part of the block that costs full calls beyond the
                # pool itself, so it carries its own gate rather than riding on the pool's.
                if want("best-of-M, mixture NELBO"):
                    t0 = time.time()
                    sc, sel_cost = P.score_mixture_nelbo(model, cands, draws=args.nelbo_draws)
                    stash("best-of-M, mixture NELBO", P.pick(cands, sc.argmin(0)),
                          pool_cost + sel_cost, dt_pool + (time.time() - t0))

                for r in [rr for rr in args.subset if 1 < rr < Mm]:
                    sub = torch.stack([torch.randperm(Mm, device=device)[:r] for _ in range(b)], 1)
                    evs = ev[sub, ar[None, :]]                          # [r, b]
                    win = sub[evs.argmax(0), ar]
                    stash(f"best-of-{r}, running evidence", P.pick(cands, win),
                          S + r * Lk + (K - 1) * r * D, dt_pool * r / Mm)

                win, spend = P.successive_halving(cfg, ev_hist, K, Mm)
                stash("successive halving, running evidence", P.pick(cands, win),
                      spend, dt_pool * spend / max(pool_cost, 1e-9))
                del cands, ev, ev_hist

            if Mm > 1 and want(*gated_pool_names):
                # Selection and confidence gating are orthogonal, so we also run the pool with
                # every chain gated to ask whether picking a winner still helps once each
                # candidate is already decoded in confidence order.
                t0 = time.time()
                cands, ev, ev_hist, _, pool_cost = P.rollout_pool(model, K, b, seq_len, device,
                                                                  rho=args.pool_rho)
                dt_pool = time.time() - t0
                stash(f"best-of-M over gated chains (rho={args.pool_rho})",
                      P.pick(cands, ev.argmax(0)), pool_cost, dt_pool)
                rnd = torch.randint(0, Mm, (b,), device=device)
                stash(f"gated chain, no selection (rho={args.pool_rho})", P.pick(cands, rnd),
                      S + Lk + (K - 1) * D, dt_pool / Mm)
                win, spend = P.successive_halving(cfg, ev_hist, K, Mm)
                stash(f"successive halving over gated chains (rho={args.pool_rho})",
                      P.pick(cands, win), spend, dt_pool * spend / max(pool_cost, 1e-9))
                del cands, ev, ev_hist
            got += b

        for name, chunks in bank.items():
            samp = np.concatenate(chunks, 0)
            row = {"policy": name, "nfe": K}
            if domain == "mol":
                # The molecule judge is the RDKit parser, and validity, uniqueness and novelty
                # follow the same conventions eval_text.py uses, so that a policy row here and a
                # commit-k row of the main table can be read against each other. We cut the pool
                # into independent trials for the reason that motivates it there as well, since
                # uniqueness and novelty both depend on the size of the set they are measured on.
                seqs = data_mod.decode(ds_name, samp)
                size = max(int(args.mol_trial_size), 1)
                trials = [seqs[i:i + size] for i in range(0, len(seqs), size)] or [seqs]
                res = MM.evaluate(trials, ds_name)
                valid_smiles = res.pop("valid_smiles", [])
                for key in ("valid_pct", "unique_pct", "novel_pct",
                            "unique_canon_pct", "novel_canon_pct"):
                    row[key] = float(res[key])
                # The Frechet ChemNet distance and the MOSES panel cost a forward pass over a
                # chemistry network and a quadratic fingerprint comparison, and here that price
                # is paid once per policy rather than once per cell, so we compute them at the
                # single budget the appendix reports and nowhere else.
                if K == args.mol_dist_nfe and valid_smiles:
                    panel = MM.distributional_report(valid_smiles, ds_name, device=str(device))
                    for key, val in panel.items():
                        if val is not None:
                            row[key] = float(val)
            else:
                texts = [tt for tt in textdata.decode(ds_name, samp) if len(tt.strip()) > 0]
                gp = M.gen_ppl_gpt2(texts, model_name=args.gen_ppl_model, device=device,
                                    batch_size=8)["gen_ppl"] if texts else float("nan")
                row["gen_ppl"] = float(gp)
            row.update({"unigram_entropy": float(M.unigram_entropy(samp, cfg.vocab_size)),
                        "bigram_entropy": float(M.bigram_entropy(samp, cfg.vocab_size)),
                        "unigram_kl": float(M.unigram_kl(samp, ref, cfg.vocab_size)),
                        "unique_frac": float(M.unique_fraction(samp)),
                        "blocks_per_sample": float(cost[name]),
                        "sec_per_sample": float(wall[name] / max(len(samp), 1))})
            rows.append(row)
            if domain == "mol":
                print(f"[policies] K={K:>3} {name:<38} valid={row['valid_pct']*100:6.2f}%  "
                      f"uniq={row['unique_canon_pct']*100:6.2f}%  "
                      f"novel={row['novel_canon_pct']*100:6.2f}%  blocks={cost[name]:7.0f}")
            else:
                print(f"[policies] K={K:>3} {name:<38} gen_ppl={row['gen_ppl']:9.2f} "
                      f"H={row['unigram_entropy']:.2f} blocks={cost[name]:7.0f}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"policies": rows,
                       "meta": {"dataset": ds_name, "step": step, "latent_M": Mm,
                                "depth": cfg.depth, "latent_last_L": cfg.latent_last_L,
                                "full_call_blocks": full, "branch_call_blocks": D,
                                "shared_stack_blocks": S, "seq_len": seq_len,
                                "gen_samples": args.gen_samples,
                                "nelbo_draws": args.nelbo_draws,
                                "sampler": args.sampler,
                                "domain": domain,
                                "policies_requested": list(args.policies),
                                "ckpt": str(args.ckpt)}}, fh, indent=2)
        print(f"[policies] wrote {out} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--nfe", default="1,2,4,8,16,32")
    ap.add_argument("--gen_samples", type=int, default=512)
    ap.add_argument("--sample_batch", type=int, default=32)
    ap.add_argument("--gen_ppl_batch", type=int, default=16)
    ap.add_argument("--gen_ppl_model", default="gpt2-large")
    ap.add_argument("--nelbo_draws", type=int, default=4)
    ap.add_argument("--rho", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    ap.add_argument("--pool_rho", type=float, default=0.25)
    ap.add_argument("--sampler", choices=["ancestral", "analytic"], default="ancestral")
    ap.add_argument("--subset", type=int, nargs="*", default=[2, 4])
    # Policy names carry commas, so a comma-separated string is not usable here, and nargs="+"
    # rather than nargs="*" because a bare --policies under the latter yields an empty list,
    # which would run nothing at all and leave an empty file behind without complaining.
    ap.add_argument("--policies", nargs="+", default=["all"],
                    help="restrict the sweep to these policies, named exactly as they appear in "
                         "the output, for example --policies 'consensus commit' 'commit-k'. The "
                         "generation is skipped for everything else, which is where the saving "
                         "is. The default of 'all' runs the whole sweep.")
    ap.add_argument("--mol_trial_size", type=int, default=1024,
                    help="molecule metrics are averaged over independent trials of this size")
    ap.add_argument("--mol_dist_nfe", type=int, default=8,
                    help="the single budget at which the Frechet ChemNet distance and the MOSES "
                         "panel are computed. Set it off the grid to skip them entirely.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

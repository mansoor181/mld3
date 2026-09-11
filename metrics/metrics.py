"""Sample-level text metrics used by the evaluation sweeps.

Kept surface: unigram and bigram entropy, unigram KL against data samples, unique-sequence fraction, and generative perplexity under a frozen GPT-2 judge. eval/eval_text.py and eval/eval_policies.py import this module as `M`; the molecule, DNA, and image metrics live in their own modules.
"""
from __future__ import annotations
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def unigram_entropy(samples: np.ndarray, vocab_size: int) -> float:
    """Mean per-position unigram entropy (nats)."""
    L = samples.shape[1]
    counts = np.zeros((L, vocab_size), dtype=np.float64)
    for i in range(L):
        for v, c in zip(*np.unique(samples[:, i], return_counts=True)):
            counts[i, v] = c
    p = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1)
    return float(ent.mean())

def bigram_entropy(samples: np.ndarray, vocab_size: int) -> float:
    """Mean per-adjacent-pair bigram entropy (nats)."""
    N, L = samples.shape
    if L < 2:
        return 0.0
    pairs = samples[:, :-1].astype(np.int64) * vocab_size + samples[:, 1:].astype(np.int64)
    V2 = vocab_size * vocab_size
    ent_sum = 0.0
    for i in range(L - 1):
        u, c = np.unique(pairs[:, i], return_counts=True)
        p = c.astype(np.float64) / c.sum()
        ent_sum += float(-(p * np.log(np.clip(p, 1e-300, None))).sum())
    return ent_sum / (L - 1)

def unigram_kl(model_samples: np.ndarray, data_samples: np.ndarray,
               vocab_size: int, eps: float = 1e-6) -> float:
    """KL(model || data) between per-position marginal unigram distributions, averaged over L.

    Symmetric variant: consider using JSD if desired.
    """
    L = model_samples.shape[1]
    assert data_samples.shape[1] == L
    kls = []
    for i in range(L):
        pm = np.bincount(model_samples[:, i], minlength=vocab_size).astype(np.float64)
        pd = np.bincount(data_samples[:, i], minlength=vocab_size).astype(np.float64)
        pm = pm / pm.sum() + eps
        pd = pd / pd.sum() + eps
        pm /= pm.sum(); pd /= pd.sum()
        kls.append(float((pm * (np.log(pm) - np.log(pd))).sum()))
    return float(np.mean(kls))

def unique_fraction(samples: np.ndarray) -> float:
    """Fraction of unique sequences in the sample set."""
    uniq = np.unique(samples, axis=0)
    return float(len(uniq)) / max(len(samples), 1)

def gen_ppl_gpt2(text_samples: list[str], model_name: str = "gpt2-large",
                 device: str | torch.device = "cuda", batch_size: int = 8,
                 max_length: int = 512) -> dict[str, float]:
    """Compute generation perplexity under a HuggingFace causal LM judge.

    Args:
        text_samples: list of decoded strings.
        model_name: any causal HF model.
        device: torch device.

    Returns dict with mean_ppl, mean_nll_per_tok, n.
    """

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    mdl.eval()

    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for i in range(0, len(text_samples), batch_size):
            batch = text_samples[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(device)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            # shift-and-score
            out = mdl(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[:, :-1]
            labels = input_ids[:, 1:]
            loss_mask = attn[:, 1:].float()
            logp = F.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, labels[..., None]).squeeze(-1)                # [B, T-1]
            total_nll += float((nll * loss_mask).sum().item())
            total_tok += int(loss_mask.sum().item())

    mean_nll = total_nll / max(total_tok, 1)
    return {
        "gen_ppl": float(math.exp(mean_nll)),
        "mean_nll_per_tok": mean_nll,
        "n_tokens": total_tok,
        "n_samples": len(text_samples),
    }

# =========================================================
# 7) Wall-clock / NFE utility
# =========================================================


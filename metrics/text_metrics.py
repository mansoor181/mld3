"""Text sample statistics: generative perplexity, unigram entropy, unigram KL to the data."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def unigram_entropy(samples: np.ndarray, vocab_size: int) -> float:
    """Mean per-position unigram entropy, in nats."""
    ent = []
    for i in range(samples.shape[1]):
        counts = np.bincount(samples[:, i], minlength=vocab_size).astype(np.float64)
        p = counts / counts.sum()
        ent.append(-(p * np.log(np.clip(p, 1e-300, None))).sum())
    return float(np.mean(ent))


def unigram_kl(samples: np.ndarray, reference: np.ndarray, vocab_size: int,
               eps: float = 1e-6) -> float:
    """KL from the samples' per-position unigram marginals to the data's, averaged over L."""
    kls = []
    for i in range(samples.shape[1]):
        p = np.bincount(samples[:, i], minlength=vocab_size).astype(np.float64)
        q = np.bincount(reference[:, i], minlength=vocab_size).astype(np.float64)
        p, q = p / p.sum() + eps, q / q.sum() + eps
        p, q = p / p.sum(), q / q.sum()
        kls.append((p * (np.log(p) - np.log(q))).sum())
    return float(np.mean(kls))


@torch.no_grad()
def generative_ppl(texts: list[str], device, model_name: str = "gpt2-large",
                   batch_size: int = 8, max_length: int = 512) -> float:
    """Perplexity of decoded samples under a frozen causal judge."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(device)
        logits = judge(**enc).logits[:, :-1]
        labels, mask = enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:].float()
        nll = -F.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)
        total_nll += float((nll * mask).sum())
        total_tokens += int(mask.sum())
    return math.exp(total_nll / max(total_tokens, 1))

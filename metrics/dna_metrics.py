"""Frechet Biological Distance and activity Wasserstein for the DeepSTARR arm.

A generated enhancer is scored by a trained regressor rather than by a parser, which makes the
arm a statement about function rather than syntax. The FBD compares Gaussian fits of the
oracle's penultimate embeddings on generated and held-out real sequences, and the Wasserstein
distance compares the two predicted-activity distributions on each head.

The oracle is the released PyTorch port `multimolecule/deepstarr`, rebuilt here rather than
imported, because that package's initialiser pulls in a dependency needing a newer torch. The
modules are named so the released state dict loads without key surgery.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy import linalg
from scipy.stats import wasserstein_distance

_ORACLE_ID = "multimolecule/deepstarr"
LABEL_NAMES = ("dev", "hk")


# The oracle's own alphabet, read off the tokenizer that ships with the released weights.
# The encoder consumes a one-hot over these five symbols and nothing else, so we index the
# characters directly and never build a tokenizer object.
ORACLE_ALPHABET = "ACGTN"
ORACLE_SEQ_LEN = 249

# The released architecture, taken from the config that ships with the weights.
_CONV_CHANNELS = (256, 60, 60, 120)
_CONV_KERNELS = (7, 3, 5, 3)
_FC_DIMS = (256, 256)
_POOL = 2
_BN_EPS = 1e-3
_BN_MOMENTUM = 0.1


class DeepStarrOracle(torch.nn.Module):
    """The released DeepSTARR regressor.

    Four convolutional blocks and two fully connected layers, matching the config that
    ships with the released weights.
    """

    def __init__(self, vocab_size: int = 5, num_labels: int = 2, seq_len: int = ORACLE_SEQ_LEN):
        super().__init__()
        nn = torch.nn
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        blocks = []
        in_ch = vocab_size
        for out_ch, k in zip(_CONV_CHANNELS, _CONV_KERNELS):
            blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding="same"),
                nn.BatchNorm1d(out_ch, _BN_EPS, _BN_MOMENTUM),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=_POOL),
            ))
            in_ch = out_ch
        self.blocks = torch.nn.ModuleList(blocks)
        length = seq_len
        for _ in _CONV_CHANNELS:
            length //= _POOL
        in_feat = _CONV_CHANNELS[-1] * length
        layers = []
        for out_feat in _FC_DIMS:
            layers.append(nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat, _BN_EPS, _BN_MOMENTUM),
                nn.ReLU(),
            ))
            in_feat = out_feat
        self.fc = torch.nn.ModuleList(layers)
        self.decoder = nn.Linear(in_feat, num_labels)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Penultimate representation, which is what the head reads."""
        h = torch.nn.functional.one_hot(ids.clamp(0, self.vocab_size - 1), self.vocab_size)
        h = h.to(self.decoder.weight.dtype).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.flatten(1)
        for layer in self.fc:
            h = layer(h)
        return h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.embed(ids))


def _released_state_dict(model_id: str):
    """Fetch the released weights and rename them onto our module layout."""
    sd = load_file(hf_hub_download(model_id, "model.safetensors"))
    out = {}
    for key, value in sd.items():
        new = (key.replace("model.encoder.blocks.", "blocks.")
                  .replace("model.pooler.layers.", "fc.")
                  .replace("sequence_head.decoder.", "decoder."))
        # Our blocks are Sequential, so conv is child 0, norm is child 1, and the linear in
        # each fully connected layer is child 0 with its norm at child 1.
        new = new.replace(".conv.", ".0.").replace(".dense.", ".0.").replace(".norm.", ".1.")
        out[new] = value
    return out


@lru_cache(maxsize=2)
def load_oracle(model_id: str = _ORACLE_ID, device: str = "cuda") -> DeepStarrOracle:
    """Load the DeepSTARR regressor in eval mode with gradients switched off."""
    model = DeepStarrOracle()
    missing, unexpected = model.load_state_dict(_released_state_dict(model_id), strict=False)
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"DeepSTARR port mismatch: missing={real_missing[:4]} "
                           f"unexpected={list(unexpected)[:4]}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_for_oracle(seqs: list[str]) -> torch.Tensor:
    """Index a list of sequence strings over the oracle's own alphabet."""
    lut = np.full(256, ORACLE_ALPHABET.index("N"), dtype=np.int64)
    for i, ch in enumerate(ORACLE_ALPHABET):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return torch.from_numpy(lut[arr].reshape(len(seqs), -1))


@torch.no_grad()
def predict(seqs: list[str], device: str = "cuda", batch_size: int = 256,
            model_id: str = _ORACLE_ID) -> tuple[np.ndarray, np.ndarray]:
    """Score sequences with the oracle.

    Returns (activities [N, 2], embeddings [N, D]) where the embedding is the penultimate
    representation that the regression head reads.
    """
    model = load_oracle(model_id, device)
    ids_all = encode_for_oracle(seqs)
    if ids_all.shape[1] != ORACLE_SEQ_LEN:
        raise ValueError(f"DeepSTARR expects {ORACLE_SEQ_LEN} bp, got {ids_all.shape[1]}")
    acts, embs = [], []
    for i in range(0, len(seqs), batch_size):
        ids = ids_all[i:i + batch_size].to(device)
        emb = model.embed(ids)
        acts.append(model.decoder(emb).float().cpu())
        embs.append(emb.float().cpu())
    return torch.cat(acts).numpy(), torch.cat(embs).numpy()


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between the Gaussian fits of two embedding sets."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(ca.dot(cb), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(ca) + np.trace(cb) - 2.0 * np.trace(covmean))


def evaluate(gen_seqs: list[str], real_seqs: list[str], device: str = "cuda",
             real_activity: np.ndarray | None = None, model_id: str = _ORACLE_ID) -> dict:
    """Score a generated set against a held-out real set.

    `real_activity` may hold the measured labels, in which case the top-decile threshold is
    taken from the measurements rather than from the oracle's own predictions on the real set.
    """
    g_act, g_emb = predict(gen_seqs, device=device, model_id=model_id)
    r_act, r_emb = predict(real_seqs, device=device, model_id=model_id)
    ref_act = r_act if real_activity is None else np.asarray(real_activity, dtype=np.float64)

    out = {"n_gen": len(gen_seqs), "n_real": len(real_seqs),
           "fbd": frechet_distance(g_emb, r_emb)}
    for j, name in enumerate(LABEL_NAMES):
        out[f"w1_{name}"] = float(wasserstein_distance(g_act[:, j], ref_act[:, j]))
        out[f"mean_{name}_gen"] = float(g_act[:, j].mean())
        out[f"mean_{name}_real"] = float(ref_act[:, j].mean())
    thresh = float(np.percentile(ref_act[:, 0], 90))
    out["top_decile_thresh_dev"] = thresh
    out["pi90_dev"] = float((g_act[:, 0] > thresh).mean())
    return out

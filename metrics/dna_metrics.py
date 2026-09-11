"""Regulatory-DNA metrics for the DeepSTARR arm.

A generated enhancer is scored by a trained regressor rather than by a parser, which is what
makes this arm a statement about function instead of about syntax. We use the released
PyTorch port of the DeepSTARR convolutional oracle, `multimolecule/deepstarr`, so that nothing
in this project has to depend on TensorFlow.

Three numbers come out of the oracle. The Frechet Biological Distance compares the Gaussian
fits of the oracle's penultimate-layer embeddings on generated and on held-out real sequences,
which is the sequence analogue of the Frechet Inception Distance. The Wasserstein distance on
each of the two predicted activity heads compares the generated activity distribution with the
real one. The top-decile fraction reports how much of the generated set the oracle places above
the 90th percentile of the real developmental activity, which is the quantity a designer cares
about.

Nothing here is trustworthy until `calibrate` reproduces the oracle's published held-out
correlation, and `scripts/gate_deepstarr_oracle.py` runs that check.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy import linalg
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

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
    """A standalone port of the released DeepSTARR regressor.

    We rebuild the four convolutional blocks and the two fully connected layers here rather
    than importing `multimolecule`, whose package initialiser pulls in a dependency that
    requires a newer torch than the environment carries. The module list is named so that
    the released state dict loads into it without any key surgery, and `tests` on the
    published held-out correlation confirm the port end to end.
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


def calibrate(seqs: list[str], labels: np.ndarray, device: str = "cuda",
              model_id: str = _ORACLE_ID) -> dict:
    """Pearson and Spearman correlation of the oracle against measured labels.

    This is the hard gate on the arm. If the port does not reproduce the published held-out
    correlation then no number downstream of it means anything.
    """
    pred, _ = predict(seqs, device=device, model_id=model_id)
    labels = np.asarray(labels, dtype=np.float64)
    out = {"n": len(seqs)}
    for j, name in enumerate(LABEL_NAMES):
        out[f"pearson_{name}"] = float(pearsonr(pred[:, j], labels[:, j])[0])
        out[f"spearman_{name}"] = float(spearmanr(pred[:, j], labels[:, j])[0])
    return out


# ---------------- motif diagnostics ----------------

# Core developmental and housekeeping motifs reported for Drosophila STARR-seq enhancers.
# The counts are a diagnostic rather than a headline number, because a consensus string match
# is a coarse stand-in for a position weight matrix scan.
MOTIFS = {
    "dev_GATA": "GATAA",
    "dev_AP1": "TGACTCA",
    "dev_twist": "CATATG",
    "hk_DRE": "TATCGATA",
    "hk_Ohler1": "GTGTGACCG",
    "hk_Ohler6": "AAGTGTGA",
}


def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def motif_counts(seqs: list[str]) -> dict:
    """Fraction of sequences containing each consensus motif on either strand."""
    out = {}
    n = max(len(seqs), 1)
    for name, motif in MOTIFS.items():
        rc = _revcomp(motif)
        hits = sum(1 for s in seqs if motif in s or rc in s)
        out[f"motif_{name}"] = hits / n
    return out


__all__ = ["DeepStarrOracle", "load_oracle", "encode_for_oracle", "predict",
           "frechet_distance", "evaluate", "calibrate", "motif_counts", "MOTIFS",
           "LABEL_NAMES", "ORACLE_ALPHABET", "ORACLE_SEQ_LEN"]

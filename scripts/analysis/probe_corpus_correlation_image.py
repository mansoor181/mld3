"""Put every corpus on one correlation axis, including the image tokens.

`probe_corpus_correlation.py` measures pairwise mutual information between positions for the
molecule and DNA corpora, whose alphabets are 40, 72 and 6 symbols. The image tokens carry 1024
symbols and GPT-2 text carries 50257, and a plug-in mutual information estimator on a raw
alphabet of that size is almost all bias, hence the raw numbers would not be comparable. We hash
every alphabet through one fixed random map into a common bucket count, which can only lower the
mutual information by the data processing inequality and gives every corpus the same estimator
bias at the same sample size. We then subtract the value the same estimator returns on far-apart
random pairs, which is the bias floor for that corpus.

Separation is measured along the raveled sequence. The image grid is 16 by 16, hence separation 1
is the horizontal neighbor and separation 16 is the vertical one.

Runs on CPU so that it does not contend with training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import paths
from data.text import make_loader
from data.mols import MOL_ROOT, _REGISTRY as MOL_SPECS

N_SEQ = 40000       # sequences per corpus
BUCKETS = 64        # common alphabet after hashing
SEPS = (1, 2, 5, 16, 30)
FLOOR_PAIRS = 64    # random far pairs used as the bias floor
SEED = 0
TOPK = 1024        # cap on the native alphabet, rarer symbols fold into one bucket


def mi_nats(a: np.ndarray, b: np.ndarray, B: int) -> float:
    """Plug-in mutual information in nats between two integer columns."""
    J = np.bincount(a * B + b, minlength=B * B).astype(np.float64).reshape(B, B)
    J /= J.sum()
    pi = J.sum(1, keepdims=True)
    pj = J.sum(0, keepdims=True)
    nz = J > 0
    return float((J[nz] * (np.log(J[nz]) - np.log((pi * pj)[nz]))).sum())


def entropy_nats(a: np.ndarray, B: int) -> float:
    p = np.bincount(a, minlength=B).astype(np.float64)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def agree(a: np.ndarray, b: np.ndarray, B: int) -> float:
    """Excess probability that two positions carry the same symbol, over the independent value."""
    pa = np.bincount(a, minlength=B).astype(np.float64) / a.size
    pb = np.bincount(b, minlength=B).astype(np.float64) / b.size
    return float((a == b).mean() - (pa * pb).sum())


def profile(name: str, x: np.ndarray, V: int, buckets: int | None = BUCKETS) -> dict:
    """`x` is [N, L] of integers in [0, V)."""
    rng = np.random.default_rng(SEED)
    if buckets is None:
        # The native alphabet is unusable at the GPT-2 vocabulary, because the joint histogram
        # would need V^2 bins. We keep the TOPK most frequent symbols and fold the rest into one
        # bucket, which is again a coarsening and again a lower bound.
        if V > TOPK:
            order = np.argsort(np.bincount(x.ravel(), minlength=V))[::-1][:TOPK - 1]
            table = np.full(V, TOPK - 1, dtype=np.int64)
            table[order] = np.arange(TOPK - 1)
            h, B = table[x], TOPK
        else:
            h, B = x, V
    else:
        h, B = rng.integers(0, buckets, size=V)[x], buckets
    N, L = h.shape

    out = {"corpus": name, "L": L, "V": V, "N": N, "B": B}
    out["H"] = float(np.mean([entropy_nats(h[:, i], B) for i in
                              rng.choice(L, size=min(L, 32), replace=False)]))

    floor, afloor = [], []
    for _ in range(FLOOR_PAIRS):
        i, j = rng.choice(L, size=2, replace=False)
        floor.append(mi_nats(h[:, i], h[:, j], B))
        afloor.append(agree(h[:, i], h[:, j], B))
    # The floor is the median over pairs that are at least a quarter of the sequence apart, which
    # keeps genuine long-range structure out of the estimate of the bias.
    out["floor"] = float(np.median(floor))
    out["afloor"] = float(np.median(afloor))

    for d in SEPS:
        if d >= L:
            out[f"sep{d}"] = None
            continue
        idx = np.arange(0, L - d)
        if idx.size > 32:
            idx = rng.choice(idx, size=32, replace=False)
        out[f"sep{d}"] = float(np.mean([mi_nats(h[:, i], h[:, i + d], B) for i in idx]))
        out[f"agr{d}"] = float(np.mean([agree(h[:, i], h[:, i + d], B) for i in idx]))
    return out


def load_tensor(path: Path, n: int) -> np.ndarray:
    blob = torch.load(str(path), map_location="cpu")
    if isinstance(blob, dict):
        for k in ("x_clean", "tokens", "input_ids", "x", "data"):
            if k in blob:
                blob = blob[k]
                break
    return blob[:n].to(torch.int64).numpy()


def load_text(name: str, n: int) -> tuple[np.ndarray, int]:
    loader, meta = make_loader(name, "val", 512, shuffle=False, num_workers=0)
    xs, got = [], 0
    for xb in loader:
        xb = xb[0] if isinstance(xb, (list, tuple)) else xb
        xs.append(xb)
        got += xb.shape[0]
        if got >= n:
            break
    return torch.cat(xs)[:n].to(torch.int64).numpy(), int(meta.get("vocab_size", 50257))


def collect() -> list[np.ndarray]:
    """Return `(name, tokens, V)` for every corpus staged on this machine."""
    out = []
    img_root = paths.IMAGE_ROOT / "imagenet256"
    meta = json.loads((img_root / "meta.json").read_text())
    out.append(("imagenet256", load_tensor(img_root / "train.pt", N_SEQ), meta["codebook_size"]))

    for name in ("wikitext103", "lm1b"):
        try:
            x, V = load_text(name, N_SEQ)
        except Exception as exc:  # corpus not staged on this machine
            print(f"[skip] {name}: {exc}", file=sys.stderr)
            continue
        out.append((name, x, V))

    for name in ("qm9", "zinc250k"):
        spec = MOL_SPECS.get(name)
        p = Path(MOL_ROOT) / spec["subdir"] / "parsed" / "train.pt"
        if spec is None or not p.exists():
            continue
        x = load_tensor(p, N_SEQ)
        out.append((name, x, int(x.max()) + 1))
    return out


def report(rows: list[dict], label: str) -> None:
    print(f"\n=== {label} ===")
    hdr = "%-13s %5s %6s %6s %8s %8s " % ("corpus", "L", "V", "B", "H", "floor")
    hdr += " ".join("%9s" % f"xs{d}" for d in SEPS) + "  " + " ".join(
        "%9s" % f"agr{d}" for d in SEPS)
    print(hdr)
    for r in rows:
        line = "%-13s %5d %6d %6d %8.4f %8.5f " % (
            r["corpus"], r["L"], r["V"], r["B"], r["H"], r["floor"])
        line += " ".join(
            "%9s" % ("%.5f" % (r[f"sep{d}"] - r["floor"]) if r.get(f"sep{d}") is not None else "-")
            for d in SEPS)
        line += "  " + " ".join(
            "%9s" % ("%.5f" % (r[f"agr{d}"] - r["afloor"]) if r.get(f"sep{d}") is not None else "-")
            for d in SEPS)
        print(line)
    print("xs = excess mutual information over the far-pair floor, in nats.")
    print("agr = excess probability that the two positions carry the same symbol.")
    for r in rows:
        if r.get(f"sep{SEPS[0]}") is None:
            continue
        print("%-13s excess/H at sep1 = %.4f" % (
            r["corpus"], (r[f"sep{SEPS[0]}"] - r["floor"]) / max(r["H"], 1e-9)))


def main() -> None:
    data = collect()
    report([profile(n, x, V, buckets=BUCKETS) for n, x, V in data],
           f"hashed to {BUCKETS} buckets, equal estimator bias")
    report([profile(n, x, V, buckets=None) for n, x, V in data],
           "native alphabet, far-pair floor removes the bias")


if __name__ == "__main__":
    main()

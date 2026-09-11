"""Image-quality metrics for the ImageNet-256 VQ-token arm.

Turns the token grids the sampler produces into pixels through the MaskGIT VQGAN and scores
them with FID, Inception score, precision and recall. We drive the ReDi baseline's own
`MultiInceptionMetrics`, which wraps the `torch_fidelity` InceptionV3 port, so that our numbers
are produced by the same feature extractor the image baselines report against.

We import `Metrics.inception_metrics` directly rather than through `Metrics.sample_and_eval`,
whose third line is `import clip` and which would drag in a CLIP install we have no use for.

Decoding is streamed. Ten thousand decoded images at float32 come to 30 GB, while one chunk of
64 as uint8 is 12 MB, and nothing is ever written to disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from omegaconf import OmegaConf

from paths import IMAGE_ROOT, REDI_IMAGE_DIR

# The VQGAN and the Inception metrics live in the vendored ReDi image tree. We import them
# rather than copying, so that a future upstream change shows up as a diff instead of silently
# diverging, which is why the tree goes on the path before those two imports.
sys.path.insert(0, REDI_IMAGE_DIR)
from Metrics.inception_metrics import MultiInceptionMetrics
from Network.Taming.models.vqgan import VQModel

MASKGIT_DIR = IMAGE_ROOT / "maskgit"
REF_DIR = IMAGE_ROOT / "ref"
CODEBOOK = 1024


def build_vqgan(device: torch.device, path: Path = MASKGIT_DIR):
    """Load the MaskGIT VQGAN, frozen and in eval mode."""
    cfg = OmegaConf.load(str(path / "model.yaml"))
    vq = VQModel(**cfg.model.params)
    sd = torch.load(str(path / "last.ckpt"), map_location="cpu")["state_dict"]
    # The checkpoint also carries the discriminator and the LPIPS network, neither of which
    # VQModel defines, hence strict=False. A missing encoder or decoder key would instead mean
    # the config and the checkpoint disagree, which we refuse rather than absorb.
    missing, _ = vq.load_state_dict(sd, strict=False)
    hard = [k for k in missing if k.startswith(("encoder.", "decoder.", "quantize.",
                                                "quant_conv.", "post_quant_conv."))]
    if hard:
        raise RuntimeError(f"VQGAN is missing core weights: {hard[:8]}")
    return vq.eval().to(device).requires_grad_(False)


@torch.no_grad()
def tokens_to_uint8(tokens: np.ndarray | torch.Tensor, vq, device: torch.device,
                    chunk: int = 64, grid: int = 16) -> Iterator[torch.Tensor]:
    """Yield uint8 [chunk, 3, 256, 256] images from [N, 256] or [N, 16, 16] token ids.

    The rescaling reproduces the chain the ReDi baseline uses at
    `Metrics/sample_and_eval.py:167-170`, so a decoded image here is bit-identical to one
    decoded there.
    """
    t = torch.as_tensor(np.asarray(tokens))
    if t.ndim == 2:
        t = t.view(-1, grid, grid)
    for i in range(0, t.shape[0], chunk):
        # A position still holding the absorbing MASK has no codebook entry. It can only appear
        # if a sampler left the grid incomplete, which would make every metric meaningless, so
        # we stop rather than clamp it into codebook entry 1023.
        block = t[i:i + chunk].to(device).long()
        if int(block.max()) >= CODEBOOK:
            raise ValueError(f"token id {int(block.max())} is not a codebook entry; the sample "
                             f"is still partly masked and cannot be decoded")
        img = vq.decode_code(block)
        img = img.float().clamp(-1.0, 1.0) * 0.5 + 0.5
        yield (img * 255).round().clamp(0, 255).to(torch.uint8)


def build_metric(device: torch.device, manifold_k: int = 3):
    """A MultiInceptionMetrics configured for the unconditional comparison we run."""
    return MultiInceptionMetrics(
        reset_real_features=False,
        compute_unconditional_metrics=True,
        compute_conditional_metrics=False,
        compute_conditional_metrics_per_class=False,
        manifold_k=manifold_k,
    ).to(device)


@torch.no_grad()
def inception_features(metric, images: torch.Tensor) -> torch.Tensor:
    """Run the Inception extractor over one uint8 batch and return the 2048-d features."""
    feats, _ = metric.inception(images)
    return feats.view(feats.size(0), -1).double()


def reference_path(name: str) -> Path:
    return REF_DIR / f"{name}.npz"


def load_reference(name: str, device: torch.device) -> torch.Tensor:
    """Load a cached [N, 2048] reference feature array.

    `MultiInceptionMetrics.compute` concatenates the real features unconditionally at line 406
    even when precomputed FID moments are supplied, and precision and recall need the individual
    features rather than the moments in any case, so we cache the features themselves.
    """
    p = reference_path(name)
    if not p.exists():
        raise FileNotFoundError(f"missing reference features {p}; build them with "
                                f"`python scripts/build_image_refs.py`")
    return torch.from_numpy(np.load(p)["features"]).to(device).double()


def score(metric, fake_uint8_batches: Iterator[torch.Tensor],
          reference: torch.Tensor) -> dict[str, float]:
    """Score a stream of generated uint8 batches against cached reference features.

    The metric object is reset first and the reference is injected directly, which saves
    re-extracting features for fifty thousand real images at every point of the NFE grid.
    """
    metric.reset()
    metric.real_features = [reference]
    for batch in fake_uint8_batches:
        metric.update(batch, image_type="unconditional")
    out = metric.compute()
    return {k.replace("_unconditional", ""): float(v) for k, v in out.items()}

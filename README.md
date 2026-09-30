# Mixture-of-Latents for Few-Step Discrete Diffusion Distillation (MLD3)

Reference implementation. An MLD3 student replaces the factorized step of a masked diffusion
model with a mixture of `M` factorized components selected by a learned router,

    p_theta(x_s | x_t) = sum_k w_k(x_t, t) prod_i P_{i,k}(x_s^i | x_t, t),

which makes the step likelihood a closed-form sum over `M` terms and gives the mixing weights
a gradient. At `M = 1` it reduces to a masked diffusion student.

```
paths.py            external roots, resolved from the environment and an optional .env
models/trunk.py     DiT trunk: shared blocks, router head, component-specific top blocks
models/mld3.py      the mixture kernel and its closed-form step likelihood
models/teachers.py  frozen MDLM and MaskGIT teachers, and the analytic rollout
models/wtask_start.py  copy an MDLM backbone into a student (molecules and DNA)
training/losses.py  L_trans + lambda_0 * L_0 + R, and the time sampling
training/train.py   the trainer
sampling/sampler.py spread consensus, commit-k and best-of-M
data/               one loader per corpus, plus decoding
metrics/            text, molecule, DNA and image metrics
eval/evaluate.py    sample across a budget grid and score
configs/            one config per reported task
tests/              correctness checks for the kernel and the samplers
```

## Tasks and teachers

| task      | corpus                          | L   | teacher                    |
|-----------|---------------------------------|-----|----------------------------|
| text      | LM1B                            | 128 | MDLM                       |
| text      | WikiText-103                    | 128 | MDLM                       |
| molecules | QM9, ZINC-250k                  | 32, 74 | MDLM (PairFlow fork)    |
| DNA       | DeepSTARR                       | 249 | MDLM (PairFlow fork)       |
| images    | ImageNet-256 VQGAN f16 tokens   | 256 | frozen MaskGIT             |

## Setup

```bash
pip install -r requirements.txt
```

Point the data roots at your own directories, either through the environment or through a
`.env` file beside `paths.py`:

```
MLD3_TEXT_ROOT=/data/text
MLD3_MOL_ROOT=/data/mols
MLD3_DNA_ROOT=/data/dna
MLD3_IMAGE_ROOT=/data/imagenet256
MLD3_BASELINES=/data/baselines      # mdlm, pairflow, maskgit, redi checkouts
MLD3_TEACHERS=/data/teachers
MLD3_RESULTS=/data/results
```

The text and molecule caches are the pre-tokenized dumps the MDLM and PairFlow repositories
build, so the comparison uses one tokenization per corpus. The other two are built here:

```bash
python scripts/prep_deepstarr.py
python scripts/prep_imagenet256.py
python scripts/build_image_refs.py     # Inception references for FID
```

Teachers are loaded from their own checkpoints through `models/teachers.py`, which imports the
DiT class from the baseline checkout by file path, since both trees ship a `models` package.

## Training

```bash
python -m training.train --config configs/lm1b_M4.yaml
```

The objective is the transition loss of Eq. 6, the auxiliary loss of Eq. 11 on a quarter of
each micro-batch, and the router entropy regularizer. Half of the time pairs are drawn off the
`K`-step schedule with a log-uniform step size, so that the objective rather than the density
of supervision is what separates this from methods that sample a fine discretization.

## Sampling and evaluation

```bash
python -m eval.evaluate --ckpt results/lm1b_M4/ckpt_0050000.pt --out results/lm1b_M4/eval.json
```

`--sampler spread` is the default and is the spread consensus of Section 4.3: each step reveals
exactly the number of positions the schedule expects, taken in a randomly rotated bit-reversal
order, and only the `--components` highest-weight components are evaluated, each proposing a
continuation that the renormalised mixture scores. `--sampler commit` holds one component for
the whole trajectory and `--sampler bestofm` runs one such trajectory per component.

Costs are reported in transformer-block passes, since a call over every component costs
`C(M) = D - L_lat + M*L_lat` while a call naming one costs `D`. With `D = 12` and `L_lat = 4`,
spread consensus at two components costs 16 per step at any `M`.

## Tests

```bash
python -m pytest tests -q
```

# MLDF: Mixture-of-Latents for Distilling Flows

This is the codebase for the MLDF. It distills a pretrained masked discrete diffusion teacher into a student whose reverse kernel is a mixture of factorized heads sharing a router, which lets a single student step express correlated joint outcomes that a factorized step cannot represent. This repo holds one trainer, two evaluation entry points, and the scripts that drive them across five corpora. 

```
paths.py        every external root, resolved from the environment and the .env file
data/           one loader per corpus family: text, mols, dna, images
models/         trunk.py (the shared backbone), latent_kernel.py (the mixture kernel),
                teacher_adapters.py (frozen MDLM and MaskGIT teachers), wtask_start.py
sampling/       sampler.py (ancestral and analytic decoders), policies.py (the decoding policies)
metrics/        text statistics, plus molecule, DNA and image metric stacks
training/       trainer_distill.py (the trainer), common.py (seeding, EMA, config loading)
eval/           eval_text.py (the scoring sweep, all domains), eval_policies.py (the policy sweep)
configs/        the 28 cell configs behind the paper's tables
scripts/        prep, train, eval, collect, figures, analysis, baselines and infra drivers
tests/          correctness checks, cost probes and decoder diagnostics
```

## The five tasks

| task | corpus | teacher |
|---|---|---|
| text | LM1B, 128 tokens | MDLM-equivalent x0 model in our own checkpoint format |
| text | WikiText-103, 128 tokens | MDLM |
| molecules | QM9, ZINC-250k | MDLM trained through the PairFlow fork |
| DNA | DeepSTARR, 249 bp | MDLM trained through the PairFlow fork |
| images | ImageNet-256 VQ tokens, 256 positions | frozen MaskGIT |

`data/data.md` gives the corpus, teacher and reference artifact for each task, with the script that produces every one.

## Setup

The experiments ran under Python 3.10 with the pins in `requirements.txt`. The molecule and DNA metric stacks need extra packages that the text, and those are listed separately at the bottom of that file.

```bash
python -m pip install -r requirements.txt
source scripts/env.sh          # data roots, HuggingFace caches, allocator settings
```


## Quickstart

A short test trains for a few hundred steps and exercises the whole path, from the loader through the teacher to a checkpoint:

```bash
GPU=0 CELLS=smoke_lm1b_r2 bash scripts/train/train_cells.sh
```

Then score a finished cell on the ten-point budget grid the paper uses:

```bash
RESULTS=$MLDF_RESULTS/lm1b_distill GPU=0 POLICY=cons \
  CELLS="lm1b_M4_K4_seed0:0050000" bash scripts/eval/eval_cells.sh
```


## Running the checks

```bash
python -m tests.test_wtask_start        # wtask start preserves every single-component route
python -m tests.test_di4c_parity       # the Di4C reimplementation matches the reference term by term
python -m tests.test_supervision       # the supervision path: time sampling, the x0 KL, its gradient
python -m tests.test_policy_analytic --ckpt <a checkpoint>   # the analytic decoder's reveal schedule
```

`models/architecture.md` describes what each module does and how a training step and an evaluation sweep flow through them.



# Reproducing the results

Each section below gives the chain from raw data to the numbers in the manuscript. Every command assumes `source scripts/env.sh` has run and that the working directory is this tree.

The evaluation grid is the same everywhere: NFEs 1 through 8 plus 16 and 32, the analytic decoder, and the three decoding policies of Section 6.3. Consensus commit is the headline policy for any cell with more than one component.

## Common shape

```bash
# 1. train one or more cells on a GPU
GPU=0 CELLS="<cell> <cell>" bash scripts/train/train_cells.sh

# 2. score them on the full grid under a policy
RESULTS=$MLDF_RESULTS/<task> GPU=0 POLICY=commit CELLS="<cell>:<step>" bash scripts/eval/eval_cells.sh
RESULTS=$MLDF_RESULTS/<task> GPU=0 POLICY=cons   CELLS="<cell>:<step>" bash scripts/eval/eval_cells.sh

# 3. turn the JSONs into table rows
python scripts/collect/collect_analytic.py --root $MLDF_RESULTS/<task> --cell <cell>
```

## Text tables (LM1B and WikiText-103)

Cells: `lm1b_r2_M1`, `lm1b_r2_M4`, `lm1b_di4c_M1`, `lm1b_di4c_M4`, `wt103_r2_M1`, `wt103_r2_M4`, `wt103_r2_M8`, plus the supervision `lm1b_r0_*`, `lm1b_r1_*` and the sampling ablation `lm1b_r2_mc_M4`.

```bash
GPU=0 CELLS="lm1b_r2_M4 lm1b_di4c_M4" bash scripts/train/train_cells.sh
RESULTS=$MLDF_RESULTS/lm1b_distill GPU=0 POLICY=commit CELLS="lm1b_M4_K4_seed0:0050000" bash scripts/eval/eval_cells.sh
RESULTS=$MLDF_RESULTS/lm1b_distill GPU=0 POLICY=cons   CELLS="lm1b_M4_K4_seed0:0050000" bash scripts/eval/eval_cells.sh
RESULTS=$MLDF_RESULTS/lm1b_distill GPU=0 POLICY=bomev  CELLS="lm1b_M4_K4_seed0:0050000" bash scripts/eval/eval_cells.sh
```

The decoding-policy table comes from a single sweep per cell, and the cost-adjusted control from the high-budget driver:

```bash
RESULTS=$MLDF_RESULTS/lm1b_distill GPU=0 CELLS="lm1b_M1_K4_seed0:0050000 lm1b_M4_K4_seed0:0050000" bash scripts/eval/eval_policies.sh
RESULTS=$MLDF_RESULTS/lm1b_distill GPU=1 CELLS="lm1b_M1_K4_seed0:0050000" bash scripts/eval/eval_highnfe.sh
```

The paired reranking control, which gives a factorized student the same candidate budget as a mixture, is `scripts/eval/bestofn_control.py`.

## Molecules and DNA

The teachers are MDLM models trained through the PairFlow codebase; `data/data.md` records their checkpoints. 

```bash
GPU=0 CELLS="qm9_mdlmT_r2_M8 zinc250k_mdlmT_r2_M8 deepstarr_mdlmT_r2_M8" bash scripts/train/train_cells.sh
RESULTS=$MLDF_RESULTS/v2_nontext GPU=0 POLICY=cons CELLS="qm9_mdlmT_r2_M8" bash scripts/eval/eval_cells.sh
python scripts/collect/collect_mol_analytic.py
python scripts/collect/collect_consensus_mol.py
```

The DNA oracle must pass its workflow before its numbers mean anything:

```bash
python scripts/infra/gate_deepstarr_oracle.py
```

## Images

Prepare the corpus and the references once, check the gates, anchor the teacher, then train and score.

```bash
python scripts/prep/fetch_image_task.py          # teacher, VQGAN, token dump, raw val pixels
python scripts/prep/prep_imagenet256.py         # token tensors and meta.json
python scripts/prep/build_image_refs.py         # val_real and val_recon Inception caches
python scripts/infra/image_gates.py             # the checks that had to pass before spending a GPU day
python scripts/infra/image_teacher_anchor.py    # the teacher anchor table

GPU=0 CELLS="imagenet256_maskgitT_di4c_M8 imagenet256_maskgitT_r2_M1" LOG_DIR=$ROOT/logs/image bash scripts/train/train_cells.sh
GPU=1 CELLS="imagenet256_maskgitT_r2_M8 imagenet256_maskgitT_r2_M4"   LOG_DIR=$ROOT/logs/image bash scripts/train/train_cells.sh

GPU=0 CELLS="imagenet256_maskgitT_r2_M8" STEP=0020000 POLICIES="commit cons" bash scripts/eval/eval_image_cells.sh
```

Commit-k and consensus commit are scored at 10,000 samples, and best-of-M at 2,500, because best-of-M costs twelve times as much per sample. 

## The Di4C baseline

Besides our compute-matched reimplementation, the comparison is with authors' own pipeline run end to end from their repository. Our part of that experiments are in `scripts/baselines/`: exporting our teacher into the checkpoint format their trainer expects (`export_mol_teacher_ema_for_di4c.py`, `convert_mol_teacher_for_di4c.py`), and scoring their finished image students with our metrics so the numbers are on the same grid and references as every other row (`eval_di4c_authors_image.py`). 



## Acknowledgements

Several components build on the publicly available code of the methods that we compare against:

- [MDLM](https://github.com/kuleshov-group/mdlm) (Sahoo et al., 2024). The text teachers are MDLM checkpoints loaded through their DiT implementation, our EMA tracker mirrors the semantics of their `models/ema.py`, and the absorbing noise schedule follows their `LogLinearNoise`.
- [Di4C](https://github.com/sony/di4c) (Hayakawa et al., 2024). The Di4C baseline in every table is our own reimplementation of their objective inside our trunk, checked term by term against their released loss by `tests/test_di4c_parity.py`; the frozen ImageNet MaskGIT teacher is loaded from the MaskGIT-pytorch fork vendored in their repository.
- [PairFlow](https://github.com/KAIST-Visual-AI-Group/PairFlow) (Park et al., 2025). The molecule and DNA teachers are trained with their fork of the MDLM recipe, and their preprocessed QM9 and ZINC-250k dumps define our molecule corpora.
- [ReDi](https://github.com/Ugness/ReDi_discrete) (Kim et al., 2025). The image evaluation decodes tokens through the Taming VQGAN and scores them with the Inception metric stack from their image tree.
The trunk itself is a DiT-style backbone in the MDLM configuration.
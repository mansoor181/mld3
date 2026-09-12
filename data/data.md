# Data, teachers and vendored code

Every artifact an experiment reads, what produces it, and where it lives.

## Roots

| variable | holds |
|---|---|
| `MLDF_TEXT_ROOT` | tokenized text corpora |
| `MLDF_MOL_ROOT` | molecule token dumps (defaults into the PairFlow checkout) |
| `MLDF_DNA_ROOT` | the DeepSTARR corpus |
| `MLDF_IMAGE_ROOT` | image tokens, teacher, reference features, results |
| `MLDF_BASELINES` | the baseline repository checkouts |
| `MLDF_RESULTS` | run output directories |
| `MLDF_TEACHERS` | teacher checkpoints trained with the baseline repositories |

## Text

Corpora are HuggingFace Arrow caches under `$MLDF_TEXT_ROOT/<dataset>/`. LM1B uses a 128-token block with the BERT vocabulary of 30522; WikiText-103 uses a 128-token block with the GPT-2 vocabulary of 50257. The absorbing MASK is appended past the end of the data vocabulary, which `maybe_extend_vocab_for_mask` handles.

Teachers: the LM1B cells distil an MDLM-equivalent x0 model stored in our own checkpoint format at `$MLDF_TEACHERS/track2_*/...`, loaded by `teacher.kind: lkf`. The WikiText-103 cells distil a released MDLM checkpoint at `$MLDF_TEACHERS/baselines_text/mdlm_wikitext103/checkpoints/last.ckpt`, loaded by `teacher.kind: mdlm`.

Generative perplexity is judged by `gpt2-large` from the HuggingFace hub.

## Molecules

QM9 at 32 positions and ZINC-250k at 74, read from `$MLDF_MOL_ROOT/<name>/parsed/{train,valid}.pt`, which are the PairFlow fork's own preprocessed dumps. These corpora reserve a MASK inside their vocabulary, so the config passes `mask_id` explicitly rather than extending the vocabulary.

Tokenizers and reference SMILES come from the hub (`yairschiff/qm9-tokenizer`, `yairschiff/zinc250k-tokenizer`, and the matching datasets for the reference set). Teachers are MDLM models trained through the PairFlow fork and stored at `$MLDF_RESULTS/mol_teacher/<name>_mdlm/checkpoints/last.ckpt`.

The molecule metrics need `rdkit` and `fcd_torch`, which are optional in `requirements.txt`.

## DNA

DeepSTARR enhancer sequences at 249 bases over `ACGTN` with MASK at index 5, produced by `scripts/prep/prep_deepstarr.py` from the hub and written to `$MLDF_DNA_ROOT/deepstarr/{train,valid,test}.pt`. `scripts/prep/prep_deepstarr_pairflow.py` re-expresses the same corpus in the layout the PairFlow trainer wants, which is how the teacher is trained.

Held-out activity is scored by a ported oracle from the hub (`multimolecule/deepstarr`). `scripts/infra/gate_deepstarr_oracle.py` checks the port against the published correlation before the arm is trusted.

## Images

ImageNet-256 as VQ tokens: a 16 by 16 grid flattened to 256 positions, codebook 1024, MASK in vocabulary at 1024, 1000 class labels.

| artifact | size | produced by |
|---|---|---|
| `$MLDF_IMAGE_ROOT/imagenet256/{train,val}.pt` | 666 MB, 26 MB | `scripts/prep/prep_imagenet256.py` |
| `$MLDF_IMAGE_ROOT/maskgit/MaskGIT_ImageNet_256.pth` | 2.0 GB | `scripts/prep/fetch_image_arm.py` |
| `$MLDF_IMAGE_ROOT/maskgit/last.ckpt` (VQGAN) | 958 MB | same |
| `$MLDF_IMAGE_ROOT/ref/val_real.npz` | 410 MB | `scripts/prep/build_image_refs.py` |
| `$MLDF_IMAGE_ROOT/ref/val_recon.npz` | 410 MB | same, gives the tokenizer floor |

The teacher is class-conditional and we evaluate it unconditionally, so a label mode has to be chosen. Drawing a fresh class at every reverse step is worse than passing the null class, because one trajectory then conditions successive steps on unrelated classes. Drawing one class per row and holding it for the whole rollout gives the class marginal and is what the arm uses.

## Vendored baselines

Under `$MLDF_BASELINES`, each a pinned clone. The live code touches them only through `models/teacher_adapters.py` and `metrics/image_metrics.py`.

| directory | upstream | commit | used for |
|---|---|---|---|
| `mdlm/` | github.com/kuleshov-group/mdlm | `c112c526` | the MDLM DiT class behind the text and non-text teachers |
| `pairflow/` | github.com/KAIST-Visual-AI-Group/PairFlow | `0bdc5432` | the molecule and DNA teacher recipe and its DiT variant |
| `di4c/` | github.com/sony/di4c | `ac61ff9f` | the MaskGIT transformer, and the reference Di4C loss the parity test checks against |
| `redi/` | github.com/Ugness/ReDi_discrete | `45290923` | the VQGAN decoder and the Inception metric stack |


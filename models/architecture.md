# Architecture

This document describes what each module does and how a training step and an evaluation sweep flow through them. 

## The MLDF student

A student is a `LatentKernelFlow` (`models/latent_kernel.py`) wrapping a `LatentKernelTrunk` (`models/trunk.py`). The trunk is a DiT-style stack of `D` blocks split into a shared stack of `D - L_lat` blocks and a latent stack of `L_lat` blocks that is evaluated once per mixture component. A router head on top of the shared stack produces the mixture weights. Because only the latent stack is per-component, marginalizing over all `M` components costs a fraction of a forward pass rather than `M` forward passes, which is what makes the mixture affordable and is the cost model the decoding policies price against.


`mixture_logprob` is the one method the rest of the code calls. It returns the log mixture probability of a transition, the log router weights, and the per-component log probabilities, which is everything the objective and the policies need.

## A training step

`training/trainer_distill.py` is the only trainer. One step does the following.

1. Draw a batch and a time pair. `sample_two_times` picks `(t, s)` either on the `K`-step evaluation grid or off it, with a log-uniform horizon so that short horizons are not starved. Half the rows come from each source by default.
2. Noise the batch to `x_t` under the absorbing forward process (`mask_forward_noise`).
3. Roll the frozen teacher forward from `t` to `s` in several sub-steps (`teacher_rollout`), which is the target the student must match in one step. The teacher is loaded once by `build_teacher` and never updated.
4. Score the teacher's endpoint under the student's mixture kernel and take the negative log likelihood of that transition.
5. Add the `x0` KL term, which compares the student's `k`-marginalized denoiser against the teacher's on a fraction of rows. Marginalizing before taking the KL is the point: matching per component would let each component match the marginal separately and the mixture would carry no correlation.


## An evaluation sweep

`eval/eval_text.py` handles every domain. It resolves the domain from the dataset name, loads the checkpoint preferring EMA weights, and then for each budget on the NFE grid it generates samples under the chosen decoder and policy and scores them with the domain's metric stack. Text is scored by generative perplexity under a frozen GPT-2 judge plus entropy and uniqueness statistics; molecules by validity, uniqueness, novelty and a Frechet distance; DNA by an oracle predictor and motif counts; images by decoding tokens back to pixels through the VQGAN and scoring Frechet distance, Inception score, precision and recall against a cached reference.

Two decoders exist. The ancestral decoder samples `x_s` from the learned two-time kernel and lets the network choose its own reveal rate. The analytic decoder enforces the schedule's reveal count. The analytic decoder is the headline sampler in every table, because a trunk that ignores its destination time commits the same fraction per step under the ancestral rollout at every budget and therefore saturates early, which would flatter or penalize arms unequally.

`eval/eval_policies.py` runs the whole policy family over one checkpoint in a single pass and is the source of the decoding-policy tables.

## Decoding policies

`sampling/policies.py` implements the policies and prices each one in block passes, with `C(M)` the cost of a call that evaluates all `M` branches. The important ones:

- commit-k fixes one component for the whole trajectory and costs `K*D`, exactly the factorized price.
- consensus commit keeps one chain and, at every step, draws one proposal per component from the single call that already enumerates them, then advances with the proposal the router-weighted mixture scores highest. It costs `K*C(M)` and can realize component sequences no committed rollout can reach.
- best-of-R rolls out `R` committed chains and returns the one a selector prefers. The running evidence the trunk already accumulates is a free selector and beats a Monte Carlo estimate of the mixture NELBO, so it is what the paper reports.



# Experiment Design

## Question

Does the project's complete RLHF procedure improve convergence or annotation
efficiency beyond ordinary SFT when all conditions receive the same ground-truth
corrections from the same initial policy and interaction audio?

## Arms

| Arm | Training after each annotation round |
|---|---|
| SFT control | Plain LoRA SFT on GT correction text |
| Clean RLHF | Existing SFT, reward model, synthetic generation, DPO, and WER-gated promotion with oracle preferences |
| Noisy RLHF | The identical production pipeline after deterministically reversing the configured fraction of actionable oracle preferences |

All arms receive identical GT correction budgets. The RLHF arms additionally
use the A/B preference implied by each GT comparison. This tests the
incremental value of the preference pipeline rather than comparing it with an
untrained model.

## Surrogate Policy

For reference transcript $r$ and hypotheses $h_a$ and $h_b$, compute normalized
word error rates $W(r,h_a)$ and $W(r,h_b)$ using the same lexical normalization
as model-health evaluation. With tie threshold $\tau$:

$$
\operatorname{preference} =
\begin{cases}
a & W(r,h_a) + \tau < W(r,h_b) \\
b & W(r,h_b) + \tau < W(r,h_a) \\
\text{equal} & \text{otherwise}
\end{cases}
$$

The full reference is submitted as SFT correction text. Missing GT is never
fabricated: configuration either fails immediately or explicitly skips the
pair.

## Isolation

Each replicate begins from one common initial state cloned into independent SFT,
clean-RLHF, and noisy-RLHF arms. Mutable paths are arm-local and selected before
application
imports through `RLHF_PATHS_FILE`. Experiments must not mutate the normal
database, active-policy pointer, checkpoints, reward models, logs, or published
model directory.

## Data Partitions

The completed runner will persist three disjoint audio partitions before any
training:

1. Interaction pool for pair generation and surrogate feedback.
2. Promotion monitor for the existing RLHF deployment gate.
3. Final holdout used only to compare all conditions.

All arms use the same interaction audio IDs per round. The promotion monitor
and final holdout are excluded from every learning query.

## Reproducibility

The manifest records configuration, Git commit, seeds, data identities, initial
policy, runtime versions, GPU selection, and arm paths. Pair schedules must be
materialized before any arm executes. Completed stages will be checkpointed
atomically so resume cannot submit duplicate annotations or repeat training.

## Metrics

Per round, compare all arms on the same final holdout using:

- corpus WER and CER;
- paired clean-RLHF-minus-SFT, noisy-RLHF-minus-SFT, and noisy-minus-clean
  corpus-WER deltas with clip-resampled 95% intervals;
- improvement from the common initial policy;
- cumulative annotations and area under WER-versus-annotation curve;
- optimizer updates, losses, runtime, and available GPU memory metrics;
- RLHF promotion, guarded rejection, or failure outcome.

Corpus WER is computed from summed word errors and reference words, never by
averaging per-clip WER.

## Evidence Standard

The smoke configuration validates mechanics only. Development should use at
least three seeds. A reportable comparison should use five or more independent
seeds and retain per-seed outcomes plus paired uncertainty intervals. A guarded
RLHF candidate is a valid result, not a harness failure; a failed training stage
is reported as a failed replicate/round.

## Current Boundary

Workspace preparation, GT annotation, deterministic audio scheduling, clean
catalogue initialization, exact monitor materialization, and initial-state
cloning are implemented. Matched pair generation and HTTP surrogate annotation
are implemented with seeded sampling, per-arm resume state, and parity reports.
Plain-SFT execution and canonical coordinator execution are implemented with
arm-local outputs and explicit guarded outcomes. Final-holdout evaluation and
multi-seed statistical reporting are implemented. Automatic recovery of an
interrupted GPU training process remains intentionally fail-closed; the stage
is inspected rather than silently repeated.

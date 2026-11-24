# Direct Preference Optimisation — A Practical Primer

**Audience:** Researchers and engineers familiar with neural networks but new to RLHF.  
**Date:** 2026-07-15

---

## 1. The Problem DPO Solves

Training a language (or speech) model to produce outputs that humans prefer is hard. The classical approach — **Reinforcement Learning from Human Feedback (RLHF)** — requires three stages:

1. Collect pairwise human preferences: *"is output A or output B better?"*
2. Train an explicit **reward model** on those preferences.
3. Run **PPO** (Proximal Policy Optimisation) to fine-tune the policy model against the reward model.

PPO is notoriously unstable, computationally expensive, and requires the reward model to be available in GPU memory *during* policy training.

**DPO eliminates stage 3 entirely** — and in most cases stage 2 as well — by showing that the optimal policy can be derived directly from preference data.

---

## 2. The Core Insight

DPO starts from the standard RLHF objective: maximise expected reward while staying close to a reference policy $\pi_{ref}$:

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot \mathbb{KL}\!\left[\pi_\theta(\cdot|x) \;\|\; \pi_{ref}(\cdot|x)\right]$$

Rafailov et al. (2023) prove that **this objective has a closed-form optimal solution**:

$$\pi^*(y \mid x) \propto \pi_{ref}(y \mid x) \cdot \exp\!\left(\frac{r(x,y)}{\beta}\right)$$

Rearranging this expression, the reward function can be written in terms of the optimal policy and the reference:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)$$

where $Z(x)$ is a partition function that cancels out in pairwise comparisons. This means: **the reward is implicitly encoded in the ratio of policy to reference probabilities** — no separate reward model is needed.

---

## 3. The DPO Loss Function

Substituting the expression above into the **Bradley-Terry preference model** (which says a human prefers $y_w$ over $y_l$ with probability $\sigma(r_w - r_l)$), the partition function cancels and we get:

$$\boxed{L_{DPO}(\pi_\theta) = -\log \sigma\!\left(\beta \left[\log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right]\right)}$$

### Notation

| Symbol | Name | Meaning |
|--------|------|---------|
| $x$ | Input | The conditioning signal (prompt, audio, image, …) |
| $y_w$ | Winning / chosen output | The output annotators preferred |
| $y_l$ | Losing / rejected output | The output annotators rejected |
| $\pi_\theta$ | Policy | The model being trained (parameters $\theta$ are updated) |
| $\pi_{ref}$ | Reference policy | A frozen copy of the model before fine-tuning |
| $\sigma(\cdot)$ | Sigmoid | Squashes input to $(0, 1)$; here a preference probability |
| $\beta$ | KL coefficient | Controls how far $\pi_\theta$ may deviate from $\pi_{ref}$ |

---

## 4. Reading the Equation in Plain Language

The term inside $\sigma$ computes:

> *"How much more has the trained model shifted toward the winner than toward the loser, relative to what the reference model already knew?"*

- If $\pi_\theta$ strongly prefers $y_w$ over $y_l$ beyond the reference: bracket is large and positive → $\sigma \approx 1$ → $-\log \approx 0$ → **loss is small**.
- If $\pi_\theta$ gets it backwards (prefers the loser): bracket is negative → loss is large.

The training signal is thus not about absolute probabilities, but about **relative shifts from the prior**.

---

## 5. What the Reference Model Does

$\pi_{ref}$ serves two roles simultaneously:

**a) Prevents catastrophic forgetting**  
Without the reference, DPO could push the model so far toward preferred outputs that it forgets everything it learned during pre-training (in an ASR context: how to decode audio at all). The log-ratio keeps the model anchored to its original distribution.

**b) Prevents reward hacking**  
Without the reference, the trivially optimal solution is to assign probability 1 to $y_w$ and 0 to everything else — a degenerate lookup table. The KL penalty $\beta \cdot \mathbb{KL}[\pi_\theta \| \pi_{ref}]$ in the original objective prevents this.

In practice, $\pi_{ref}$ is usually the model after supervised fine-tuning (SFT) — the best "prior" behaviour you want to preserve while improving preferences.

---

## 6. What DPO Requires

To apply DPO you need:

1. A dataset of **triplets** $(x, y_w, y_l)$ — input + preferred output + rejected output.
2. A frozen **reference model** $\pi_{ref}$ (typically the SFT checkpoint).
3. Teacher-forced **log-probabilities** of $y_w$ and $y_l$ under both $\pi_\theta$ and $\pi_{ref}$.  
   For an autoregressive model: $\log \pi(y|x) = \sum_t \log \pi(y_t \mid x, y_{<t})$.

No reward model, no rollouts, no online sampling during training.

---

## 7. DPO for Encoder-Decoder Models (e.g. Whisper)

The original DPO paper targets autoregressive language models. For an encoder-decoder model like Whisper:

- $x$ = audio log-mel spectrogram → encoder hidden states
- $y$ = token sequence → decoded by the decoder
- $\log \pi_\theta(y \mid x)$ = sum of **teacher-forced** cross-entropy log-probabilities across all decoder timesteps (using the full reference sequence, not sampled tokens)

The loss is identical in form; only the forward pass differs.

---

## 8. Limitations and Known Issues

### a) Offline / static training data
DPO is an **offline** algorithm. Both $y_w$ and $y_l$ are generated once and frozen. If the current policy has diverged significantly from the policy that generated the pairs, the gradient signal becomes unreliable. This is sometimes called the **distribution shift problem**.

Mitigation: regenerate pairs from the current policy before each DPO round (*iterative DPO*).

### b) DPO on on-policy data ≈ supervised fine-tuning
When both $y_w$ and $y_l$ are generated by $\pi_{ref}$ itself, $\log \pi_{ref}(y_w|x)$ is already high (it is a probable output). The log-ratio term is small, and the training signal approximates maximising $\log \pi_\theta(y_w|x)$ — i.e., plain supervised fine-tuning on the winner. The rejection component is present but weak.

### c) No exploration
The model never generates outputs outside the fixed pair set. Novel, potentially better hypotheses are never discovered during training.

---

## 9. Variants and Follow-ups

| Paper | Key contribution |
|-------|-----------------|
| **DPO** — Rafailov et al., NeurIPS 2023 ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) | Original formulation |
| **IPO** — Azar et al., AISTATS 2024 ([arXiv:2310.12036](https://arxiv.org/abs/2310.12036)) | Fixes overfitting when preferences are deterministic; replaces log $\sigma$ with a squared loss |
| **ORPO** — Hong et al., 2024 ([arXiv:2403.07691](https://arxiv.org/abs/2403.07691)) | Removes the reference model entirely using an odds-ratio formulation |
| **RSO** — Liu et al., Google, 2023 ([arXiv:2309.06657](https://arxiv.org/abs/2309.06657)) | Addresses offline distribution shift via statistical rejection sampling |

---

## 11. Synthetic DPO Data Generation with a Reward Model

One of the most practical extensions of DPO is using a trained reward model to **generate preference pairs automatically** — eliminating the need for human annotation of every training example.

### The loop

Once a reward model $RM(x, y) \to \mathbb{R}$ is available (trained on an initial set of human-labelled pairs), preference data can be generated at scale:

```
For each input x in your dataset:
  1. Sample N candidate outputs: y_1, …, y_N ~ π_θ(· | x)
  2. Score each:  r_i = RM(x, y_i)
  3. y_w = argmax_i r_i       ← chosen
     y_l  = argmin_i  r_i     ← rejected
  4. If |r_w - r_l| < τ: discard (near-tie, weak signal)
  5. Otherwise: add (x, y_w, y_l) to DPO training set
```

The threshold $\tau$ filters out ambiguous pairs where the reward model is not confident enough to produce a reliable label.

### Why this is powerful

Human annotations only ever need to train the reward model once (or periodically retrain it as more annotations accumulate). After that, the reward model acts as an **automatic preference labeller** that scales to the full dataset:

```
~100 human annotations
        │
        ▼
  Train RM(audio, hypothesis) → scalar reward
        │
        ▼
  Generate thousands of (audio, y_w, y_l) triples
  using N-best sampling + RM scoring
        │
        ▼
  DPO fine-tuning on synthetic pairs
        │
        ▼
  Updated policy π_θ → regenerate pairs → repeat
```

### Iterative variant (addresses distribution shift)

After each DPO round the policy $\pi_\theta$ has changed. Regenerating pairs from the **updated** policy before the next round ensures $y_w$ and $y_l$ are always on-policy — directly addressing the distribution shift limitation from section 8.

This is the **iterative DPO** or **online DPO** regime:

| Round | Pairs generated from | Labelled by |
|-------|---------------------|-------------|
| Bootstrap | Fixed decoding strategies | Humans |
| Round 1 | Current policy (N-best) | Reward model |
| Round 2 | Updated policy (N-best) | Reward model |
| … | … | … |

### Connection to RSO

This approach is formalised in **Rejection Sampling Optimisation (RSO)** (Liu et al., Google, 2023 — [arXiv:2309.06657](https://arxiv.org/abs/2309.06657)), already listed in section 9. In this project the two decoding passes provide candidate diversity, while `active_policy_checkpoint` determines the policy that generates both passes. The reward model ranks those candidates; it is not the generation policy.

At each rollover, unannotated pairs from the previous policy are retired and the same audio becomes eligible under the new `model_version`. Annotated pairs remain available as historical preference data.

### Practical consideration: how many candidates N?

More candidates per input gives a larger reward gap between $y_w$ and $y_l$, producing cleaner signal:

- $N = 2$: equivalent to the current two-strategy setup, but now labelled by the reward model
- $N = 4$–$8$: good balance of diversity and compute cost
- $N \geq 16$: diminishing returns; best-of-16 is usually near the ceiling for typical model variance

For ASR with short utterances, $N = 4$–$8$ sampled hypotheses per clip is a reasonable default.

---

## 12. One-Sentence Summary

DPO replaces reward modelling + PPO with a single supervised-style loss that directly optimises the policy to assign higher probability to preferred outputs than rejected ones — relative to a frozen reference — while the KL term to that reference prevents catastrophic forgetting and reward hacking.

# Original Design Rationale

This file records the durable intent of the original 2026-06-27 design. It is
not the current implementation specification. The complete pre-consolidation
design remains available in Git history.

For current behavior use:

- [CONTRACT.md](CONTRACT.md) for invariants and post-mortem rules.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for current modules and data flow.
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for the operating workflow.
- [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) for persistence details.
- [docs/DPO_PRIMER.md](docs/DPO_PRIMER.md) for algorithm background.

## Durable Intent

- Improve Polish medical ASR through pairwise human feedback.
- Keep annotation continuous: annotators submit independently of training rounds.
- Use a browser UI backed by local SQLite, with UUID4 identifiers for safe merge.
- Keep the annotation service CPU-only and isolate GPU workloads in the
  coordinator.
- Fine-tune Whisper with LoRA so checkpoints stay small and comparable.
- Generate future pairs from the improved policy, not indefinitely from the
  bootstrap model.
- Preserve an auditable before/after WER record for every successful policy
  update.

## Superseded Ideas

The original design discussed Gemma/RLAIF, REINFORCE, PPO, diverse-beam search,
and a Tkinter annotation client. These are not active features. Their removal
and replacements are documented chronologically in
[docs/MAINTENANCE_LOG.md](docs/MAINTENANCE_LOG.md).
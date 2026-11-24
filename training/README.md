# Training component

This directory owns the GPU image recipe. Build context is `../rlhf/`; the
coordinator and all learning code remain in that shared source tree.

The root Compose service executes `scripts/coordinator.py` inside the image.
Its existing subprocess orchestration is retained: generation, reward training,
optional SFT, temporary synthetic preferences, DPO and evaluation.

Use the root [README](../README.md) and [Compose configuration](../compose.yaml).
Do not run a second coordinator through the imported upstream Compose file.
TTS orchestration, Large-v3 migration and converted-model publication remain
future work.

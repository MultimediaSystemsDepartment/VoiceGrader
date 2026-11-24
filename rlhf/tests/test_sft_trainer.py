from pathlib import Path
import json
from types import SimpleNamespace

import torch

import rlhf.sft_trainer as sft_module
from rlhf.sft_trainer import WhisperSFTTrainer, _effective_warmup_steps
from utils.training_control import EarlyStopping


class _Store:
    def __init__(self):
        self.settings = {}

    def get_sft_corrections(self):
        return [
            {"file_path": "first.wav", "correction_text": "pierwsza"},
            {"file_path": "second.wav", "correction_text": "druga"},
        ]

    def set_setting(self, key, value):
        self.settings[key] = value


class _Collator:
    def __init__(self, processor, config):
        pass

    def __call__(self, batch):
        size = len(batch)
        return {
            "input_features": torch.zeros(size, 1),
            "labels": torch.zeros(size, 1, dtype=torch.long),
        }


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.dtype = torch.float32
        self.config = SimpleNamespace(forced_decoder_ids=None)
        self.base_model = SimpleNamespace(model=self._forward_base)
        self.saved_to = None

    def _forward_base(self, input_features, labels):
        return SimpleNamespace(loss=self.weight.square())

    def save_pretrained(self, output_dir):
        self.saved_to = Path(output_dir)


class _Processor:
    def get_decoder_prompt_ids(self, language, task):
        return []

    def save_pretrained(self, output_dir):
        pass


def test_sft_disables_long_warmup_for_small_round() -> None:
    warmup_steps, planned_updates = _effective_warmup_steps(
        batches_per_epoch=11,
        num_epochs=3,
        gradient_accumulation_steps=8,
        requested_warmup_steps=10,
    )

    assert planned_updates == 6
    assert warmup_steps == 0


def test_early_stopping_counts_stale_epochs_and_can_be_disabled() -> None:
    stopping = EarlyStopping(patience=2)

    assert stopping.update(1.0) is False
    assert stopping.update(0.9) is False
    assert stopping.update(0.91) is False
    assert stopping.update(0.92) is True

    disabled = EarlyStopping(patience=0)
    assert disabled.update(1.0) is False
    assert disabled.update(1.1) is False


def test_sft_flushes_partial_gradient_accumulation_at_epoch_tail(monkeypatch, tmp_path):
    config = SimpleNamespace(
        batch_size=1,
        learning_rate=5e-5,
        warmup_steps=10,
        device="cpu",
        num_epochs=1,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        early_stopping_patience=0,
        language="pl",
        task="transcribe",
        logging_steps=10,
        save_steps=200,
    )
    model = _Model()
    trainer = WhisperSFTTrainer(config, _Store(), paths=None)
    monkeypatch.setattr(sft_module, "_SFTCollator", _Collator)
    monkeypatch.setattr(trainer, "_load_policy", lambda: (model, _Processor()))

    result = trainer.train(tmp_path / "sft")

    assert result == tmp_path / "sft"
    assert model.saved_to == result


def test_sft_freezes_candidates_caps_each_epoch_and_records_provenance(monkeypatch, tmp_path):
    class Store(_Store):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def get_setting(self, key, default):
            return "50"

        def get_sft_training_candidates(self):
            self.reads += 1
            return [
                {"audio_id": "recorded", "file_path": "r.wav", "correction_text": "human",
                 "origin": "recorded", "dataset": "recorded"},
                *[{"audio_id": f"tts{i}", "file_path": f"t{i}.wav", "correction_text": "requested",
                   "origin": "synthesized", "utterance_id": "same-text", "dataset": "TTSCompare",
                   "provenance_json": '{"job_id":"bounded-job"}'} for i in range(3)],
            ]

    config = SimpleNamespace(
        batch_size=1, learning_rate=5e-5, warmup_steps=10, device="cpu",
        num_epochs=2, gradient_accumulation_steps=8, max_grad_norm=1.0,
        early_stopping_patience=0, language="pl", task="transcribe",
        logging_steps=10, save_steps=200,
    )
    store = Store()
    trainer = WhisperSFTTrainer(config, store, paths=None)
    monkeypatch.setattr(sft_module, "_SFTCollator", _Collator)
    monkeypatch.setattr(trainer, "_load_policy", lambda: (_Model(), _Processor()))
    result = trainer.train(tmp_path / "sft")
    snapshot = json.loads((result / "sft_data_snapshot.json").read_text())
    assert store.reads == 1
    assert len(snapshot["epochs"]) == 2
    for epoch in snapshot["epochs"]:
        assert epoch["origins"] == {"recorded": 1, "synthesized": 1}
        assert epoch["total"] == 2
        assert any(r["provenance_json"] for r in epoch["samples"])

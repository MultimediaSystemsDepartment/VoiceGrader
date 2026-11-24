from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference.config import InferenceConfig
from inference.decoding_strategies import BEAM_5
from inference.hypothesis_generator import HypothesisGenerator, _clip_generation_seed
from paths import PathsConfig
from rlhf.config import RLHFConfig
from rlhf.trainer import WhisperRLHFTrainer, _scaled_optimizer_step
from run_generate_pairs import _generate_nbest, _load_audio_ids


def test_generation_kwargs_lock_polish_transcription() -> None:
    kwargs = BEAM_5.to_generate_kwargs()

    assert kwargs["language"] == "pl"
    assert kwargs["task"] == "transcribe"


def test_generation_rejects_config_lineage_mismatch_before_creating_database(
    tmp_path, monkeypatch
) -> None:
    import run_generate_pairs

    config_file = tmp_path / "medium.json"
    config_file.write_text('{"model_id": "openai/whisper-medium"}', encoding="utf-8")
    db_path = tmp_path / "fresh.sqlite"
    monkeypatch.setattr(run_generate_pairs, "DEFAULT_PATHS", PathsConfig(db_path=str(db_path)))
    monkeypatch.setattr(
        sys, "argv", ["run_generate_pairs.py", "--config", str(config_file), "--catalogue_only"]
    )
    with pytest.raises(ValueError, match="configured lineage"):
        run_generate_pairs.main()
    assert not db_path.exists()


@pytest.mark.parametrize("mel_bins", [80, 128])
def test_training_collators_preserve_model_specific_features(monkeypatch, mel_bins) -> None:
    from rlhf.dataset import WhisperRLHFCollator
    from rlhf.sft_trainer import _SFTCollator

    features = torch.zeros((1, mel_bins, 3000))

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 0]] * len(texts)))

    processor = SimpleNamespace(
        feature_extractor=lambda *args, **kwargs: SimpleNamespace(input_features=features),
        tokenizer=Tokenizer(),
    )
    for module in ("rlhf.dataset", "rlhf.sft_trainer"):
        monkeypatch.setattr(
            f"{module}.load_audio", lambda path: (torch.zeros(16_000).numpy(), 16_000)
        )
    batch = [{
        "file_path": "fixture.wav", "correction_text": "Corrected.",
        "chosen_hyp": "Chosen.", "rejected_hyp": "Rejected.",
    }]
    for collator in (WhisperRLHFCollator, _SFTCollator):
        result = collator(processor, RLHFConfig())(batch)
        assert result["input_features"].shape == (1, mel_bins, 3000)
        assert result["input_features"] is features


@pytest.mark.parametrize("loader", ["generation", "sft", "dpo", "reference", "evaluation"])
def test_incompatible_adapters_fail_before_model_import(tmp_path, loader) -> None:
    from evaluation.evaluator import WhisperEvaluator
    from rlhf.reference_model import ReferenceWhisper
    from rlhf.sft_trainer import WhisperSFTTrainer

    checkpoint = tmp_path / "medium"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "openai/whisper-medium"}', encoding="utf-8"
    )
    config = RLHFConfig(resume_from_checkpoint=str(checkpoint))
    paths = PathsConfig()
    loaders = {
        "generation": lambda: HypothesisGenerator(
            object(), InferenceConfig(), paths, checkpoint
        ).load_model(),
        "sft": lambda: WhisperSFTTrainer(config, object(), paths)._load_policy(),
        "dpo": lambda: WhisperRLHFTrainer(config, object(), paths)._load_policy(),
        "reference": lambda: ReferenceWhisper(adapter_checkpoint=str(checkpoint)).load(),
        "evaluation": lambda: WhisperEvaluator(
            model_id=config.whisper_model_id, paths=paths, store=object(),
            lora_checkpoint=str(checkpoint),
        )._load_model(),
    }
    with pytest.raises(ValueError, match="does not match"):
        loaders[loader]()

def test_generation_loads_policy_adapter_and_records_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "whisper_dpo_20260731_120000"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "openai/whisper-large-v3"}', encoding="utf-8"
    )
    calls = {}

    class FakeBase:
        def to(self, device):
            calls["device"] = device
            return self

    class FakeModel:
        def eval(self):
            calls["eval"] = True

    class FakeWhisperModel:
        @staticmethod
        def from_pretrained(model_id, torch_dtype, attn_implementation):
            calls["model_id"] = model_id
            calls["attn_implementation"] = attn_implementation
            return FakeBase()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id):
            calls["processor_id"] = model_id
            return object()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, adapter):
            calls["adapter"] = Path(adapter)
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperForConditionalGeneration=FakeWhisperModel,
            WhisperProcessor=FakeProcessor,
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))

    config = InferenceConfig(device="cpu", torch_dtype="float16")
    generator = HypothesisGenerator(
        store=object(),
        config=config,
        paths=PathsConfig(),
        policy_checkpoint=checkpoint,
    )
    generator.load_model()

    assert calls["adapter"] == checkpoint
    assert calls["eval"] is True
    assert calls["attn_implementation"] == "sdpa"
    assert generator.model_version == (
        "openai/whisper-large-v3+lora:whisper_dpo_20260731_120000"
    )


def test_generation_passes_features_by_keyword_for_peft(
    monkeypatch,
) -> None:
    calls = []

    class FakeInputs:
        input_features = torch.zeros((1, 80, 10))

    class FakeProcessor:
        def __call__(self, *args, **kwargs):
            return FakeInputs()

        def batch_decode(self, ids, skip_special_tokens):
            return ["transkrypcja"] * len(ids)

    class KeywordOnlyPeftModel:
        device = "cpu"
        dtype = torch.float32

        def generate(self, **kwargs):
            calls.append(kwargs)
            batch_size = kwargs["input_features"].shape[0]
            return torch.zeros((batch_size, 2), dtype=torch.long)

    monkeypatch.setattr(
        "inference.hypothesis_generator.load_audio",
        lambda path: (torch.zeros(16_000).numpy(), 16_000),
    )
    generator = HypothesisGenerator(
        store=object(),
        config=InferenceConfig(device="cpu", batch_size=1),
        paths=PathsConfig(),
    )
    generator._processor = FakeProcessor()
    generator._model = KeywordOnlyPeftModel()

    candidates = generator._transcribe_files_nbest(["audio.wav"], n_candidates=2)

    assert candidates == [["transkrypcja", "transkrypcja"]]
    assert len(calls) == 2
    assert all("input_features" in kwargs for kwargs in calls)


def test_generation_seed_is_applied_before_sampling(monkeypatch) -> None:
    observed_seeds = []

    class FakeInputs:
        input_features = torch.zeros((1, 80, 10))

    class FakeProcessor:
        def __call__(self, *args, **kwargs):
            return FakeInputs()

        def batch_decode(self, ids, skip_special_tokens):
            return ["transkrypcja"] * len(ids)

    class SeedObservingModel:
        device = "cpu"
        dtype = torch.float32

        def generate(self, **kwargs):
            observed_seeds.append(torch.initial_seed())
            return torch.zeros((1, 2), dtype=torch.long)

    monkeypatch.setattr(
        "inference.hypothesis_generator.load_audio",
        lambda path: (torch.zeros(16_000).numpy(), 16_000),
    )
    generator = HypothesisGenerator(
        store=object(),
        config=InferenceConfig(device="cpu", batch_size=1, generation_seed=137),
        paths=PathsConfig(),
    )
    generator._processor = FakeProcessor()
    generator._model = SeedObservingModel()

    generator._transcribe_files_nbest(["audio.wav"], n_candidates=2)

    assert observed_seeds[-1] == _clip_generation_seed(137, "audio.wav")


def test_generation_seed_is_clip_stable_across_batch_context(monkeypatch) -> None:
    observed_seeds = []

    class FakeInputs:
        def __init__(self, batch_size):
            self.input_features = torch.zeros((batch_size, 80, 10))

    class FakeProcessor:
        def __call__(self, *args, **kwargs):
            return FakeInputs(len(args[0]))

        def batch_decode(self, ids, skip_special_tokens):
            return ["transkrypcja"] * len(ids)

    class SeedObservingModel:
        device = "cpu"
        dtype = torch.float32

        def generate(self, **kwargs):
            observed_seeds.append(torch.initial_seed())
            batch_size = kwargs["input_features"].shape[0]
            return torch.zeros((batch_size, 2), dtype=torch.long)

    monkeypatch.setattr(
        "inference.hypothesis_generator.load_audio",
        lambda path: (torch.zeros(16_000).numpy(), 16_000),
    )
    generator = HypothesisGenerator(
        store=object(),
        config=InferenceConfig(device="cpu", batch_size=8, generation_seed=137),
        paths=PathsConfig(),
    )
    generator._processor = FakeProcessor()
    generator._model = SeedObservingModel()

    generator._transcribe_files_nbest(["first.wav", "target.wav"], n_candidates=2)
    seed_in_batch = observed_seeds[-1]
    observed_seeds.clear()
    generator._transcribe_files_nbest(["target.wav"], n_candidates=2)

    assert observed_seeds[-1] == seed_in_batch


def test_explicit_audio_schedule_parser_rejects_duplicates(tmp_path: Path) -> None:
    schedule = tmp_path / "audio_ids.json"
    schedule.write_text('["audio-1", "audio-1"]', encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        _load_audio_ids(schedule)


def test_explicit_audio_generation_preserves_schedule_order() -> None:
    audio = {
        "audio-1": SimpleNamespace(id="audio-1", file_path="first.wav"),
        "audio-2": SimpleNamespace(id="audio-2", file_path="second.wav"),
    }

    class Store:
        def __init__(self):
            self.inserted = []

        def get_eligible_audio_files_by_ids(self, audio_ids, model_version):
            return [audio[audio_id] for audio_id in audio_ids if audio_id in audio]

        def get_paired_audio_ids_by_ids(self, audio_ids, model_version):
            return set()

        def insert_hypothesis_pair(self, pair):
            self.inserted.append(pair)

    class Generator:
        model_version = "policy-v1"

        def generate_candidates(self, audio_files, n_candidates):
            return [
                [f"beam-{item.id}", f"sample-{item.id}"] for item in audio_files
            ]

    store = Store()
    generated = _generate_nbest(
        gen=Generator(),
        store=store,
        config=InferenceConfig(),
        n_to_generate=2,
        n_candidates=2,
        rm_checkpoint=None,
        scheduled_audio_ids=["audio-2", "audio-1"],
    )

    assert generated == 2
    assert [pair.audio_id for pair in store.inserted] == ["audio-2", "audio-1"]


def test_explicit_audio_generation_rejects_ineligible_id() -> None:
    class Store:
        def get_eligible_audio_files_by_ids(self, audio_ids, model_version):
            return []

        def get_paired_audio_ids_by_ids(self, audio_ids, model_version):
            return set()

    generator = SimpleNamespace(model_version="policy-v1")

    with pytest.raises(ValueError, match="audio-missing"):
        _generate_nbest(
            gen=generator,
            store=Store(),
            config=InferenceConfig(),
            n_to_generate=1,
            n_candidates=2,
            rm_checkpoint=None,
            scheduled_audio_ids=["audio-missing"],
        )


def test_explicit_audio_generation_rejects_candidate_collapse() -> None:
    audio = SimpleNamespace(id="audio-1", file_path="audio.wav")

    class Store:
        def get_eligible_audio_files_by_ids(self, audio_ids, model_version):
            return [audio]

        def get_paired_audio_ids_by_ids(self, audio_ids, model_version):
            return set()

        def insert_hypothesis_pair(self, pair):
            raise AssertionError("Collapsed candidates must not be inserted")

    generator = SimpleNamespace(
        model_version="policy-v1",
        generate_candidates=lambda audio_files, n_candidates: [["same", "same"]],
    )

    with pytest.raises(RuntimeError, match="0/1 pairs"):
        _generate_nbest(
            gen=generator,
            store=Store(),
            config=InferenceConfig(),
            n_to_generate=1,
            n_candidates=2,
            rm_checkpoint=None,
            scheduled_audio_ids=["audio-1"],
        )


def test_explicit_audio_generation_resumes_partial_schedule() -> None:
    remaining = SimpleNamespace(id="audio-2", file_path="second.wav")

    class Store:
        def __init__(self):
            self.inserted = []

        def get_eligible_audio_files_by_ids(self, audio_ids, model_version):
            return [remaining]

        def get_paired_audio_ids_by_ids(self, audio_ids, model_version):
            return {"audio-1"}

        def insert_hypothesis_pair(self, pair):
            self.inserted.append(pair)

    generator = SimpleNamespace(
        model_version="policy-v1",
        generate_candidates=lambda audio_files, n_candidates: [["beam", "sample"]],
    )
    store = Store()

    generated = _generate_nbest(
        gen=generator,
        store=store,
        config=InferenceConfig(),
        n_to_generate=2,
        n_candidates=2,
        rm_checkpoint=None,
        scheduled_audio_ids=["audio-1", "audio-2"],
    )

    assert generated == 1
    assert [pair.audio_id for pair in store.inserted] == ["audio-2"]


def test_unscheduled_generation_remains_supported() -> None:
    audio = SimpleNamespace(id="audio-1", file_path="audio.wav")

    class Store:
        def __init__(self):
            self.inserted = []

        def get_unqueued_audio_files(self, limit, model_version):
            return [audio]

        def insert_hypothesis_pair(self, pair):
            self.inserted.append(pair)

    generator = SimpleNamespace(
        model_version="policy-v1",
        generate_candidates=lambda audio_files, n_candidates: [["beam", "sample"]],
    )
    store = Store()

    generated = _generate_nbest(
        gen=generator,
        store=store,
        config=InferenceConfig(),
        n_to_generate=1,
        n_candidates=2,
        rm_checkpoint=None,
    )

    assert generated == 1
    assert len(store.inserted) == 1


def test_dpo_reference_defaults_to_resumed_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "whisper_dpo_previous"
    config = RLHFConfig(resume_from_checkpoint=str(checkpoint))
    trainer = WhisperRLHFTrainer(config, store=object(), paths=PathsConfig())

    reference = trainer._build_ref_model()

    assert reference.adapter_checkpoint == str(checkpoint)


def test_dpo_scaled_update_reports_only_successful_steps(caplog) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scales = iter([8.0, 4.0, 4.0])
    scaler = SimpleNamespace(
        unscale_=lambda optimizer: None,
        is_enabled=lambda: True,
        get_scale=lambda: next(scales),
        step=lambda optimizer: None,
        update=lambda: None,
    )
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    assert _scaled_optimizer_step(model, optimizer, scaler, 1.0) is None
    assert torch.equal(model.weight, before)
    assert "optimizer update skipped" in caplog.text

    model.weight.grad = torch.ones_like(model.weight)
    scaler.get_scale = lambda: 4.0
    scaler.step = lambda optimizer: optimizer.step()
    assert _scaled_optimizer_step(model, optimizer, scaler, 1.0) == pytest.approx(1.0)
    assert not torch.equal(model.weight, before)
    assert model.weight.grad is None


def test_dpo_nonfinite_unscaled_update_fails_before_changing_weights() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    with pytest.raises(RuntimeError, match="non-finite"):
        _scaled_optimizer_step(model, optimizer, scaler, 1.0)
    assert torch.equal(model.weight, before)


def test_dpo_accepts_one_effective_batch(tmp_path: Path, monkeypatch) -> None:
    config = RLHFConfig(batch_size=4, gradient_accumulation_steps=8)
    trainer = WhisperRLHFTrainer(config, store=object(), paths=PathsConfig())
    rows = [{"row": index} for index in range(34)]
    monkeypatch.setattr(
        "rlhf.trainer.build_dpo_rows",
        lambda *args, **kwargs: (rows, 17, 17),
    )
    monkeypatch.setattr(trainer, "_train_dpo", lambda actual, output: output)

    assert trainer.train(tmp_path / "output") == tmp_path / "output"


def test_dpo_rejects_less_than_one_effective_batch(
    tmp_path: Path, monkeypatch
) -> None:
    config = RLHFConfig(batch_size=4, gradient_accumulation_steps=8)
    trainer = WhisperRLHFTrainer(config, store=object(), paths=PathsConfig())
    rows = [{"row": index} for index in range(31)]
    monkeypatch.setattr(
        "rlhf.trainer.build_dpo_rows",
        lambda *args, **kwargs: (rows, 16, 15),
    )

    with pytest.raises(ValueError, match="Only 31 DPO pairs available"):
        trainer.train(tmp_path / "output")
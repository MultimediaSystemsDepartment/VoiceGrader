from types import SimpleNamespace

import torch

from reward_model.train_reward_model import RewardModelTrainer


class _AudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        assert not self.training
        return SimpleNamespace(last_hidden_state=features.unsqueeze(1))


class _TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        assert not self.training
        values = input_ids.float().repeat(1, 3)
        return SimpleNamespace(pooler_output=values)


def test_reward_encoders_run_once_per_batch_not_per_epoch(monkeypatch, tmp_path):
    config = SimpleNamespace(
        device="cpu",
        torch_dtype="float32",
        learning_rate=1e-3,
        batch_size=2,
        num_epochs=3,
    )
    audio_encoder = _AudioEncoder()
    text_encoder = _TextEncoder()
    module = SimpleNamespace(
        whisper_encoder=audio_encoder,
        bert=text_encoder,
        mlp=torch.nn.Linear(5, 1),
        eval=lambda: None,
    )
    rm = SimpleNamespace(
        module=module,
        _feature_extractor=object(),
        _bert_tokenizer=object(),
    )
    trainer = RewardModelTrainer(config, store=object(), output_dir=tmp_path)

    def encode_batch(batch, *args):
        size = len(batch)
        features = torch.ones(size, 2)
        winner = {
            "input_ids": torch.ones(size, 1, dtype=torch.long),
            "attention_mask": torch.ones(size, 1, dtype=torch.long),
        }
        loser = {
            "input_ids": torch.zeros(size, 1, dtype=torch.long),
            "attention_mask": torch.ones(size, 1, dtype=torch.long),
        }
        return features, winner, loser

    monkeypatch.setattr(trainer, "_encode_batch", encode_batch)
    pairs = [(f"{index}.wav", "winner", "loser") for index in range(4)]

    trainer._run_training(rm, pairs)

    assert audio_encoder.calls == 2
    assert text_encoder.calls == 2
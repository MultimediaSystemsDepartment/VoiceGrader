from types import SimpleNamespace

import torch

from reward_model.audio_text_reward import AudioTextRewardModel, _build_module
from reward_model.config import RewardModelConfig
from run_generate_synthetic import _pool_size, _select_pair, _token_edit_distance


def test_select_pair_uses_reward_extremes() -> None:
    selected = _select_pair(["middle", "best", "worst"], [0.2, 0.9, -0.4], 0.5)

    assert selected == ("best", "worst", 1.3)


def test_select_pair_rejects_weak_gap() -> None:
    assert _select_pair(["a", "b"], [0.2, 0.25], 0.1) is None


def test_select_pair_rejects_too_few_token_edits() -> None:
    assert (
        _select_pair(["jeden dwa trzy", "jeden dwa cztery"], [0.9, 0.1], 0.1, 2) is None
    )


def test_token_edit_distance_and_pool_size() -> None:
    assert _token_edit_distance("a b c", "a x c d") == 2
    assert _pool_size(34, 0.5, 4.0) == 68


def test_candidate_scoring_encodes_shared_audio_once() -> None:
    class FakeAudioEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, features):
            self.calls += 1
            return SimpleNamespace(
                last_hidden_state=torch.zeros((features.shape[0], 2, 1024))
            )

    class FakeTextEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(pooler_output=torch.zeros((input_ids.shape[0], 768)))

    audio_encoder = FakeAudioEncoder()
    module = _build_module(
        RewardModelConfig(device="cpu"), audio_encoder, FakeTextEncoder()
    )

    rewards = module.score_candidates(
        torch.zeros((1, 80, 20)),
        torch.ones((3, 4), dtype=torch.long),
        torch.ones((3, 4), dtype=torch.long),
    )

    assert rewards.shape == (3,)
    assert audio_encoder.calls == 1


def test_candidate_batch_scoring_encodes_each_audio_once() -> None:
    class FakeAudioEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, features):
            self.calls += 1
            values = torch.arange(features.shape[0], dtype=torch.float32)
            return SimpleNamespace(
                last_hidden_state=values[:, None, None].expand(-1, 2, 1024)
            )

    class FakeTextEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(pooler_output=torch.zeros((input_ids.shape[0], 768)))

    audio_encoder = FakeAudioEncoder()
    module = _build_module(
        RewardModelConfig(device="cpu"), audio_encoder, FakeTextEncoder()
    )
    rewards = module.score_candidate_batches(
        torch.zeros((2, 80, 20)),
        torch.ones((5, 4), dtype=torch.long),
        torch.ones((5, 4), dtype=torch.long),
        torch.tensor([0, 0, 1, 1, 1]),
    )

    assert rewards.shape == (5,)
    assert audio_encoder.calls == 1


def test_reward_checkpoint_architecture_config_is_authoritative(tmp_path) -> None:
    stored = RewardModelConfig(mlp_hidden_dim=123, device="cuda")
    stored.to_json(tmp_path / "rm_config.json")

    model = AudioTextRewardModel(
        RewardModelConfig(mlp_hidden_dim=999, device="cpu"),
        checkpoint_path=tmp_path,
    )

    assert model.config.mlp_hidden_dim == 123
    assert model.config.device == "cpu"

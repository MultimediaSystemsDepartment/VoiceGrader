from __future__ import annotations

import torch

from rlhf.trainer import _initial_reference_offset


def test_identical_policy_and_reference_have_zero_initial_offset() -> None:
    chosen = torch.tensor([-10.0, -20.0])
    rejected = torch.tensor([-12.0, -19.0])

    assert _initial_reference_offset(
        chosen, rejected, chosen.clone(), rejected.clone()
    ) == 0.0


def test_initial_offset_detects_reference_likelihood_mismatch() -> None:
    offset = _initial_reference_offset(
        torch.tensor([-10.0]),
        torch.tensor([-12.0]),
        torch.tensor([-15.0]),
        torch.tensor([-11.0]),
    )

    assert offset == 6.0
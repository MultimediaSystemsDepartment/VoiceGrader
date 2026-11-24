from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_file_transcription_defaults_to_pinned_publication_and_retains_stock_override(
    monkeypatch,
):
    calls = []
    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(
        load_model=lambda model, **kwargs: calls.append((model, kwargs)),
    ))
    utilities = ModuleType("lightning.pytorch.utilities")
    utilities.disable_possible_user_warnings = lambda: None
    monkeypatch.setitem(sys.modules, "lightning.pytorch.utilities", utilities)
    path = ROOT / "asr" / "asr_processor" / "whisperx_transcriber.py"
    spec = importlib.util.spec_from_file_location("asr_transcriber_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VOICEGRADER_ACTIVE_MODEL_DIR", str(ROOT / "test-publication"))
    module.WhisperXTranscriber(use_gpu=False)
    module.WhisperXTranscriber(model_name="large-v3", use_gpu=False)
    assert calls[0][0] == str(ROOT / "test-publication")
    assert calls[1][0] == "large-v3"
    assert calls[0][1]["device"] == "cpu"

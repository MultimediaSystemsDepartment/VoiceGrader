from gpu_stress_tester.admedvoice import AudioRecord
from gpu_stress_tester.training import normalize_text, oracle_preference


def test_normalize_text_handles_polish_punctuation() -> None:
    assert normalize_text("  Dzień, dobry! ") == "dzień dobry"


def test_oracle_selects_lower_wer_hypothesis() -> None:
    record = AudioRecord("clip.wav", "pacjent czuje się dobrze")

    result = oracle_preference(record, "pacjent czuje się dobrze", "pacjent źle")

    assert result is not None
    assert result.chosen == "pacjent czuje się dobrze"
    assert result.rejected == "pacjent źle"


def test_oracle_skips_identical_correct_hypotheses() -> None:
    record = AudioRecord("clip.wav", "tekst")

    assert oracle_preference(record, "tekst", "tekst") is None
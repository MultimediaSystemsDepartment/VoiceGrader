"""evaluation package -- WER/CER evaluation and reporting."""

from .evaluator import EvalResult, WhisperEvaluator
from .report import generate_markdown_report, generate_text_report

__all__ = [
    "WhisperEvaluator",
    "EvalResult",
    "generate_text_report",
    "generate_markdown_report",
]

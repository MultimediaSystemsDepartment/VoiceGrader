from __future__ import annotations

import argparse
import http.cookiejar
import json
import random
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from evaluation.model_health import compute_clip_metrics


@dataclass(frozen=True)
class OracleDecision:
    preference: str
    wer_a: float
    wer_b: float
    correction_text: str


def decide_from_ground_truth(
    reference: str,
    hypothesis_a: str,
    hypothesis_b: str,
    tie_threshold: float = 0.05,
) -> OracleDecision:
    reference = reference.strip()
    if not reference:
        raise ValueError("Ground-truth reference is missing")
    wer_a = float(compute_clip_metrics(reference, hypothesis_a)["wer"])
    wer_b = float(compute_clip_metrics(reference, hypothesis_b)["wer"])
    if abs(wer_a - wer_b) <= tie_threshold:
        preference = "equal"
    else:
        preference = "a" if wer_a < wer_b else "b"
    return OracleDecision(preference, wer_a, wer_b, reference)


class SurrogateAnnotator:
    def __init__(
        self,
        base_url: str,
        participant_id: str,
        tie_threshold: float = 0.05,
        missing_gt: str = "error",
        submit_gt_correction: bool = True,
        timeout: float = 30.0,
        error_rate: float = 0.0,
        seed: int = 42,
        rng_offset: int = 0,
    ) -> None:
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")
        self.base_url = base_url.rstrip("/")
        self.participant_id = participant_id
        self.tie_threshold = tie_threshold
        self.missing_gt = missing_gt
        self.submit_gt_correction = submit_gt_correction
        self.timeout = timeout
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        for _ in range(rng_offset):
            self.rng.random()
        cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookies)
        )

    def open_session(self) -> None:
        self._request("/setup", {"participant_id": self.participant_id}, expect_json=False)

    def annotate(self, budget: int, output_jsonl: Path) -> list[dict]:
        self.open_session()
        records = []
        while len(records) < budget:
            self._request("/api/heartbeat", {})
            pair = self._request("/api/pair")
            if pair.get("done"):
                break
            reference = (pair.get("reference") or "").strip()
            if not reference:
                if self.missing_gt == "skip":
                    self._request("/api/skip", {"pair_id": pair["pair_id"]})
                    continue
                raise ValueError(f"Pair {pair['pair_id']} has no ground truth")
            decision = decide_from_ground_truth(
                reference,
                pair.get("hyp_a", ""),
                pair.get("hyp_b", ""),
                self.tie_threshold,
            )
            preference = decision.preference
            flipped = False
            if preference in {"a", "b"} and self.rng.random() < self.error_rate:
                preference = "b" if preference == "a" else "a"
                flipped = True
            response = self._request(
                "/api/submit",
                {
                    "pair_id": pair["pair_id"],
                    "preference": preference,
                    "correction_text": (
                        decision.correction_text if self.submit_gt_correction else ""
                    ),
                    "strength": "10",
                    "derive_preference": "false",
                },
            )
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "participant_id": self.participant_id,
                "pair_id": pair["pair_id"],
                "audio_id": pair.get("audio_id"),
                "model_version": pair.get("model_version"),
                "hyp_a": pair.get("hyp_a", ""),
                "hyp_b": pair.get("hyp_b", ""),
                "reference": reference,
                **asdict(decision),
                "oracle_preference": decision.preference,
                "preference": preference,
                "preference_flipped": flipped,
                "submitted_correction_text": (
                    decision.correction_text if self.submit_gt_correction else ""
                ),
                "response_ok": bool(response.get("ok")),
            }
            self._append_jsonl(output_jsonl, record)
            records.append(record)
        return records

    def _request(
        self,
        path: str,
        form: dict[str, str] | None = None,
        expect_json: bool = True,
    ):
        data = urllib.parse.urlencode(form).encode() if form is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method="POST" if form is not None else "GET",
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            payload = response.read()
        return json.loads(payload) if expect_json else payload

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit deterministic GT annotations")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--participant-id", default="surrogate-gt")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--tie-threshold", type=float, default=0.05)
    parser.add_argument("--missing-gt", choices=("error", "skip"), default="error")
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotator = SurrogateAnnotator(
        args.base_url,
        args.participant_id,
        tie_threshold=args.tie_threshold,
        missing_gt=args.missing_gt,
        submit_gt_correction=not args.no_correction,
        error_rate=args.error_rate,
        seed=args.seed,
    )
    records = annotator.annotate(args.budget, args.output)
    print(f"Submitted {len(records)} annotations to {args.base_url}")


if __name__ == "__main__":
    main()
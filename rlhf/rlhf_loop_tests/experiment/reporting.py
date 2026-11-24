from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from statistics import mean

from evaluation.model_health import paired_bootstrap_delta


def build_experiment_report(manifest_path: Path) -> dict:
    manifest = _read_json(manifest_path)
    config = manifest["config"]
    evaluations = {
        (item["replicate"], item["round"]): _read_json(Path(item["report_file"]))
        for item in manifest.get("evaluation_runs", [])
    }
    expected = {
        (replicate, round_number)
        for replicate in range(1, len(config["seeds"]) + 1)
        for round_number in range(1, config["rounds"] + 1)
    }
    missing = sorted(expected - evaluations.keys())
    if missing:
        raise ValueError(f"Holdout evaluations are incomplete: {missing}")

    replicate_reports = []
    arm_names = ("sft", "rlhf", "rlhf_noisy")
    final_wers = {arm: [] for arm in arm_names}
    for replicate, seed in enumerate(config["seeds"], 1):
        rounds = [evaluations[(replicate, number)] for number in range(1, config["rounds"] + 1)]
        arms = {}
        for arm in arm_names:
            initial = rounds[0]["arms"][arm]["initial_metrics"]
            points = [{"annotations": 0, "wer": initial["wer"], "cer": initial["cer"]}]
            for round_number, result in enumerate(rounds, 1):
                metrics = result["arms"][arm]["active_metrics"]
                points.append(
                    {
                        "annotations": round_number * config["annotations_per_round"],
                        "wer": metrics["wer"],
                        "cer": metrics["cer"],
                    }
                )
            arms[arm] = {
                "points": points,
                "final_wer": points[-1]["wer"],
                "final_cer": points[-1]["cer"],
                "wer_improvement_from_initial": points[-1]["wer"] - points[0]["wer"],
                "normalized_wer_annotation_auc": _normalized_auc(points),
            }
        for arm in arm_names:
            final_wers[arm].append(arms[arm]["final_wer"])
        replicate_reports.append(
            {
                "replicate": replicate,
                "seed": seed,
                "arms": arms,
                "final_rlhf_minus_sft_wer": (
                    arms["rlhf"]["final_wer"] - arms["sft"]["final_wer"]
                ),
                "final_rlhf_noisy_minus_sft_wer": (
                    arms["rlhf_noisy"]["final_wer"] - arms["sft"]["final_wer"]
                ),
                "final_rlhf_noisy_minus_rlhf_wer": (
                    arms["rlhf_noisy"]["final_wer"] - arms["rlhf"]["final_wer"]
                ),
                "round_comparisons": [result["rlhf_minus_sft"] for result in rounds],
                "round_noisy_comparisons": [
                    result["rlhf_noisy_minus_sft"] for result in rounds
                ],
            }
        )

    across_seeds = {
        "rlhf_minus_sft": paired_bootstrap_delta(
            final_wers["sft"], final_wers["rlhf"], seed=config["seeds"][0]
        ),
        "rlhf_noisy_minus_sft": paired_bootstrap_delta(
            final_wers["sft"], final_wers["rlhf_noisy"], seed=config["seeds"][0]
        ),
        "rlhf_noisy_minus_rlhf": paired_bootstrap_delta(
            final_wers["rlhf"], final_wers["rlhf_noisy"], seed=config["seeds"][0]
        ),
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "replicate_count": len(config["seeds"]),
        "rounds": config["rounds"],
        "annotations_per_round": config["annotations_per_round"],
        "noisy_feedback_error_rate": config.get("noisy_feedback_error_rate", 0.05),
        "reportable_seed_count": len(config["seeds"]) >= 5,
        "replicates": replicate_reports,
        "across_seed_final_rlhf_minus_sft_wer": across_seeds["rlhf_minus_sft"],
        "across_seed_final_rlhf_noisy_minus_sft_wer": across_seeds[
            "rlhf_noisy_minus_sft"
        ],
        "across_seed_final_rlhf_noisy_minus_rlhf_wer": across_seeds[
            "rlhf_noisy_minus_rlhf"
        ],
    }
    json_path = manifest_path.parent / "report.json"
    markdown_path = manifest_path.parent / "report.md"
    csv_path = manifest_path.parent / "convergence.csv"
    svg_path = manifest_path.parent / "convergence.svg"
    _write_json_atomic(json_path, report)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    _write_convergence_csv(csv_path, report)
    _write_convergence_svg(svg_path, report)
    manifest["report"] = {
        "json_file": str(json_path),
        "markdown_file": str(markdown_path),
        "csv_file": str(csv_path),
        "svg_file": str(svg_path),
    }
    manifest["status"] = "complete"
    _write_json_atomic(manifest_path, manifest)
    return report


def _normalized_auc(points: list[dict]) -> float:
    maximum_annotations = points[-1]["annotations"]
    if maximum_annotations <= 0:
        raise ValueError("AUC requires at least one annotation round")
    area = sum(
        (right["annotations"] - left["annotations"])
        * (left["wer"] + right["wer"])
        / 2
        for left, right in zip(points, points[1:])
    )
    return area / maximum_annotations


def _markdown_report(report: dict) -> str:
    comparison = report["across_seed_final_rlhf_minus_sft_wer"]
    noisy_comparison = report["across_seed_final_rlhf_noisy_minus_sft_wer"]
    noise_cost = report["across_seed_final_rlhf_noisy_minus_rlhf_wer"]
    lines = [
        f"# {report['experiment_id']} convergence report",
        "",
        f"Replicates: {report['replicate_count']}  ",
        f"Rounds: {report['rounds']}  ",
        f"Annotations per round: {report['annotations_per_round']}",
        "",
        "## Final paired result",
        "",
        (
            "RLHF minus SFT corpus WER across seeds: "
            f"{comparison['delta']:.4f} "
            f"(95% CI {comparison['ci_low']:.4f}, {comparison['ci_high']:.4f})."
        ),
        (
            "Noisy RLHF minus SFT corpus WER across seeds: "
            f"{noisy_comparison['delta']:.4f} "
            f"(95% CI {noisy_comparison['ci_low']:.4f}, "
            f"{noisy_comparison['ci_high']:.4f})."
        ),
        (
            "Noisy minus clean RLHF corpus WER across seeds: "
            f"{noise_cost['delta']:.4f} "
            f"(95% CI {noise_cost['ci_low']:.4f}, {noise_cost['ci_high']:.4f})."
        ),
        "",
        "| Replicate | Seed | SFT WER | RLHF WER | Noisy RLHF WER | Clean delta | Noisy delta | SFT AUC | RLHF AUC | Noisy AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["replicates"]:
        sft = item["arms"]["sft"]
        rlhf = item["arms"]["rlhf"]
        noisy = item["arms"]["rlhf_noisy"]
        lines.append(
            f"| {item['replicate']} | {item['seed']} | {sft['final_wer']:.4f} | "
            f"{rlhf['final_wer']:.4f} | {noisy['final_wer']:.4f} | "
            f"{item['final_rlhf_minus_sft_wer']:.4f} | "
            f"{item['final_rlhf_noisy_minus_sft_wer']:.4f} | "
            f"{sft['normalized_wer_annotation_auc']:.4f} | "
            f"{rlhf['normalized_wer_annotation_auc']:.4f} | "
            f"{noisy['normalized_wer_annotation_auc']:.4f} |"
        )
    lines.append("")
    if not report["reportable_seed_count"]:
        lines.append("This run has fewer than five seeds and is not reportable evidence.")
        lines.append("")
    return "\n".join(lines)


def _write_convergence_csv(path: Path, report: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("replicate", "seed", "arm", "annotations", "wer", "cer"),
        )
        writer.writeheader()
        for replicate in report["replicates"]:
            for arm, result in replicate["arms"].items():
                for point in result["points"]:
                    writer.writerow(
                        {
                            "replicate": replicate["replicate"],
                            "seed": replicate["seed"],
                            "arm": arm,
                            **point,
                        }
                    )


def _write_convergence_svg(path: Path, report: dict) -> None:
    width, height = 1000, 580
    left, right, top, bottom = 85, 30, 65, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    noise_percent = report["noisy_feedback_error_rate"] * 100
    arm_labels = {
        "sft": ("SFT", "#356aa0"),
        "rlhf": ("Production RLHF", "#087f7a"),
        "rlhf_noisy": (f"RLHF with {noise_percent:g}% feedback errors", "#d6533c"),
    }
    aggregate = {}
    for arm in arm_labels:
        point_sets = [item["arms"][arm]["points"] for item in report["replicates"]]
        aggregate[arm] = [
            {
                "annotations": point["annotations"],
                "wer": mean(values[index]["wer"] for values in point_sets),
            }
            for index, point in enumerate(point_sets[0])
        ]
    all_points = [point for values in aggregate.values() for point in values]
    maximum_x = max(point["annotations"] for point in all_points) or 1
    minimum_y = min(point["wer"] for point in all_points)
    maximum_y = max(point["wer"] for point in all_points)
    padding = max((maximum_y - minimum_y) * 0.12, 0.01)
    minimum_y = max(0.0, minimum_y - padding)
    maximum_y += padding

    def x(value: float) -> float:
        return left + value / maximum_x * plot_width

    def y(value: float) -> float:
        return top + (maximum_y - value) / (maximum_y - minimum_y) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        '<text x="85" y="35" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#182225">Mean holdout WER convergence</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#667176"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#667176"/>',
    ]
    for index in range(6):
        value = minimum_y + (maximum_y - minimum_y) * index / 5
        position = y(value)
        elements.extend(
            (
                f'<line x1="{left}" y1="{position:.1f}" x2="{left + plot_width}" y2="{position:.1f}" stroke="#e1e5e2"/>',
                f'<text x="{left - 10}" y="{position + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#667176">{value * 100:.1f}%</text>',
            )
        )
    for arm_index, (arm, (label, color)) in enumerate(arm_labels.items()):
        points = aggregate[arm]
        coordinates = " ".join(
            f"{x(point['annotations']):.1f},{y(point['wer']):.1f}" for point in points
        )
        elements.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for point in points:
            elements.append(
                f'<circle cx="{x(point["annotations"]):.1f}" cy="{y(point["wer"]):.1f}" r="4" fill="{color}"/>'
            )
        legend_x = left + arm_index * 260
        elements.extend(
            (
                f'<line x1="{legend_x}" y1="{height - 25}" x2="{legend_x + 28}" y2="{height - 25}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 36}" y="{height - 20}" font-family="Arial, sans-serif" font-size="13" fill="#182225">{escape(label)}</text>',
            )
        )
    elements.extend(
        (
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 48}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#667176">Accepted feedback annotations</text>',
            f'<text x="18" y="{top + plot_height / 2:.1f}" transform="rotate(-90 18 {top + plot_height / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#667176">Corpus WER</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)

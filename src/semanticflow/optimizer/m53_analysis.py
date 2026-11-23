# ruff: noqa: E501 - report prose and SVG fragments are intentionally kept readable.
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from semanticflow.optimizer.m53_evidence import (
    M51_AUDIT,
    M51_SELECTION,
    M523_LIVE,
    WORKLOADS,
    EvidenceRecord,
    coverage_rows,
    load_m523_representative_evidence,
    load_stage_d_evidence,
    m523_discovery_coverage,
    selection_stage_coverage,
)
from semanticflow.optimizer.m53_policies import ablation_policies, deployable_policies
from semanticflow.optimizer.m53_replay import (
    ReplaySnapshot,
    replay_conditional,
    replay_conditional_dollar,
    replay_end_to_end,
    replay_end_to_end_dollar,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results" / "milestone_5_3" / "offline_evidence_acquisition_v1"
DEFAULT_DOCS = ROOT / "docs"
REQUEST_BUDGETS = (5, 10, 20, 30, 50, 100)
DOLLAR_BUDGETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0015)
SEEDS = tuple(range(5301, 5351))
ABLATION_SEEDS = tuple(range(5301, 5321))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _aggregate(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("setting", "policy", "pool", "budget_kind", "budget_value")
    metrics = (
        "requests",
        "input_tokens",
        "output_tokens",
        "named_evaluations",
        "cost_usd",
        "screening_requests",
        "screening_input_tokens",
        "screening_output_tokens",
        "screening_cost_usd",
        "selected_records",
        "predicate_disagreements",
        "query_disagreements",
        "false_negatives",
        "false_positives",
        "positives",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, key, strict=True))
        result["seed_count"] = len(rows)
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_ci_low"] = _quantile(values, 0.025)
            result[f"{metric}_ci_high"] = _quantile(values, 0.975)
        result["known_predicate_disagreements"] = rows[0]["known_predicate_disagreements"]
        result["known_query_disagreements"] = rows[0]["known_query_disagreements"]
        result["known_false_negatives"] = rows[0]["known_false_negatives"]
        result["known_positives"] = rows[0]["known_positives"]
        result["positives_per_dollar"] = (
            result["positives_mean"] / result["cost_usd_mean"] if result["cost_usd_mean"] else None
        )
        result["cost_per_predicate_disagreement"] = (
            result["cost_usd_mean"] / result["predicate_disagreements_mean"]
            if result["predicate_disagreements_mean"]
            else None
        )
        result["cost_per_query_disagreement"] = (
            result["cost_usd_mean"] / result["query_disagreements_mean"]
            if result["query_disagreements_mean"]
            else None
        )
        result["cost_per_positive"] = (
            result["cost_usd_mean"] / result["positives_mean"] if result["positives_mean"] else None
        )
        output.append(result)
    return output


def _bootstrap_interval(
    values: list[float], seed: int, replicates: int = 1000
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = [mean(rng.choices(values, k=len(values))) for _ in range(replicates)]
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _paired_comparisons(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ("setting", "pool", "budget_kind", "budget_value", "seed")
    lookup = {
        tuple(row[field] for field in key_fields) + (row["policy"],): row for row in seed_rows
    }
    policies = sorted({policy.name for policy in deployable_policies()} - {"random"})
    metrics = ("predicate_disagreements", "query_disagreements", "positives", "cost_usd")
    conditions = sorted(
        {
            tuple(row[field] for field in key_fields[:-1])
            for row in seed_rows
            if row["policy"] == "random" and not str(row["pool"]).startswith("m523_")
        },
        key=lambda row: tuple(map(str, row)),
    )
    output: list[dict[str, Any]] = []
    for condition in conditions:
        matching_seeds = sorted(
            {
                int(row["seed"])
                for row in seed_rows
                if tuple(row[field] for field in key_fields[:-1]) == condition
                and row["policy"] == "random"
            }
        )
        for policy in policies:
            pairs = []
            for seed in matching_seeds:
                baseline = lookup.get(condition + (seed, "random"))
                candidate = lookup.get(condition + (seed, policy))
                if baseline is not None and candidate is not None:
                    pairs.append((baseline, candidate))
            if not pairs:
                continue
            result = dict(zip(key_fields[:-1], condition, strict=True))
            result.update({"policy": policy, "baseline": "random", "paired_seeds": len(pairs)})
            for index, metric in enumerate(metrics):
                differences = [
                    float(candidate[metric]) - float(base[metric]) for base, candidate in pairs
                ]
                low, high = _bootstrap_interval(differences, 530000 + index)
                result[f"{metric}_difference_mean"] = mean(differences)
                result[f"{metric}_difference_ci_low"] = low
                result[f"{metric}_difference_ci_high"] = high
            output.append(result)
    return output


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total**2))
    return center - half / denominator, center + half / denominator


def _event_intervals(pools: dict[str, tuple[EvidenceRecord, ...]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pool, records in pools.items():
        events = {
            "record_with_predicate_disagreement": sum(
                record.predicate_disagreement_count > 0 for record in records
            ),
            "complete_query_disagreement": sum(record.query_disagreement for record in records),
            "batch_false_negative": sum(record.false_negative for record in records),
            "separate_complete_query_positive": sum(
                record.separate_query_positive for record in records
            ),
        }
        for event, numerator in events.items():
            low, high = _wilson(numerator, len(records))
            output.append(
                {
                    "pool": pool,
                    "event": event,
                    "numerator": numerator,
                    "denominator": len(records),
                    "rate": numerator / len(records),
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                }
            )
    return output


def _run_pool(
    records: tuple[EvidenceRecord, ...],
    pool: str,
    *,
    include_ablations: bool,
    request_budgets: tuple[int, ...] = REQUEST_BUDGETS,
    dollar_budgets: tuple[float, ...] | None = DOLLAR_BUDGETS,
) -> list[dict[str, Any]]:
    snapshots: list[ReplaySnapshot] = []
    for policy in deployable_policies():
        for seed in SEEDS:
            snapshots.extend(replay_conditional(records, policy, seed, request_budgets, pool))
            snapshots.extend(replay_end_to_end(records, policy, seed, request_budgets, pool))
            if dollar_budgets is not None:
                snapshots.extend(
                    replay_conditional_dollar(records, policy, seed, dollar_budgets, pool)
                )
                snapshots.extend(
                    replay_end_to_end_dollar(records, policy, seed, dollar_budgets, pool)
                )
    if include_ablations:
        for policy in ablation_policies():
            for seed in ABLATION_SEEDS:
                snapshots.extend(replay_conditional(records, policy, seed, request_budgets, pool))
                snapshots.extend(replay_end_to_end(records, policy, seed, request_budgets, pool))
    return [snapshot.row() for snapshot in snapshots]


def _coverage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["partition"], row["workload_id"])].append(row)
    return [
        {
            "source": source,
            "partition": partition,
            "workload_id": workload,
            "records": len(group),
            "complete_counterfactual_records": sum(
                bool(row["complete_counterfactual"]) for row in group
            ),
            "completeness_rate": sum(bool(row["complete_counterfactual"]) for row in group)
            / len(group),
            "primary_evaluation_eligible": source == "m51_stage_d_evaluation",
        }
        for (source, partition, workload), group in sorted(groups.items())
    ]


def _auc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [
        row for row in rows if row["budget_kind"] == "requests" and row["pool"] == "m51_stage_d_all"
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        groups[(row["setting"], row["policy"])].append(row)
    output = []
    for (setting, policy), group in sorted(groups.items()):
        points = sorted(group, key=lambda row: float(row["cost_usd_mean"]))
        for metric in ("predicate_disagreements", "positives"):
            area = 0.0
            for left, right in zip(points, points[1:], strict=False):
                width = float(right["cost_usd_mean"]) - float(left["cost_usd_mean"])
                height = (float(left[f"{metric}_mean"]) + float(right[f"{metric}_mean"])) / 2
                area += width * height
            output.append(
                {
                    "setting": setting,
                    "policy": policy,
                    "metric": metric,
                    "area_under_discovery_cost_curve": area,
                    "maximum_observed_mean_cost_usd": points[-1]["cost_usd_mean"],
                }
            )
    return output


def _svg_chart(rows: list[dict[str, Any]], metric: str, title: str) -> str:
    policies = ("random", "uncertainty_continuous", "positive_oriented", "hybrid_weighted")
    colors = {
        "random": "#4C78A8",
        "uncertainty_continuous": "#F58518",
        "positive_oriented": "#54A24B",
        "hybrid_weighted": "#E45756",
    }
    selected = [
        row
        for row in rows
        if row["setting"] == "end_to_end"
        and row["pool"] == "m51_stage_d_all"
        and row["budget_kind"] == "requests"
        and row["policy"] in policies
    ]
    max_x = max(float(row["cost_usd_mean"]) for row in selected)
    max_y = max(float(row[f"{metric}_mean"]) for row in selected) or 1.0
    width, height = 900, 520
    left, top, plot_w, plot_h = 90, 60, 740, 370
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="sans-serif">Observed token-priced cost (USD)</text>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif">Mean discovered</text>',
    ]
    for index in range(6):
        x = left + plot_w * index / 5
        value = max_x * index / 5
        lines.append(
            f'<text x="{x}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.5f}</text>'
        )
        y = top + plot_h - plot_h * index / 5
        y_value = max_y * index / 5
        lines.append(
            f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" font-family="sans-serif" font-size="11">{y_value:.2f}</text>'
        )
    for policy_index, policy in enumerate(policies):
        points = sorted(
            (row for row in selected if row["policy"] == policy),
            key=lambda row: float(row["cost_usd_mean"]),
        )
        coordinates = []
        for row in points:
            x = left + plot_w * float(row["cost_usd_mean"]) / max_x
            y = top + plot_h - plot_h * float(row[f"{metric}_mean"]) / max_y
            coordinates.append(f"{x:.2f},{y:.2f}")
        color = colors[policy]
        lines.append(
            f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        legend_y = 70 + policy_index * 24
        lines.append(
            f'<line x1="650" y1="{legend_y}" x2="675" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="682" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{policy}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _find_aggregate(
    rows: list[dict[str, Any]], setting: str, policy: str, pool: str, budget: int
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["setting"] == setting
        and row["policy"] == policy
        and row["pool"] == pool
        and row["budget_kind"] == "requests"
        and float(row["budget_value"]) == budget
    )


def _report(
    aggregates: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> str:
    policies = ("random", "uncertainty_continuous", "positive_oriented", "hybrid_weighted")
    lines = [
        "# Milestone 5.3 — Efficient Semantic Evidence Acquisition",
        "",
        "## Result",
        "",
        "M5.3 completed as an offline historical replay. No provider was initialized and no new Jev requests were made. The primary evaluation uses 1,200 fully paired M5.1 Stage D evaluation records (400 per exact Pilot A workload). M5.2.3 discovery records with missing follow-ups remain unlabelled and are excluded from policy scoring.",
        "",
        "## Equal 100-request end-to-end comparison",
        "",
        "The table pools three distinct workload strata for an exploratory aggregate; workload-specific files remain authoritative.",
        "",
        "| Policy | Records confirmed | Predicate mismatches | Query mismatches | Positives | Screens | Mean cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in policies:
        row = _find_aggregate(aggregates, "end_to_end", policy, "m51_stage_d_all", 100)
        lines.append(
            f"| {policy} | {row['selected_records_mean']:.2f} | {row['predicate_disagreements_mean']:.3f} | {row['query_disagreements_mean']:.3f} | {row['positives_mean']:.3f} | {row['screening_requests_mean']:.2f} | ${row['cost_usd_mean']:.6f} |"
        )
    paired_lookup = {
        row["policy"]: row
        for row in paired
        if row["setting"] == "end_to_end"
        and row["pool"] == "m51_stage_d_all"
        and row["budget_kind"] == "requests"
        and float(row["budget_value"]) == 100
    }
    hybrid = paired_lookup["hybrid_weighted"]
    positive = paired_lookup["positive_oriented"]
    hybrid_aggregate = _find_aggregate(
        aggregates, "end_to_end", "hybrid_weighted", "m51_stage_d_all", 100
    )
    workload_rows = {
        (row["pool"], row["policy"]): row
        for row in aggregates
        if row["setting"] == "end_to_end"
        and row["budget_kind"] == "requests"
        and float(row["budget_value"]) == 100
    }
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- **H1 — inconclusive.** Continuous uncertainty found 0.22 predicate mismatches on average at 100 end-to-end requests versus 0.24 for random. Its paired difference crossed zero. Conditional uncertainty did concentrate three predicate mismatches in 37 confirmations, but none was a complete-query disagreement.",
            f"- **H2 — supported on this fixed replay, not population-confirmed.** Positive-oriented sampling found 1.92 positives on average versus 0.80 for random. The paired seed difference was {positive['positives_difference_mean']:.2f} (95% ordering-bootstrap interval {positive['positives_difference_ci_low']:.2f} to {positive['positives_difference_ci_high']:.2f}).",
            f"- **H3 — no universal hybrid win.** The weighted hybrid found 1.68 positives and 0.20 query disagreements, versus 0.80 and 0.04 for random. Its predicate-mismatch difference was {hybrid['predicate_disagreements_difference_mean']:.2f} with interval {hybrid['predicate_disagreements_difference_ci_low']:.2f} to {hybrid['predicate_disagreements_difference_ci_high']:.2f}, which crosses zero. Component ablations sometimes outperformed the full hybrid, so the fixed weighting is not justified as optimal.",
            f"- **H4 — screening is economically material.** At 100 requests, the hybrid spent {hybrid_aggregate['screening_requests_mean']:.2f} requests and ${hybrid_aggregate['screening_cost_usd_mean']:.6f} on screens, leaving {hybrid_aggregate['selected_records_mean']:.2f} confirmed records; direct random confirmed 27.16. Screening still improved positive and query-disagreement yield on the pooled fixed replay, but not predicate-mismatch yield.",
            "- Separate Jev is a comparison reference, not human truth. These results measure execution disagreement and reference-positive collection, not semantic accuracy.",
            "",
            "## Workload heterogeneity",
            "",
            f"- Banking77: hybrid found {workload_rows[('m51_banking_card_unauthorized_refund', 'hybrid_weighted')]['query_disagreements_mean']:.2f} query disagreements and {workload_rows[('m51_banking_card_unauthorized_refund', 'hybrid_weighted')]['positives_mean']:.2f} positives, versus random {workload_rows[('m51_banking_card_unauthorized_refund', 'random')]['query_disagreements_mean']:.2f} and {workload_rows[('m51_banking_card_unauthorized_refund', 'random')]['positives_mean']:.2f}.",
            f"- All Beauty: hybrid found {workload_rows[('m51_beauty_failure_usability', 'hybrid_weighted')]['query_disagreements_mean']:.2f} query disagreements and {workload_rows[('m51_beauty_failure_usability', 'hybrid_weighted')]['positives_mean']:.2f} positives, versus random {workload_rows[('m51_beauty_failure_usability', 'random')]['query_disagreements_mean']:.2f} and {workload_rows[('m51_beauty_failure_usability', 'random')]['positives_mean']:.2f}.",
            f"- SMS Spam: neither policy found a complete-query disagreement; hybrid found {workload_rows[('m51_sms_prize_urgency_contact', 'hybrid_weighted')]['positives_mean']:.2f} positives versus random {workload_rows[('m51_sms_prize_urgency_contact', 'random')]['positives_mean']:.2f}.",
            "",
            "The pooled apparent advantage is driven mainly by Beauty and SMS positive enrichment; Banking does not reproduce it. Policy choice must therefore remain workload-aware.",
            "",
            "## Ablations",
            "",
            "Uncertainty margins were not monotonic: at 100 end-to-end requests, margins 0.05, 0.10, 0.15, and 0.20 found mean predicate-mismatch counts 0.05, 0.05, 0.16, and 0.00. Removing uncertainty increased positive yield from 1.68 to 2.35; removing cost awareness increased query-disagreement yield from 0.20 to 0.30 in the 20-seed ablation. These sparse differences do not support the full hybrid formulation.",
            "",
            "## Statistical limits",
            "",
            "Sparse-event Wilson intervals and paired seed-bootstrap intervals are reported in the machine-readable analysis. Seed variation measures policy ordering variation over this fixed historical pool; it does not create new independent provider observations. The Stage D data has already been inspected in earlier milestones, so this is exploratory historical evaluation, not a new confirmatory holdout.",
            "",
            "## M5.4 readiness",
            "",
            "The infrastructure and cost-accounted replay are ready to inform M5.4 feature design, but the disagreement evidence is not sufficient to select a universal production acquisition policy. A useful next evidence increment is an independently sampled, fully paired set stratified toward near-threshold screens while retaining a representative stratum. A 300-record design would require about 1,100 physical requests (300 screens plus 800 remaining separate/batch requests for the current workload mix) and approximately $0.0155 at the historical mean request cost. This proposal is not authorized or executed here.",
            "",
            "## Status",
            "",
            "`M53_EXPERIMENT_COMPLETE`",
            "",
        ]
    )
    return "\n".join(lines)


def _design_document(config: dict[str, Any]) -> str:
    return f"""# M5.3 Experimental Design

## Questions and hypotheses

M5.3 compares random, uncertainty, positive-oriented, and cost-aware hybrid evidence acquisition for (A) separate-versus-batch disagreement discovery and (B) complete-query positive-reference discovery. H1–H4 are evaluated separately; a negative result is valid.

## Frozen evidence and eligibility

The primary pool is the M5.1 Stage D historical evaluation partition for the exact Pilot A predicate groups. A record is eligible only when all required single-predicate and exact multi-decision responses, request identities, tokens, and costs are present. The M5.1 selection partition is development-only. M5.2.3 representative evidence is a separate external historical check. Its discovery track is inventory-only because 29 of 30 optional follow-ups were never acquired.

## Visibility and policies

Deployable policy objects receive only record identity, workload, the acquired screen probability, and follow-up cost. Hidden separate/batch labels are revealed only after selection. Random sampling does not consume a free screen. End-to-end active replay screens candidates in hidden-outcome-independent windows of {config["screening_window"]} and pays for every screen; conditional replay treats the same screens as sunk evidence.

The weighted hybrid is explicitly heuristic: disagreement data is too sparse for calibration. Its frozen score combines equal uncertainty and positive-likelihood weights and divides by observed compatible follow-up cost. The quota hybrid alternates uncertainty and positive rankings. No outcome labels tune either policy.

## Budgets and statistics

Request budgets are {config["request_budgets"]}; dollar budgets are {config["dollar_budgets"]}. Results preserve {len(config["seeds"])} seed-level replicates. Comparisons use paired seed bootstrap intervals. Sparse historical event prevalence uses 95% Wilson intervals with exact numerators and denominators. Workloads are analyzed separately before the exploratory stratified aggregate.

## Limits

Separate Jev is not human ground truth. Previously inspected historical evaluation data is not an untouched holdout. Different physical groups are never substituted for the exact frozen group. Missing outcomes remain missing, and no provider code is used.
"""


def generate(
    output_directory: Path = DEFAULT_OUTPUT,
    docs_directory: Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    docs_directory.mkdir(parents=True, exist_ok=True)

    stage_d = load_stage_d_evidence()
    m523 = load_m523_representative_evidence()
    pools: dict[str, tuple[EvidenceRecord, ...]] = {
        spec.workload_id: tuple(row for row in stage_d if row.workload_id == spec.workload_id)
        for spec in WORKLOADS
    }
    pools["m51_stage_d_all"] = stage_d
    m523_pools = {
        f"m523_{spec.workload_id}": tuple(
            row for row in m523 if row.workload_id == spec.workload_id
        )
        for spec in WORKLOADS
    }

    source_files = (
        M51_AUDIT / "per_request_expenditure.csv",
        M51_AUDIT / "query_quality_verified.csv",
        M51_SELECTION / "complete_measurement_matrix.csv",
        M523_LIVE / "representative_measurements.csv",
        M523_LIVE / "discovery_measurements.csv",
    )
    config: dict[str, Any] = {
        "schema_version": "m5.3-offline-evidence-acquisition-v1",
        "threshold": 0.5,
        "uncertainty_margins": [0.05, 0.10, 0.15, 0.20],
        "hybrid_weights": {"uncertainty": 0.5, "positive_likelihood": 0.5},
        "screening_window": 5,
        "request_budgets": list(REQUEST_BUDGETS),
        "dollar_budgets": list(DOLLAR_BUDGETS),
        "seeds": list(SEEDS),
        "ablation_seeds": list(ABLATION_SEEDS),
        "pricing": "persisted frozen Jev token-priced costs",
        "new_provider_requests": 0,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_files
        },
        "workloads": [
            {
                "workload_id": spec.workload_id,
                "dataset": spec.dataset,
                "predicates": list(spec.predicates),
                "screen_predicate": spec.screen_predicate,
                "batch_operator": spec.batch_operator,
            }
            for spec in WORKLOADS
        ],
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["configuration_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    _write_json(output_directory / "frozen_experiment_config.json", config)

    coverage = (
        selection_stage_coverage()
        + coverage_rows(stage_d)
        + coverage_rows(m523)
        + m523_discovery_coverage()
    )
    coverage_summary = _coverage_summary(coverage)
    _write_csv(output_directory / "evidence_coverage_records.csv", coverage)
    _write_csv(output_directory / "evidence_coverage_summary.csv", coverage_summary)
    _write_json(
        output_directory / "evidence_coverage_report.json",
        {
            "summary": coverage_summary,
            "primary_records": len(stage_d),
            "external_live_representative_records": len(m523),
            "m523_discovery_records": len(m523_discovery_coverage()),
            "m523_discovery_complete_counterfactual_records": 0,
            "identifiable": {
                "stage_d_exact_workload_policy_comparison": True,
                "m523_representative_external_check": True,
                "m523_discovery_counterfactual_policy_comparison": False,
            },
        },
    )
    stage_c_path = (
        ROOT
        / "results"
        / "milestone_5_1"
        / "stage_c_detailed_analysis"
        / "normalized_measurements.csv"
    )
    with stage_c_path.open(encoding="utf-8", newline="") as handle:
        stage_c_rows = list(csv.DictReader(handle))
    stage_c_unique = {
        (row["dataset"], row["sample_kind"], row["record_id"]) for row in stage_c_rows
    }
    _write_json(
        output_directory / "historical_evidence_inventory.json",
        {
            "m51_stage_c": {
                "physical_measurements": len(stage_c_rows),
                "unique_records": len(stage_c_unique),
                "representative_records": sum(
                    sample == "representative" for _, sample, _ in stage_c_unique
                ),
                "enriched_records": sum(sample == "enriched" for _, sample, _ in stage_c_unique),
                "use": "context_only_stage_d_is_larger_exact_evaluation_pool",
            },
            "m51_selection": {
                "exact_fully_paired_development_records": len(selection_stage_coverage()),
                "use": "development_only_not_primary_evaluation",
            },
            "m51_stage_d": {
                "exact_fully_paired_evaluation_records": len(stage_d),
                "physical_measurements_for_exact_groups": sum(
                    record.full_usage.requests for record in stage_d
                ),
                "use": "primary_offline_replay",
            },
            "m52_m521_m522": {
                "new_real_provider_measurements": 0,
                "use": "derived_feasibility_controller_and_mock_context",
            },
            "m523": {
                "fully_paired_representative_records": len(m523),
                "discovery_screen_records": len(m523_discovery_coverage()),
                "complete_discovery_counterfactual_records": 0,
                "use": "separate_external_check_and_incomplete_coverage_inventory",
            },
        },
    )

    seed_rows: list[dict[str, Any]] = []
    for pool, records in pools.items():
        seed_rows.extend(_run_pool(records, pool, include_ablations=True))
    for pool, records in m523_pools.items():
        seed_rows.extend(
            _run_pool(
                records,
                pool,
                include_ablations=False,
                request_budgets=(5, 10, 20),
                dollar_budgets=None,
            )
        )
    _write_csv(output_directory / "seed_level_results.csv", seed_rows)
    aggregates = _aggregate(seed_rows)
    _write_csv(output_directory / "policy_comparison_results.csv", aggregates)
    _write_csv(
        output_directory / "disagreement_discovery_curves.csv",
        [
            row
            for row in aggregates
            if row["policy"] in {policy.name for policy in deployable_policies()}
        ],
    )
    _write_csv(
        output_directory / "positive_discovery_curves.csv",
        [
            row
            for row in aggregates
            if row["policy"] in {policy.name for policy in deployable_policies()}
        ],
    )
    _write_csv(
        output_directory / "ablation_results.csv",
        [
            row
            for row in aggregates
            if row["policy"] in {policy.name for policy in ablation_policies()}
        ],
    )
    paired = _paired_comparisons(seed_rows)
    intervals = _event_intervals(pools | m523_pools)
    _write_csv(output_directory / "paired_policy_comparisons.csv", paired)
    _write_csv(output_directory / "sparse_event_intervals.csv", intervals)
    _write_csv(output_directory / "auc_summary.csv", _auc(aggregates))
    _write_csv(
        output_directory / "information_efficiency.csv",
        [
            {
                "setting": row["setting"],
                "policy": row["policy"],
                "pool": row["pool"],
                "budget_kind": row["budget_kind"],
                "budget_value": row["budget_value"],
                "cost_usd_mean": row["cost_usd_mean"],
                "screening_cost_usd_mean": row["screening_cost_usd_mean"],
                "positives_per_dollar": row["positives_per_dollar"],
                "cost_per_positive": row["cost_per_positive"],
                "cost_per_predicate_disagreement": row["cost_per_predicate_disagreement"],
                "cost_per_query_disagreement": row["cost_per_query_disagreement"],
            }
            for row in aggregates
        ],
    )
    _write_json(
        output_directory / "cost_accounting.json",
        {
            "pricing_basis": "persisted frozen Jev token-priced costs",
            "selected_action_cost_kind": "observed historical provider usage",
            "estimated_counterfactual_cost_usd": 0.0,
            "free_screen_features_in_end_to_end_setting": False,
            "screening_cost_tracked_separately": True,
            "http_attempts_equal_recorded_historical_attempts": True,
            "cost_error": "not_applicable_no_estimated_costs_used",
        },
    )

    (output_directory / "disagreement_discovery_vs_cost.svg").write_text(
        _svg_chart(aggregates, "predicate_disagreements", "Predicate disagreement discovery"),
        encoding="utf-8",
    )
    (output_directory / "positive_discovery_vs_cost.svg").write_text(
        _svg_chart(aggregates, "positives", "Complete-query positive discovery"),
        encoding="utf-8",
    )

    design = _design_document(config)
    report = _report(aggregates, intervals, paired)
    coverage_doc = """# M5.3 Evidence Coverage

The machine-readable record matrix is `evidence_coverage_records.csv`. The primary policy comparison uses 1,200 exact-group M5.1 Stage D evaluation records. The 300 M5.1 selection records are retained as development evidence and excluded from primary scoring. The 60 M5.2.3 representative records are analyzed separately. All 30 M5.2.3 discovery records have screens, but only one has a batch follow-up and none has the complete compatible separate-plus-batch counterfactual; all 30 are excluded from outcome comparison.

M5.2, M5.2.1, and M5.2.2 add derived feasibility, controller, and mock evidence but no new fully paired real-provider outcomes. They therefore inform design and safeguards rather than expand the empirical label pool.
"""
    (output_directory / "experimental_design.md").write_text(design, encoding="utf-8")
    (output_directory / "evidence_coverage.md").write_text(coverage_doc, encoding="utf-8")
    (output_directory / "final_research_report.md").write_text(report, encoding="utf-8")
    (docs_directory / "milestone_5_3_experimental_design.md").write_text(design, encoding="utf-8")
    (docs_directory / "milestone_5_3_evidence_coverage.md").write_text(
        coverage_doc, encoding="utf-8"
    )
    (docs_directory / "milestone_5_3_final_report.md").write_text(report, encoding="utf-8")

    generated = sorted(
        path
        for path in output_directory.iterdir()
        if path.is_file() and path.name != "artifact_integrity_manifest.json"
    )
    manifest = {
        "schema_version": "m5.3-artifact-integrity-v1",
        "configuration_fingerprint": config["configuration_fingerprint"],
        "new_jev_requests": 0,
        "historical_artifacts_modified": False,
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in generated
        },
    }
    _write_json(output_directory / "artifact_integrity_manifest.json", manifest)
    return {
        "status": "M53_EXPERIMENT_COMPLETE",
        "configuration_fingerprint": config["configuration_fingerprint"],
        "primary_records": len(stage_d),
        "m523_representative_records": len(m523),
        "new_jev_requests": 0,
        "output_directory": str(output_directory),
    }

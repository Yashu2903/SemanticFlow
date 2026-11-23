from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from semanticflow.optimizer.m53_analysis import _aggregate, _event_intervals, _paired_comparisons
from semanticflow.optimizer.m53_evidence import EvidenceRecord, Usage
from semanticflow.optimizer.m53_policies import (
    HybridPolicy,
    PositivePolicy,
    QuotaHybridPolicy,
    RandomPolicy,
    UncertaintyPolicy,
)
from semanticflow.optimizer.m53_replay import (
    replay_conditional,
    replay_conditional_dollar,
    replay_end_to_end,
    replay_end_to_end_dollar,
)
from semanticflow.optimizer.m522_action_universe import AuthorizedActionUniverse
from semanticflow.optimizer.m522_session import AppendOnlyEventLog
from semanticflow.optimizer.m531_live import (
    _event_state,
    _write_artifact_manifest,
    _write_csv,
    _write_json,
)
from semanticflow.optimizer.m531_protocol import (
    DOLLAR_BUDGETS,
    REPLAY_SEEDS,
    REQUEST_BUDGETS,
    build_experiment,
)


def load_fresh_evidence(session: Path) -> tuple[tuple[EvidenceRecord, ...], dict[str, Any]]:
    config, records, expected_universe, plan = build_experiment()
    persisted = json.loads((session / "frozen_experiment_config.json").read_text("utf-8"))
    if persisted["configuration_fingerprint"] != config["configuration_fingerprint"]:
        raise ValueError("session does not match the frozen M5.3.1 configuration")
    universe = AuthorizedActionUniverse.model_validate_json(
        (session / "authorized_action_universe.json").read_text("utf-8")
    )
    if universe.universe_fingerprint != expected_universe.universe_fingerprint:
        raise ValueError("session action universe does not match source-derived scope")
    state = _event_state(AppendOnlyEventLog(session / "execution_events.jsonl").read())
    plan_index = {row["action_id"]: row for row in plan}
    actions_by_record: dict[str, list[Any]] = defaultdict(list)
    for action in universe.actions:
        actions_by_record[action.record_id].append(action)
    output: list[EvidenceRecord] = []
    partial: list[dict[str, Any]] = []
    for record in records:
        actions = actions_by_record[record.record_id]
        missing = [
            action.action_id for action in actions if action.action_id not in state["receipts"]
        ]
        if missing:
            partial.append(
                {
                    "record_id": record.record_id,
                    "workload_id": record.workload_id,
                    "stratum": record.stratum,
                    "completed_actions": len(actions) - len(missing),
                    "required_actions": len(actions),
                    "missing_action_ids": missing,
                }
            )
            continue
        singles: dict[str, dict[str, Any]] = {}
        batch: dict[str, Any] | None = None
        request_ids: list[str] = []
        for action in actions:
            receipt = state["receipts"][action.action_id]
            request_ids.append(action.action_id)
            if plan_index[action.action_id]["role"] == "complete_multi_predicate":
                batch = receipt
            else:
                singles[action.predicate_ids[0]] = receipt
        if batch is None or len(singles) != len(batch["expected_question_ids"]):
            raise ValueError(f"incompatible paired evidence for {record.record_id}")

        def usage(receipt: dict[str, Any]) -> Usage:
            return Usage(
                requests=1,
                attempts=int(receipt["http_attempts"]),
                input_tokens=int(receipt["input_tokens"]),
                output_tokens=int(receipt["output_tokens"]),
                named_evaluations=len(receipt["expected_question_ids"]),
                cost_usd=float(receipt["token_priced_cost_usd"]),
            )

        predicates = tuple(map(str, batch["expected_question_ids"]))
        screen_predicate = predicates[0]
        screen_usage = usage(singles[screen_predicate])
        full_usage = usage(batch)
        followup_usage = usage(batch)
        for predicate in predicates:
            full_usage += usage(singles[predicate])
            if predicate != screen_predicate:
                followup_usage += usage(singles[predicate])
        output.append(
            EvidenceRecord(
                source="m531_fresh_jev",
                partition=record.stratum,
                workload_id=record.workload_id,
                dataset=record.dataset,
                record_id=record.record_id,
                predicate_ids=predicates,
                batch_operator_id="m531_exact_multi_decision",
                screen_predicate_id=screen_predicate,
                separate_probabilities={
                    predicate: float(singles[predicate]["probabilities"][predicate])
                    for predicate in predicates
                },
                batch_probabilities={
                    predicate: float(batch["probabilities"][predicate]) for predicate in predicates
                },
                separate_decisions={
                    predicate: bool(singles[predicate]["decisions"][predicate])
                    for predicate in predicates
                },
                batch_decisions={
                    predicate: bool(batch["decisions"][predicate]) for predicate in predicates
                },
                request_ids=tuple(request_ids),
                screen_usage=screen_usage,
                followup_usage=followup_usage,
                full_usage=full_usage,
            )
        )
    coverage = {
        "selected_records": len(records),
        "fully_paired_records": len(output),
        "partial_records": len(partial),
        "partial_record_details": partial,
        "completed_requests": len(state["completed"]),
        "http_attempts": state["attempts"],
        "token_priced_cost_usd": state["cost_usd"],
        "session_status": state["status"],
    }
    return tuple(output), coverage


def _policies() -> tuple[Any, ...]:
    return (
        RandomPolicy(),
        UncertaintyPolicy(),
        PositivePolicy(),
        HybridPolicy(),
        QuotaHybridPolicy(),
    )


def _run_pool(records: tuple[EvidenceRecord, ...], pool: str) -> list[dict[str, Any]]:
    rows = []
    for policy in _policies():
        for seed in REPLAY_SEEDS:
            rows.extend(
                snapshot.row()
                for snapshot in replay_conditional(records, policy, seed, REQUEST_BUDGETS, pool)
            )
            rows.extend(
                snapshot.row()
                for snapshot in replay_end_to_end(records, policy, seed, REQUEST_BUDGETS, pool)
            )
            rows.extend(
                snapshot.row()
                for snapshot in replay_conditional_dollar(
                    records, policy, seed, DOLLAR_BUDGETS, pool
                )
            )
            rows.extend(
                snapshot.row()
                for snapshot in replay_end_to_end_dollar(
                    records, policy, seed, DOLLAR_BUDGETS, pool
                )
            )
    return rows


def _review_rows(records: tuple[EvidenceRecord, ...], session: Path) -> list[dict[str, Any]]:
    _, source_records, universe, plan = build_experiment()
    text_index = {record.record_id: record.text for record in source_records}
    action_index = {action.action_id: action for action in universe.actions}
    state = _event_state(AppendOnlyEventLog(session / "execution_events.jsonl").read())
    plan_index = {row["action_id"]: row for row in plan}
    output = []
    for record in records:
        if not (record.predicate_disagreement_count or record.query_disagreement):
            continue
        record_actions = [
            action for action in universe.actions if action.record_id == record.record_id
        ]
        response_metadata = {
            action.action_id: {
                "role": plan_index[action.action_id]["role"],
                "response_identity": state["receipts"][action.action_id]["response_identity"],
                "resolved_model": state["receipts"][action.action_id]["resolved_model"],
                "input_tokens": state["receipts"][action.action_id]["input_tokens"],
                "output_tokens": state["receipts"][action.action_id]["output_tokens"],
                "http_attempts": state["receipts"][action.action_id]["http_attempts"],
            }
            for action in record_actions
        }
        output.append(
            {
                "workload_id": record.workload_id,
                "record_id": record.record_id,
                "stratum": record.partition,
                "original_input_text": text_index[record.record_id],
                "predicate_definitions_json": json.dumps(
                    {
                        predicate: action_index[
                            next(
                                action_id
                                for action_id in record.request_ids
                                if predicate in action_index[action_id].predicate_ids
                                and len(action_index[action_id].predicate_ids) == 1
                            )
                        ].predicate_instructions[0]
                        for predicate in record.predicate_ids
                    },
                    sort_keys=True,
                ),
                "screen_output_json": json.dumps(
                    {record.screen_predicate_id: record.screen_probability}, sort_keys=True
                ),
                "separate_outputs_json": json.dumps(record.separate_probabilities, sort_keys=True),
                "complete_query_output_json": json.dumps(
                    record.batch_probabilities, sort_keys=True
                ),
                "predicate_disagreement_count": record.predicate_disagreement_count,
                "complete_query_disagreement": record.query_disagreement,
                "batching_false_negative": record.false_negative,
                "batching_false_positive": record.false_positive,
                "execution_metadata_json": json.dumps(response_metadata, sort_keys=True),
                "human_review_category": "",
                "human_review_notes": "",
            }
        )
    return output


def _validation_report(
    records: tuple[EvidenceRecord, ...],
    paired: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> str:
    checkpoint = {
        row["policy"]: row
        for row in paired
        if row["setting"] == "end_to_end"
        and row["pool"] == "m531_all"
        and row["budget_kind"] == "requests"
        and float(row["budget_value"]) == 100
    }
    positive = checkpoint["positive_oriented"]
    hybrid = checkpoint["hybrid_weighted"]

    def verdict(row: dict[str, Any], metric: str) -> str:
        low = float(row[f"{metric}_difference_ci_low"])
        high = float(row[f"{metric}_difference_ci_high"])
        return "supported" if low > 0 else "unsupported" if high < 0 else "inconclusive"

    workload_counts = []
    for workload in sorted({record.workload_id for record in records}):
        group = [record for record in records if record.workload_id == workload]
        workload_counts.append(
            f"| {workload} | {len(group)} | "
            f"{sum(row.predicate_disagreement_count for row in group)} | "
            f"{sum(row.query_disagreement for row in group)} | "
            f"{sum(row.separate_query_positive for row in group)} |"
        )
    return "\n".join(
        [
            "# Milestone 5.3.1 — Independent Live Validation",
            "",
            "## Completion status",
            "",
            "Implementation complete; dry-run validated; live provider execution complete; "
            "independent evaluation complete.",
            "",
            f"The session acquired {coverage['completed_requests']} physical requests across "
            f"{len(records)} fully paired independent records, used "
            f"{coverage['http_attempts']} HTTP attempts, and incurred "
            f"${coverage['token_priced_cost_usd']:.8f} in token-priced cost.",
            "",
            "| Workload | Records | Predicate disagreements | Query disagreements | Positives |",
            "|---|---:|---:|---:|---:|",
            *workload_counts,
            "",
            "## RQ1 — Positive acquisition",
            "",
            f"At 100 end-to-end requests, positive-oriented minus random had a paired mean "
            f"positive difference of {positive['positives_difference_mean']:.3f} "
            f"(policy-seed 95% bootstrap interval "
            f"{positive['positives_difference_ci_low']:.3f} to "
            f"{positive['positives_difference_ci_high']:.3f}). Verdict: "
            f"**{verdict(positive, 'positives')}**.",
            "",
            "## RQ2 — Disagreement acquisition",
            "",
            f"Weighted hybrid minus random had a paired mean predicate-disagreement "
            f"difference of {hybrid['predicate_disagreements_difference_mean']:.3f} "
            f"({hybrid['predicate_disagreements_difference_ci_low']:.3f} to "
            f"{hybrid['predicate_disagreements_difference_ci_high']:.3f}). Verdict for "
            f"execution-disagreement discovery: "
            f"**{verdict(hybrid, 'predicate_disagreements')}**. Human review is required "
            "before calling any case a genuine semantic error.",
            "",
            "## RQ3 — Workload generalization",
            "",
            "`workload_specific_results.csv` contains matched workload comparisons. A strategy "
            "generalizes only if its direction is consistent across all three workload pools; "
            "pooled ordering alone is insufficient.",
            "",
            "## Limitations",
            "",
            "Representative and enriched strata have different selection mechanisms. Repeated "
            "policy seeds measure selection variability over one fixed provider outcome per "
            "record, not provider-response variability or additional independent evidence. Rare "
            "disagreement counts can make disagreement-policy verdicts inconclusive.",
            "",
        ]
    )


def generate_analysis(session: Path, output: Path) -> dict[str, Any]:
    records, coverage = load_fresh_evidence(session)
    if coverage["fully_paired_records"] != coverage["selected_records"]:
        observed = coverage["fully_paired_records"]
        selected = coverage["selected_records"]
        raise RuntimeError(
            f"independent analysis requires complete compatible evidence; "
            f"coverage={observed}/{selected}"
        )
    pools: dict[str, tuple[EvidenceRecord, ...]] = {"m531_all": records}
    for workload in sorted({record.workload_id for record in records}):
        pools[f"workload:{workload}"] = tuple(
            record for record in records if record.workload_id == workload
        )
    for stratum in sorted({record.partition for record in records}):
        pools[f"stratum:{stratum}"] = tuple(
            record for record in records if record.partition == stratum
        )
    seed_rows: list[dict[str, Any]] = []
    for pool, pool_records in pools.items():
        seed_rows.extend(_run_pool(pool_records, pool))
    aggregates = _aggregate(seed_rows)
    paired = _paired_comparisons(seed_rows)
    intervals = _event_intervals(pools)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "seed_level_results.csv", seed_rows)
    _write_csv(output / "policy_comparison_results.csv", aggregates)
    _write_csv(output / "paired_policy_comparisons.csv", paired)
    workload_rows = [row for row in aggregates if str(row["pool"]).startswith("workload:")]
    _write_csv(output / "workload_specific_results.csv", workload_rows)
    _write_csv(output / "sparse_event_intervals.csv", intervals)
    reviews = _review_rows(records, session)
    review_fields = [
        "workload_id",
        "record_id",
        "stratum",
        "original_input_text",
        "predicate_definitions_json",
        "screen_output_json",
        "separate_outputs_json",
        "complete_query_output_json",
        "predicate_disagreement_count",
        "complete_query_disagreement",
        "batching_false_negative",
        "batching_false_positive",
        "execution_metadata_json",
        "human_review_category",
        "human_review_notes",
    ]
    _write_csv(output / "human_disagreement_review.csv", reviews, fieldnames=review_fields)
    counts = {
        "predicate_disagreements": sum(record.predicate_disagreement_count for record in records),
        "complete_query_disagreements": sum(record.query_disagreement for record in records),
        "batching_false_negatives": sum(record.false_negative for record in records),
        "separate_query_positives": sum(record.separate_query_positive for record in records),
    }
    report = {
        "schema_version": "m5.3.1-independent-validation-analysis-v1",
        "status": "INDEPENDENT_EVALUATION_COMPLETE",
        "coverage": coverage,
        "independent_records": len(records),
        "distinct_event_counts": counts,
        "provider_response_repetitions_per_record": 1,
        "provider_response_variability_estimated": False,
        "policy_selection_seeds": len(REPLAY_SEEDS),
        "seed_interpretation": (
            "repeated seeds estimate policy-selection variability, not independent "
            "provider observations"
        ),
        "human_review_required_for_genuine_semantic_error_claims": True,
    }
    _write_json(output / "analysis_summary.json", report)
    _write_json(output / "provider_evidence_coverage_report.json", coverage)
    rendered_report = _validation_report(records, paired, coverage)
    (output / "live_validation_report.md").write_text(rendered_report, encoding="utf-8")
    (Path("docs") / "milestone_5_3_1_live_validation_report.md").write_text(
        rendered_report, encoding="utf-8"
    )
    _write_artifact_manifest(
        output,
        json.loads((session / "frozen_experiment_config.json").read_text("utf-8"))[
            "configuration_fingerprint"
        ],
    )
    return report

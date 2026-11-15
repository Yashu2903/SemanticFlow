from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from semanticflow.dataset.loader import load_dataset

INPUT_PRICE_PER_MILLION = 0.042


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _cost(rows: Iterable[dict[str, Any]]) -> float:
    return sum(float(row.get("input_tokens") or 0) for row in rows) * (
        INPUT_PRICE_PER_MILLION / 1_000_000
    )


def diagnose_stage_b(root: str | Path, data_root: str | Path) -> dict[str, Any]:
    """Decompose frozen M5 Stage B evidence without provider access."""
    root = Path(root)
    data_root = Path(data_root)
    groups: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    reference_uncertainty: list[dict[str, Any]] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset_id = dataset_dir.name
        records = {
            record.review_id: record.review_text
            for record in load_dataset(data_root / f"{dataset_id}.parquet")
        }
        reference_rows = _jsonl(dataset_dir / "separate_reference" / "decisions.jsonl")
        reference = {
            (str(row["review_id"]), str(row["predicate_id"])): row
            for row in reference_rows
        }
        for group_dir in sorted(dataset_dir.glob("conjunctive_binary_*")):
            operator = json.loads((group_dir / "operator.json").read_text("utf-8"))
            fused_rows = _jsonl(group_dir / "decisions.jsonl")
            predicate_ids = list(operator["predicate_ids"])
            all_separate: list[dict[str, Any]] = []
            early_separate: list[dict[str, Any]] = []
            accepted = 0
            for fused in fused_rows:
                record_id = str(fused["review_id"])
                rows = [reference[(record_id, predicate_id)] for predicate_id in predicate_ids]
                all_separate.extend(rows)
                for row in rows:
                    early_separate.append(row)
                    if not bool(row["passed"]):
                        break
                reference_pass = all(bool(row["passed"]) for row in rows)
                accepted += reference_pass
                if reference_pass != bool(fused["passed"]):
                    text = records[record_id]
                    lowered = text.casefold()
                    categories: list[str] = []
                    if any(word in lowered.split() for word in ("not", "no", "never", "without")):
                        categories.append("possible_negation_handling")
                    if len(predicate_ids) > 2:
                        categories.append("possible_compound_condition_confusion")
                    if text.count(".") + text.count(";") >= 2:
                        categories.append("possible_scope_or_multiple_situations")
                    if not categories:
                        categories.append("unknown_cause")
                    mismatches.append(
                        {
                            "dataset": dataset_id,
                            "record_id": record_id,
                            "predicate_group": predicate_ids,
                            "fused_question": operator["generated_prompt"],
                            "individual_questions": operator["source_instructions"],
                            "record_text": text,
                            "separate_outputs": [
                                {
                                    "predicate_id": row["predicate_id"],
                                    "probability": row["probability"],
                                    "passed": row["passed"],
                                }
                                for row in rows
                            ],
                            "fused_output": {
                                "probability": fused["group_probability"],
                                "passed": fused["passed"],
                            },
                            "disagreement_type": (
                                "false_negative_against_separate_reference"
                                if reference_pass
                                else "false_positive_against_separate_reference"
                            ),
                            "diagnostic_categories": categories,
                            "linguistic_cause_established": False,
                        }
                    )
            fused_cost = _cost(fused_rows)
            early_cost = _cost(early_separate)
            full_separate_cost = _cost(all_separate)
            fused_input = sum(int(row.get("input_tokens") or 0) for row in fused_rows)
            fused_output = sum(int(row.get("output_tokens") or 0) for row in fused_rows)
            early_input = sum(int(row.get("input_tokens") or 0) for row in early_separate)
            early_output = sum(int(row.get("output_tokens") or 0) for row in early_separate)
            groups.append(
                {
                    "dataset": dataset_id,
                    "group_id": operator["group_id"],
                    "predicate_ids": predicate_ids,
                    "records": len(fused_rows),
                    "accepted_by_separate_reference": accepted,
                    "separate_all_request_count": len(all_separate),
                    "separate_early_reject_request_count": len(early_separate),
                    "fused_request_count": len(fused_rows),
                    "separate_all_input_tokens": sum(
                        int(row.get("input_tokens") or 0) for row in all_separate
                    ),
                    "separate_early_reject_input_tokens": early_input,
                    "fused_input_tokens": fused_input,
                    "separate_early_reject_output_tokens": early_output,
                    "fused_output_tokens": fused_output,
                    "separate_early_reject_cost_usd": early_cost,
                    "fused_cost_usd": fused_cost,
                    "economic_difference_usd": fused_cost - early_cost,
                    "loss_of_early_rejection_cost_avoided_usd": (
                        full_separate_cost - early_cost
                    ),
                    "observed_fused_input_token_delta": fused_input - early_input,
                    "observed_fused_output_token_delta": fused_output - early_output,
                    "additional_prompt_characters_vs_longest_source": len(
                        operator["generated_prompt"]
                    )
                    - max(map(len, operator["source_instructions"])),
                    "cost_per_input_record_usd": fused_cost / len(fused_rows),
                    "cost_per_accepted_reference_record_usd": (
                        fused_cost / accepted if accepted else None
                    ),
                    "causal_decomposition_limitation": (
                        "token components overlap; provider request overhead is not separately "
                        "observable in stored usage"
                    ),
                }
            )
        multi_dirs = sorted(dataset_dir.glob("multi_decision_*"))
        for group_dir in multi_dirs:
            rows = _jsonl(group_dir / "decisions.jsonl")
            for row in rows:
                for predicate_id, probability in row["predicate_probabilities"].items():
                    separate = reference[(str(row["review_id"]), predicate_id)]
                    batch_pass = float(probability) >= float(row["threshold"])
                    if batch_pass != bool(separate["passed"]):
                        reference_uncertainty.append(
                            {
                                "dataset": dataset_id,
                                "record_id": row["review_id"],
                                "predicate_id": predicate_id,
                                "separate_probability": separate["probability"],
                                "batched_probability": probability,
                                "interpretation": (
                                    "context-dependent disagreement; not a repeated "
                                    "identical request"
                                ),
                            }
                        )
    return {
        "schema_version": "m5.1-stage-b-fusion-diagnostics-v1",
        "stage_b_groups": groups,
        "measured_disagreements": mismatches,
        "multi_decision_context_disagreements": reference_uncertainty,
        "reference_reliability": {
            "identical_request_replicates_available": False,
            "semantic_ground_truth_available": False,
            "conclusion": (
                "Separate execution is a reproducible LLM reference, not ground truth; "
                "stored Stage B traces cannot identify model-output instability independently."
            ),
        },
        "live_jev_calls": 0,
    }

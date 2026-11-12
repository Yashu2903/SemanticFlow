"""Risk-aware plan selection over paired bootstrap cost distributions.

Decision codes are machine-readable. Bootstrap frequencies are resampling diagnostics, not
calibrated production probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from semanticflow.optimizer.bootstrap import (
    collect_paired_bootstrap,
    order_key,
    select_minimum,
)
from semanticflow.optimizer.disagreement import compare_estimators as estimator_disagreement
from semanticflow.optimizer.economics import (
    break_even_executions,
    decide_adaptive_profiling_start,
    decide_optimization,
)
from semanticflow.optimizer.matrix import ProfileMatrix
from semanticflow.optimizer.models import (
    BootstrapSettings,
    CorrelationSettings,
    EconomicsSettings,
    OptimizationMetric,
    RiskSettings,
)
from semanticflow.optimizer.regret import plan_risk_metrics
from semanticflow.optimizer.uncertainty import cost_distributions

RISK_SCHEMA_VERSION = "m4-risk-aware-v1"
Decision = Literal[
    "KEEP_ORIGINAL",
    "ADOPT_REORDER",
    "SKIP_PROFILING",
    "REUSE_EXISTING_PLAN",
]


def net_benefit_draws(
    savings: Sequence[float],
    *,
    executions: int,
    new_profiling_cost: float,
    optimization_overhead: float,
) -> list[float]:
    charge = new_profiling_cost + optimization_overhead
    return [executions * value - charge for value in savings]


def _list(order: Sequence[str]) -> list[str]:
    return list(order)


def _bootstrap(
    matrix: ProfileMatrix,
    *,
    estimator_name: str,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    settings: BootstrapSettings,
    correlation: CorrelationSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    tie_relative_tolerance: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return collect_paired_bootstrap(
        matrix,
        estimator_name=estimator_name,
        original_order=original_order,
        objective=objective,
        evaluation_record_count=evaluation_record_count,
        question_hashes=question_hashes,
        settings=settings,
        correlation=correlation,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        tie_relative_tolerance=tie_relative_tolerance,
        iterations=iterations,
        seed=seed,
    )


def _eligible(
    metrics: Mapping[str, Any],
    *,
    original: tuple[str, ...],
    candidate: tuple[str, ...],
    risk: RiskSettings,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected_saving = float(metrics["predicted_saving_vs_original"]["mean"])
    relative = float(metrics["predicted_relative_saving_vs_original"]["mean"])
    probability = float(metrics["probability_cheaper_than_original"])
    expected_regret = float(metrics["expected_regret_vs_modeled_minimum"]["mean"])
    cvar_loss = float(metrics["cvar_loss_vs_original"]["cvar"])
    if candidate == original:
        return True, []
    if expected_saving < risk.minimum_expected_saving:
        reasons.append("expected_saving_below_minimum")
    if relative < risk.minimum_relative_saving:
        reasons.append("expected_saving_smaller_than_configured_risk_tolerance")
    if probability < risk.minimum_saving_probability:
        reasons.append("bootstrap_saving_probability_below_minimum")
    if risk.maximum_expected_regret is not None and expected_regret > risk.maximum_expected_regret:
        reasons.append("expected_regret_above_maximum")
    if risk.maximum_cvar_loss is not None and cvar_loss > risk.maximum_cvar_loss:
        reasons.append("cvar_loss_above_maximum")
    return not reasons, reasons


def _reason_text(decision: Decision, reasons: Sequence[str], selected: Sequence[str]) -> str:
    if decision == "KEEP_ORIGINAL":
        if "expected_saving_smaller_than_configured_risk_tolerance" in reasons:
            return "Expected saving is smaller than the configured risk tolerance."
        if "bootstrap_saving_probability_below_minimum" in reasons:
            return (
                "Bootstrap resampling did not show a stable saving relative to the original order."
            )
        if "estimators_disagree_that_original_is_worse" in reasons:
            return "Estimators disagree on whether leaving the original order is justified."
        if "all_reorders_fail_risk_gates" in reasons:
            return "No reordered plan passed the configured risk gates."
        if "economics_skip_profiling" in reasons:
            return "Expected savings cannot justify new profiling within the reuse horizon."
        if "economics_decline_optimization" in reasons:
            return "Forward-looking net benefit is not above the configured minimum."
        return "Retaining the original order is the conservative feasible decision."
    if decision == "SKIP_PROFILING":
        return "No additional profiling; execute the original order."
    if decision == "REUSE_EXISTING_PLAN":
        return "Reuse the existing selected plan; no new profiling."
    return f"Adopt {' -> '.join(selected)} under the configured risk gates."


def select_risk_aware_plan(
    matrix: ProfileMatrix | None,
    *,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    bootstrap: BootstrapSettings,
    correlation: CorrelationSettings,
    risk: RiskSettings,
    economics: EconomicsSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    tie_relative_tolerance: float,
    primary_estimator: str,
    robust: bool = False,
    statistics_state: Literal["none", "available"] = "available",
    new_profiling_cost: float | None = None,
    sunk_profiling_cost: float = 0.0,
    existing_plan_available: bool = False,
    first_checkpoint_decisions: int | None = None,
    first_checkpoint_cost: float | None = None,
    savings_upper_bound_per_execution: float | None = None,
    primary_collected: Mapping[str, Any] | None = None,
    independent_collected: Mapping[str, Any] | None = None,
    correlation_collected: Mapping[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    original = tuple(original_order)
    iterations = bootstrap.iterations
    seed = bootstrap.seed
    policy = "risk_aware_robust" if robust else "risk_aware"
    explanation: dict[str, Any] = {
        "schema_version": RISK_SCHEMA_VERSION,
        "policy": policy,
        "primary_estimator": primary_estimator,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "interpretation": "sampling_uncertainty_diagnostic",
        "bootstrap_probability_is_not_calibrated": True,
        "original_order": _list(original),
        "risk_settings": risk.model_dump(mode="json"),
    }

    if statistics_state == "none":
        start = decide_adaptive_profiling_start(
            expected_executions=economics.horizon(),
            first_checkpoint_decisions=first_checkpoint_decisions or 0,
            first_checkpoint_cost=first_checkpoint_cost,
            settings=economics,
            savings_upper_bound_per_execution=savings_upper_bound_per_execution,
            savings_bound_basis="structural_upper_bound"
            if savings_upper_bound_per_execution is not None
            else None,
            statistics_available=False,
        )
        explanation["pre_profiling_economics"] = start
        if start["action"] != "start_adaptive_profiling":
            decision: Decision = "SKIP_PROFILING"
            reasons = ["economics_skip_profiling", *start.get("reasons", [])]
            explanation.update(
                {
                    "decision": decision,
                    "reasons": reasons,
                    "selected_order": _list(original),
                    "reason": _reason_text(decision, reasons, original),
                }
            )
            return _list(original), explanation

    if matrix is None and primary_collected is None:
        raise ValueError("risk-aware selection requires the profiling decision matrix")

    def collected(name: str, cached: Mapping[str, Any] | None) -> dict[str, Any]:
        if cached is not None:
            return dict(cached)
        if matrix is None:
            raise ValueError("risk-aware selection requires the profiling decision matrix")
        return _bootstrap(
            matrix,
            estimator_name=name,
            original_order=original,
            objective=objective,
            evaluation_record_count=evaluation_record_count,
            question_hashes=question_hashes,
            settings=bootstrap,
            correlation=correlation,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            tie_relative_tolerance=tie_relative_tolerance,
            iterations=iterations,
            seed=seed,
        )

    primary = collected(primary_estimator, primary_collected)
    primary_min = tuple(primary["full_sample_selected_order"])
    distributions = cost_distributions(
        primary["draws"],
        original_order=original,
        interval_level=bootstrap.interval_level,
        cvar_level=risk.cvar_level,
    )
    explanation["uncertainty"] = {
        key: value for key, value in distributions.items() if key != "risk_by_order"
    }
    explanation["risk_by_order"] = distributions["risk_by_order"]

    if robust:
        independent_pack = collected("independent", independent_collected)
        correlation_pack = (
            primary
            if primary_estimator == "correlation_aware"
            else collected("correlation_aware", correlation_collected)
        )
        if primary_estimator == "independent":
            independent_pack = primary
        disagreement = estimator_disagreement(
            independent=independent_pack["draws"],
            correlation_aware=correlation_pack["draws"],
            original_order=original,
            independent_selected=independent_pack["full_sample_selected_order"],
            correlation_selected=correlation_pack["full_sample_selected_order"],
            interval_level=bootstrap.interval_level,
        )
        explanation["disagreement"] = disagreement
        if (
            risk.disagreement_policy == "agree_original_is_worse"
            and not disagreement["agree_original_is_worse"]
            and not disagreement["select_same_order"]
        ):
            reasons = ["estimators_disagree_that_original_is_worse"]
            explanation.update(
                {
                    "decision": "KEEP_ORIGINAL",
                    "reasons": reasons,
                    "selected_order": _list(original),
                    "minimum_objective_order": _list(primary_min),
                    "reason": _reason_text("KEEP_ORIGINAL", reasons, original),
                }
            )
            return _list(original), explanation

    eligible: list[tuple[str, ...]] = []
    rejections: dict[str, list[str]] = {}
    for order in primary["orders"]:
        metrics = distributions["risk_by_order"][order_key(order)]
        ok, reasons = _eligible(metrics, original=original, candidate=order, risk=risk)
        if ok:
            eligible.append(order)
        else:
            rejections[order_key(order)] = reasons
    explanation["gate_rejections"] = rejections

    if not eligible:
        selected = original
        decision = "KEEP_ORIGINAL"
        reasons = ["all_reorders_fail_risk_gates"]
    else:
        count = primary["iterations"]
        means = {order: float(sum(primary["draws"][order]) / count) for order in eligible}
        selected = select_minimum(means, original, tie_relative_tolerance)
        if selected == original:
            decision = "KEEP_ORIGINAL"
            min_key = order_key(primary_min)
            if primary_min != original and min_key in rejections:
                reasons = ["all_reorders_fail_risk_gates", *rejections[min_key]]
            elif set(eligible) <= {original}:
                reasons = ["all_reorders_fail_risk_gates"]
            else:
                reasons = ["original_is_minimum_among_gated_candidates"]
        else:
            decision = "ADOPT_REORDER"
            reasons = ["passed_risk_gates"]

    selected_metrics = plan_risk_metrics(
        primary["draws"],
        original_order=original,
        candidate_order=selected,
        interval_level=bootstrap.interval_level,
        cvar_level=risk.cvar_level,
    )
    savings = [
        primary["draws"][original][index] - primary["draws"][selected][index]
        for index in range(primary["iterations"])
    ]
    horizon = economics.horizon()
    incremental = 0.0 if statistics_state == "available" else (new_profiling_cost or 0.0)
    overhead = 0.0 if statistics_state == "available" else economics.optimization_overhead_usd
    net = net_benefit_draws(
        savings,
        executions=horizon,
        new_profiling_cost=incremental,
        optimization_overhead=overhead,
    )
    explanation["net_benefit"] = {
        "horizon_executions": horizon,
        "expected_executions_known": economics.expected_executions is not None,
        "new_profiling_cost_charged_usd": incremental,
        "sunk_profiling_cost_usd": sunk_profiling_cost,
        "optimization_overhead_usd": overhead,
        "mean_usd": sum(net) / len(net),
        "probability_positive": sum(value > 0 for value in net) / len(net),
        "break_even_executions_vs_new_profiling": break_even_executions(
            (new_profiling_cost or 0.0) + economics.optimization_overhead_usd,
            float(selected_metrics["predicted_saving_vs_original"]["mean"]),
        ),
        "note": (
            "sunk profiling is reported and not subtracted from forward net benefit; "
            "an optimistic structural bound is never treated as an expected saving"
        ),
    }

    if risk.apply_profiling_economics and statistics_state == "available":
        economics_decision = decide_optimization(
            settings=economics,
            original_cost_per_execution=float(
                sum(primary["draws"][original]) / primary["iterations"]
            ),
            optimized_cost_per_execution=float(
                sum(primary["draws"][selected]) / primary["iterations"]
            ),
            statistics_state="available",
            new_profiling_cost=new_profiling_cost,
            sunk_profiling_cost=sunk_profiling_cost,
            existing_plan_available=existing_plan_available,
            savings_interval=(
                float(selected_metrics["predicted_saving_vs_original"]["lower"]),
                float(selected_metrics["predicted_saving_vs_original"]["upper"]),
            ),
            savings_probability=float(selected_metrics["probability_cheaper_than_original"]),
            savings_basis="bootstrap_interval",
        )
        explanation["profiling_economics"] = economics_decision
        if economics_decision["action"] in {
            "decline_optimization_execute_original",
            "execute_original_without_profiling",
        }:
            selected = original
            decision = "KEEP_ORIGINAL"
            reasons = ["economics_decline_optimization", *economics_decision.get("reasons", [])]
        elif economics_decision["action"] == "reuse_existing_plan":
            decision = "REUSE_EXISTING_PLAN"

    explanation.update(
        {
            "decision": decision,
            "reasons": reasons,
            "selected_order": _list(selected),
            "minimum_objective_order": _list(primary_min),
            "selected_risk": selected_metrics,
            "reason": _reason_text(decision, reasons, selected),
        }
    )
    return _list(selected), explanation

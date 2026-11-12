"""Joint sequential profiling and risk-aware plan selection.

The controller receives only the rows revealed at the current checkpoint.  Stored rows from
later checkpoints are owned by the offline replay harness and are revealed only after a
``PROFILE_MORE`` action.  Statistical simulations in this module are predictive draws, not
fabricated Jev observations.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from math import sqrt
from statistics import NormalDist
from typing import Any, Literal

from semanticflow.optimizer.adaptive import sampling_order
from semanticflow.optimizer.bootstrap import (
    _summary,
    collect_paired_bootstrap,
    collect_sparse_posterior_bootstrap,
    order_key,
    sample_std,
    select_minimum,
)
from semanticflow.optimizer.economics import decide_adaptive_profiling_start
from semanticflow.optimizer.matrix import ProfileMatrix, joint_statistics
from semanticflow.optimizer.models import (
    BootstrapSettings,
    CorrelationSettings,
    EconomicsSettings,
    JointAdaptiveSettings,
    OptimizationMetric,
    RiskSettings,
)
from semanticflow.optimizer.risk import select_risk_aware_plan

JOINT_SCHEMA_VERSION = "m4-joint-adaptive-risk-v1"
JointAction = Literal[
    "ADOPT_REORDER",
    "PROFILE_MORE",
    "KEEP_ORIGINAL",
    "SKIP_PROFILING",
    "REUSE_EXISTING_PLAN",
]


def _beta_uncertainty(successes: int, trials: int) -> dict[str, float | int]:
    """Jeffreys-smoothed Bernoulli uncertainty used only for checkpoint diagnostics."""
    alpha = successes + 0.5
    beta = trials - successes + 0.5
    total = alpha + beta
    return {
        "successes": successes,
        "trials": trials,
        "posterior_mean": alpha / total,
        "posterior_std": sqrt(alpha * beta / (total * total * (total + 1.0))),
    }


def _checkpoint_statistical_uncertainty(
    evidence: ProfileMatrix,
    collected: Mapping[str, Any],
    explanation: Mapping[str, Any],
    *,
    next_sample_size: int | None,
    contraction_exponent: float,
) -> dict[str, Any]:
    """Structured uncertainty diagnostics, separate from the controller's decision inputs."""
    marginals = {}
    for index, predicate_id in enumerate(evidence.predicate_ids):
        successes = sum(row.outcomes[index] for row in evidence.rows)
        marginals[predicate_id] = _beta_uncertainty(successes, evidence.size)
    conditionals = {}
    for left_index, left in enumerate(evidence.predicate_ids):
        support = sum(row.outcomes[left_index] for row in evidence.rows)
        for right_index, right in enumerate(evidence.predicate_ids):
            if left == right:
                continue
            successes = sum(
                row.outcomes[left_index] and row.outcomes[right_index]
                for row in evidence.rows
            )
            conditionals[f"{right} | {left}"] = _beta_uncertainty(successes, support)
    frequency = Counter(order_key(order) for order in collected["selected_orders"])
    iterations = int(collected["iterations"])
    plan_ranking_uncertainty = 1.0 - max(frequency.values()) / iterations
    selected_risk = explanation.get("selected_risk") or {}
    selected_regret = selected_risk.get("expected_regret_vs_modeled_minimum") or {}
    contraction = (
        None
        if next_sample_size is None
        else (evidence.size / next_sample_size) ** contraction_exponent
    )
    return {
        "method": "checkpoint_local_jeffreys_and_posterior_predictive_diagnostics",
        "decision_input": False,
        "marginal_selectivity": marginals,
        "conditional_selectivity": conditionals,
        "candidate_cost_std_by_order": {
            order_key(order): sample_std(values)
            for order, values in sorted(collected["draws"].items())
        },
        "plan_ranking_uncertainty": plan_ranking_uncertainty,
        "selected_plan_expected_regret": {
            "mean": selected_regret.get("mean"),
            "std": selected_regret.get("std"),
        },
        "root_n_projection_to_next_checkpoint": {
            "next_sample_size": next_sample_size,
            "contraction": contraction,
            "exponent": contraction_exponent,
        },
        "limitations": (
            "Jeffreys beta summaries are marginal and pairwise diagnostics; they do not replace "
            "the joint outcome-pattern posterior or model Jev response drift"
        ),
    }


def _profile_cost(
    matrix: ProfileMatrix,
    *,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> float | None:
    if input_price_per_million is None or output_price_per_million is None:
        return None
    total = 0.0
    for row in matrix.rows:
        for input_tokens, output_tokens in zip(row.input_tokens, row.output_tokens, strict=True):
            if input_tokens is None or output_tokens is None:
                return None
            total += (
                input_tokens * input_price_per_million
                + output_tokens * output_price_per_million
            ) / 1_000_000
    return total


def collect_checkpoint_uncertainty(
    evidence: ProfileMatrix,
    *,
    estimator_name: str,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    bootstrap: BootstrapSettings,
    correlation: CorrelationSettings,
    joint: JointAdaptiveSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    tie_relative_tolerance: float,
    seed: int,
    sparse_posterior_predictive: bool | None = None,
) -> dict[str, Any]:
    """Build aligned cost draws using only ``evidence`` from the current checkpoint."""
    sparse = (
        joint.sparse_posterior_predictive
        if sparse_posterior_predictive is None
        else sparse_posterior_predictive
    )
    common = {
        "estimator_name": estimator_name,
        "original_order": original_order,
        "objective": objective,
        "evaluation_record_count": evaluation_record_count,
        "question_hashes": question_hashes,
        "settings": bootstrap,
        "correlation": correlation,
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
        "tie_relative_tolerance": tie_relative_tolerance,
        "iterations": joint.bootstrap_iterations,
        "seed": seed,
    }
    if sparse:
        return collect_sparse_posterior_bootstrap(
            evidence,
            prior_strength=joint.predictive_prior_strength,
            **common,
        )
    return collect_paired_bootstrap(evidence, **common)


def _projected_collection(
    collected: Mapping[str, Any],
    *,
    scenario_index: int,
    next_sample_size: int,
    contraction_exponent: float,
) -> dict[str, Any]:
    """Approximate the posterior cost distribution after one additional checkpoint.

    The future evidence center is a sample-size-weighted update toward one predictive cost
    draw.  Dispersion around that center contracts at the configured root-n-like rate.  No
    stored future row appears in this calculation.
    """
    current_size = int(collected["sample_size"])
    if next_sample_size <= current_size:
        raise ValueError("next profiling checkpoint must exceed the current sample size")
    count = int(collected["iterations"])
    orders = tuple(collected["orders"])
    draws = collected["draws"]
    means = {order: sum(draws[order]) / count for order in orders}
    added = next_sample_size - current_size
    future_center = {
        order: (
            current_size * means[order] + added * draws[order][scenario_index]
        )
        / next_sample_size
        for order in orders
    }
    contraction = (current_size / next_sample_size) ** contraction_exponent
    future_draws = {
        order: [
            future_center[order] + contraction * (value - means[order])
            for value in draws[order]
        ]
        for order in orders
    }
    original = tuple(collected["original_order"])
    tolerance = float(collected["tie_relative_tolerance"])
    selected = select_minimum(future_center, original, tolerance)
    selected_orders = []
    for index in range(count):
        costs = {order: future_draws[order][index] for order in orders}
        selected_orders.append(select_minimum(costs, original, tolerance))
    return {
        **dict(collected),
        "sample_size": next_sample_size,
        "full_sample_costs": future_center,
        "full_sample_selected_order": selected,
        "draws": future_draws,
        "selected_orders": selected_orders,
        "projection": {
            "method": "predictive_cost_center_with_root_n_uncertainty_contraction",
            "current_sample_size": current_size,
            "next_sample_size": next_sample_size,
            "contraction": contraction,
            "uses_stored_future_rows": False,
        },
    }


def estimate_additional_profiling_value(
    collected: Mapping[str, Any],
    *,
    current_order: Sequence[str],
    next_sample_size: int,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    bootstrap: BootstrapSettings,
    correlation: CorrelationSettings,
    risk: RiskSettings,
    economics: EconomicsSettings,
    joint: JointAdaptiveSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    tie_relative_tolerance: float,
    estimator_name: str,
    expected_executions: int,
    incremental_profiling_cost_usd: float | None,
    seed: int,
    robust: bool = False,
    independent_collected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Approximate EVSI from current evidence, then subtract the next checkpoint cost."""
    orders = tuple(collected["orders"])
    draws = collected["draws"]
    iterations = int(collected["iterations"])
    current = tuple(current_order)
    scenario_count = min(joint.voi_scenarios, iterations)
    indices = list(range(iterations))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[:scenario_count])
    reductions: list[float] = []
    current_losses: list[float] = []
    future_losses: list[float] = []
    changed_plan_reductions: list[float] = []
    unchanged_plan_reductions: list[float] = []
    projected_plan_ranking_uncertainties: list[float] = []
    projected_selected_regret_stds: list[float] = []
    action_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    for index in indices:
        latent_minimum = min(draws[order][index] for order in orders)
        current_loss = draws[current][index] - latent_minimum
        projected = _projected_collection(
            collected,
            scenario_index=index,
            next_sample_size=next_sample_size,
            contraction_exponent=joint.uncertainty_contraction_exponent,
        )
        projected_independent = (
            _projected_collection(
                independent_collected,
                scenario_index=index,
                next_sample_size=next_sample_size,
                contraction_exponent=joint.uncertainty_contraction_exponent,
            )
            if independent_collected is not None
            else None
        )
        future_order, future_explanation = select_risk_aware_plan(
            None,
            original_order=original_order,
            objective=objective,
            evaluation_record_count=evaluation_record_count,
            question_hashes=question_hashes,
            bootstrap=bootstrap,
            correlation=correlation,
            risk=risk,
            economics=economics,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            tie_relative_tolerance=tie_relative_tolerance,
            primary_estimator=estimator_name,
            robust=robust,
            statistics_state="available",
            primary_collected=projected,
            independent_collected=projected_independent,
            correlation_collected=projected if estimator_name == "correlation_aware" else None,
        )
        future = tuple(future_order)
        future_loss = draws[future][index] - latent_minimum
        current_losses.append(current_loss)
        future_losses.append(future_loss)
        reductions.append(current_loss - future_loss)
        if future == current:
            unchanged_plan_reductions.append(current_loss - future_loss)
        else:
            changed_plan_reductions.append(current_loss - future_loss)
        projected_frequency = Counter(order_key(order) for order in projected["selected_orders"])
        projected_plan_ranking_uncertainties.append(
            1.0 - max(projected_frequency.values()) / iterations
        )
        projected_selected_risk = future_explanation.get("selected_risk") or {}
        projected_regret = (
            projected_selected_risk.get("expected_regret_vs_modeled_minimum") or {}
        )
        if projected_regret.get("std") is not None:
            projected_selected_regret_stds.append(float(projected_regret["std"]))
        action_counts[str(future_explanation["decision"])] += 1
        selected_counts[order_key(future)] += 1
    current_mean = sum(current_losses) / scenario_count
    future_mean = sum(future_losses) / scenario_count
    reduction = sum(reductions) / scenario_count
    reduction_std = sample_std(reductions)
    reduction_standard_error = reduction_std / sqrt(scenario_count)
    normal_95_half_width = 1.959963984540054 * reduction_standard_error
    one_sided_z = NormalDist().inv_cdf(joint.voi_confidence_level)
    lower_confidence_reduction = reduction - one_sided_z * reduction_standard_error
    gross = expected_executions * reduction
    net = None if incremental_profiling_cost_usd is None else gross - incremental_profiling_cost_usd
    probability_plan_changes = len(changed_plan_reductions) / scenario_count
    changed_plan_contribution = sum(changed_plan_reductions) / scenario_count
    unchanged_plan_contribution = sum(unchanged_plan_reductions) / scenario_count
    gross_lower_95 = expected_executions * (reduction - normal_95_half_width)
    net_lower_95 = (
        None
        if incremental_profiling_cost_usd is None
        else gross_lower_95 - incremental_profiling_cost_usd
    )
    lower_confidence_gross = expected_executions * lower_confidence_reduction
    lower_confidence_net = (
        None
        if incremental_profiling_cost_usd is None
        else lower_confidence_gross - incremental_profiling_cost_usd
    )
    continuation_value = (
        net
        if joint.voi_decision_rule == "point_estimate"
        else lower_confidence_net
    )
    return {
        "method": "posterior_predictive_expected_decision_loss_reduction",
        "uses_only_current_checkpoint_evidence": True,
        "uses_stored_future_rows": False,
        "scenario_count": scenario_count,
        "current_sample_size": int(collected["sample_size"]),
        "next_sample_size": next_sample_size,
        "current_expected_decision_loss_per_execution": current_mean,
        "projected_expected_decision_loss_per_execution": future_mean,
        "expected_reduction_in_decision_loss_per_execution": reduction,
        "decision_loss_reduction_distribution": _summary(
            reductions, bootstrap.interval_level
        ),
        "monte_carlo_standard_error_per_execution": reduction_standard_error,
        "normal_95_interval_per_execution": {
            "lower": reduction - normal_95_half_width,
            "upper": reduction + normal_95_half_width,
            "diagnostic_only": True,
        },
        "expected_executions": expected_executions,
        "expected_remaining_execution_count": expected_executions,
        "gross_value_of_information_usd": gross,
        "incremental_profiling_cost_usd": incremental_profiling_cost_usd,
        "net_value_of_information_usd": net,
        "normal_95_lower_gross_value_usd": gross_lower_95,
        "normal_95_lower_net_value_usd": net_lower_95,
        "monte_carlo_confidence": {
            "method": "one_sided_normal_mean_bound",
            "level": joint.voi_confidence_level,
            "z_value": one_sided_z,
            "lower_reduction_per_execution": lower_confidence_reduction,
            "lower_gross_value_usd": lower_confidence_gross,
            "lower_net_value_usd": lower_confidence_net,
            "scope": "finite predictive-scenario Monte Carlo error only",
            "does_not_cover": (
                "posterior model misspecification, contraction error, Jev response drift, "
                "or held-out execution variability"
            ),
        },
        "continuation_value_basis": joint.voi_decision_rule,
        "continuation_value_usd": continuation_value,
        "probability_next_checkpoint_changes_selected_plan": probability_plan_changes,
        "estimated_benefit_from_plan_changes_per_execution": changed_plan_contribution,
        "estimated_benefit_conditional_on_plan_change_per_execution": (
            sum(changed_plan_reductions) / len(changed_plan_reductions)
            if changed_plan_reductions
            else 0.0
        ),
        "estimated_benefit_without_plan_change_per_execution": unchanged_plan_contribution,
        "unchanged_plan_value_interpretation": (
            "zero under the expected decision-loss objective: confidence without an action "
            "change has no execution-cost value"
        ),
        "projected_plan_ranking_uncertainty": (
            sum(projected_plan_ranking_uncertainties)
            / len(projected_plan_ranking_uncertainties)
        ),
        "projected_selected_plan_regret_std": (
            sum(projected_selected_regret_stds) / len(projected_selected_regret_stds)
            if projected_selected_regret_stds
            else None
        ),
        "projected_decision_frequency": {
            key: value / scenario_count for key, value in sorted(action_counts.items())
        },
        "projected_selected_order_frequency": {
            key: value / scenario_count for key, value in sorted(selected_counts.items())
        },
        "limitations": (
            "approximate EVSI under exchangeable reviews, a smoothed outcome-pattern model, "
            "and root-n cost-uncertainty contraction; response variability and distribution "
            "shift are not modeled"
        ),
        "objective_consistency": {
            "decision_loss": "expected cost regret versus the modeled minimum",
            "future_action": "configured risk-aware policy",
            "profiling_cost": "expected USD cost of the next checkpoint",
            "note": (
                "This is policy-constrained EVSI, not unconstrained expected-cost EVSI; risk "
                "gates determine which future action may be selected"
            ),
        },
    }


def analyze_joint_checkpoint(
    evidence: ProfileMatrix,
    *,
    next_sample_size: int | None,
    incremental_profiling_cost_usd: float | None,
    expected_executions: int,
    estimator_name: str,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    bootstrap: BootstrapSettings,
    correlation: CorrelationSettings,
    risk: RiskSettings,
    economics: EconomicsSettings,
    joint: JointAdaptiveSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    tie_relative_tolerance: float,
    seed: int,
    sparse_posterior_predictive: bool | None = None,
    checkpoint_collected: Mapping[str, Any] | None = None,
    robust: bool = False,
    independent_collected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze one evidence snapshot; this function has no future-matrix parameter."""
    collected = (
        dict(checkpoint_collected)
        if checkpoint_collected is not None
        else collect_checkpoint_uncertainty(
            evidence,
            estimator_name=estimator_name,
            original_order=original_order,
            objective=objective,
            evaluation_record_count=evaluation_record_count,
            question_hashes=question_hashes,
            bootstrap=bootstrap,
            correlation=correlation,
            joint=joint,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            tie_relative_tolerance=tie_relative_tolerance,
            seed=seed,
            sparse_posterior_predictive=sparse_posterior_predictive,
        )
    )
    selected, explanation = select_risk_aware_plan(
        evidence,
        original_order=original_order,
        objective=objective,
        evaluation_record_count=evaluation_record_count,
        question_hashes=question_hashes,
        bootstrap=bootstrap,
        correlation=correlation,
        risk=risk,
        economics=economics,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        tie_relative_tolerance=tie_relative_tolerance,
        primary_estimator=estimator_name,
        robust=robust,
        statistics_state="available",
        primary_collected=collected,
        independent_collected=independent_collected,
        correlation_collected=collected if estimator_name == "correlation_aware" else None,
    )
    frequency = Counter(order_key(order) for order in collected["selected_orders"])
    uncertainty = explanation.get("uncertainty") or {}
    snapshot = joint_statistics(evidence)
    minimum_key = order_key(explanation.get("minimum_objective_order") or original_order)
    minimum_risk = (explanation.get("risk_by_order") or {}).get(minimum_key)
    statistical_uncertainty = _checkpoint_statistical_uncertainty(
        evidence,
        collected,
        explanation,
        next_sample_size=next_sample_size,
        contraction_exponent=joint.uncertainty_contraction_exponent,
    )
    value = None
    if next_sample_size is not None and explanation["decision"] not in {
        "ADOPT_REORDER",
        "REUSE_EXISTING_PLAN",
    }:
        value = estimate_additional_profiling_value(
            collected,
            current_order=selected,
            next_sample_size=next_sample_size,
            original_order=original_order,
            objective=objective,
            evaluation_record_count=evaluation_record_count,
            question_hashes=question_hashes,
            bootstrap=bootstrap,
            correlation=correlation,
            risk=risk,
            economics=economics,
            joint=joint,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            tie_relative_tolerance=tie_relative_tolerance,
            estimator_name=estimator_name,
            expected_executions=expected_executions,
            incremental_profiling_cost_usd=incremental_profiling_cost_usd,
            seed=seed + 7919,
            robust=robust,
            independent_collected=independent_collected,
        )
    return {
        "schema_version": JOINT_SCHEMA_VERSION,
        "profiled_reviews": evidence.size,
        "profiling_decisions_consumed": evidence.size * len(evidence.predicate_ids),
        "controller_evidence_fingerprint": evidence.fingerprint(),
        "future_rows_accessible_to_controller": False,
        "current_selectivities": snapshot["marginals"],
        "current_correlations": snapshot["pairwise"],
        "current_estimated_plan_costs": {
            order_key(order): cost
            for order, cost in sorted(collected["full_sample_costs"].items())
        },
        "candidate_cost_distributions": uncertainty.get("by_order"),
        "bootstrap_selection_frequency": {
            key: value / int(collected["iterations"]) for key, value in sorted(frequency.items())
        },
        "risk_decision": explanation["decision"],
        "risk_selected_order": list(selected),
        "risk_reasons": explanation.get("reasons"),
        "risk_reason": explanation.get("reason"),
        "minimum_objective_order": explanation.get("minimum_objective_order"),
        "minimum_objective_risk": minimum_risk,
        "selected_risk": explanation.get("selected_risk"),
        "risk_by_order": explanation.get("risk_by_order"),
        "statistical_uncertainty": statistical_uncertainty,
        "sparse_data_treatment": collected.get("sparse_data_treatment")
        or {"method": "empirical_paired_bootstrap", "can_generate_unseen_positive": False},
        "next_sample_size": next_sample_size,
        "value_of_additional_profiling": value,
        "decision_diagnostics": {
            "current_best_candidate_plan": explanation.get("minimum_objective_order"),
            "current_risk_selected_plan": list(selected),
            "risk_decision": explanation["decision"],
            "risk_reason_codes": explanation.get("reasons") or [],
            "current_selection_confidence": (
                None
                if minimum_risk is None
                else minimum_risk.get("probability_cheaper_than_original")
            ),
            "current_expected_regret": (
                None
                if explanation.get("selected_risk") is None
                else explanation["selected_risk"].get(
                    "expected_regret_vs_modeled_minimum"
                )
            ),
            "current_downside_risk": (
                None
                if explanation.get("selected_risk") is None
                else {
                    "loss": explanation["selected_risk"].get(
                        "downside_loss_vs_original"
                    ),
                    "cvar": explanation["selected_risk"].get("cvar_loss_vs_original"),
                }
            ),
            "next_checkpoint": {
                "sample_size": next_sample_size,
                "incremental_cost_usd": incremental_profiling_cost_usd,
                "probability_selected_plan_changes": (
                    None
                    if value is None
                    else value["probability_next_checkpoint_changes_selected_plan"]
                ),
                "gross_voi_usd": (
                    None if value is None else value["gross_value_of_information_usd"]
                ),
                "net_voi_usd": (
                    None if value is None else value["net_value_of_information_usd"]
                ),
                "continuation_value_basis": (
                    None if value is None else value["continuation_value_basis"]
                ),
                "continuation_value_usd": (
                    None if value is None else value["continuation_value_usd"]
                ),
            },
        },
    }


def choose_joint_action(
    checkpoint: Mapping[str, Any],
    *,
    original_order: Sequence[str],
    profiling_cost_incurred_usd: float | None,
    profiling_budget_usd: float | None,
    maximum_profiling_decisions: int | None,
    minimum_voi_usd: float,
) -> tuple[JointAction, list[str], list[str]]:
    """Map one checkpoint analysis to an action without consulting later checkpoints."""
    risk_decision = checkpoint["risk_decision"]
    selected = list(checkpoint["risk_selected_order"])
    if risk_decision in {"ADOPT_REORDER", "REUSE_EXISTING_PLAN"}:
        return "ADOPT_REORDER", selected, ["candidate_passed_configured_risk_gates"]
    next_size = checkpoint.get("next_sample_size")
    if next_size is None:
        return "KEEP_ORIGINAL", list(original_order), ["maximum_checkpoint_reached"]
    reasons: list[str] = []
    width = len(original_order)
    if maximum_profiling_decisions is not None and next_size * width > maximum_profiling_decisions:
        reasons.append("profiling_decision_budget_exhausted")
    value = checkpoint.get("value_of_additional_profiling") or {}
    increment = value.get("incremental_profiling_cost_usd")
    if (
        profiling_budget_usd is not None
        and profiling_cost_incurred_usd is not None
        and increment is not None
        and profiling_cost_incurred_usd + increment > profiling_budget_usd
    ):
        reasons.append("profiling_cost_budget_exhausted")
    net = value.get("continuation_value_usd", value.get("net_value_of_information_usd"))
    if net is None:
        reasons.append("value_of_information_unavailable")
    elif net <= minimum_voi_usd:
        reasons.append(
            "value_of_additional_profiling_not_positive"
            if value.get("continuation_value_basis", "point_estimate") == "point_estimate"
            else "lower_confidence_value_of_additional_profiling_not_positive"
        )
    if reasons:
        return "KEEP_ORIGINAL", list(original_order), reasons
    return "PROFILE_MORE", list(original_order), ["positive_value_of_additional_profiling"]


def simulate_joint_adaptive_risk(
    matrix: ProfileMatrix,
    *,
    estimator_name: str,
    original_order: Sequence[str],
    objective: OptimizationMetric,
    evaluation_record_count: int,
    question_hashes: Mapping[str, str],
    checkpoints: Sequence[int],
    sampling_seed: int,
    bootstrap: BootstrapSettings,
    correlation: CorrelationSettings,
    risk: RiskSettings,
    economics: EconomicsSettings,
    joint: JointAdaptiveSettings,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    expected_executions: int,
    tie_relative_tolerance: float = 1e-9,
    savings_upper_bound_per_execution: float | None = None,
    estimated_cost_per_decision: float | None = None,
    statistics_available: bool = False,
    reusable_order: Sequence[str] | None = None,
    profiling_budget_usd: float | None = None,
    maximum_profiling_decisions: int | None = None,
    sparse_posterior_predictive: bool | None = None,
) -> dict[str, Any]:
    """Strict offline replay: reveal a new stored prefix only after ``PROFILE_MORE``."""
    if not matrix.complete:
        raise ValueError("joint adaptive replay requires a complete profiling matrix")
    original = list(original_order)
    usable = [value for value in checkpoints if value <= matrix.size]
    if not usable:
        raise ValueError("no joint profiling checkpoint fits the matrix")
    width = len(original)
    first_decisions = usable[0] * width
    first_cost = (
        estimated_cost_per_decision * first_decisions
        if estimated_cost_per_decision is not None
        else None
    )
    start = decide_adaptive_profiling_start(
        expected_executions=expected_executions,
        first_checkpoint_decisions=first_decisions,
        first_checkpoint_cost=first_cost,
        settings=economics,
        savings_upper_bound_per_execution=savings_upper_bound_per_execution,
        savings_bound_basis=(
            "structural_upper_bound" if savings_upper_bound_per_execution is not None else None
        ),
        statistics_available=statistics_available,
        maximum_profiling_decisions=maximum_profiling_decisions,
        adaptive_budget_usd=profiling_budget_usd,
    )
    if start["action"] == "skip_profiling_use_existing_statistics":
        selected = list(reusable_order or original)
        return {
            "schema_version": JOINT_SCHEMA_VERSION,
            "action": "REUSE_EXISTING_PLAN",
            "selected_order": selected,
            "reason": "reusable_statistics_available",
            "start_decision": start,
            "profiling_reviews_consumed": 0,
            "profiling_decisions_consumed": 0,
            "profiling_cost_usd": 0.0,
            "trajectory": [],
            "live_provider_calls": 0,
        }
    if start["action"] != "start_adaptive_profiling":
        return {
            "schema_version": JOINT_SCHEMA_VERSION,
            "action": "SKIP_PROFILING",
            "selected_order": original,
            "reason": ";".join(start.get("reasons") or ["profiling_not_economic"]),
            "start_decision": start,
            "profiling_reviews_consumed": 0,
            "profiling_decisions_consumed": 0,
            "profiling_cost_usd": 0.0,
            "trajectory": [],
            "live_provider_calls": 0,
        }
    ranked = sampling_order([row.review_id for row in matrix.rows], sampling_seed)
    trajectory = []
    final_action: JointAction = "KEEP_ORIGINAL"
    final_order = original
    final_reasons: list[str] = []
    incurred: float | None = 0.0
    for index, size in enumerate(usable):
        # This is the only reveal point.  The controller below receives ``evidence`` and no
        # reference to ``matrix`` or to ranked IDs beyond this prefix.
        evidence = matrix.select_reviews(ranked[:size])
        incurred = _profile_cost(
            evidence,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
        next_size = usable[index + 1] if index + 1 < len(usable) else None
        if next_size is None:
            increment = None
        elif incurred is not None:
            increment = incurred / size * (next_size - size)
        elif estimated_cost_per_decision is not None:
            increment = estimated_cost_per_decision * (next_size - size) * width
        else:
            increment = None
        checkpoint = analyze_joint_checkpoint(
            evidence,
            next_sample_size=next_size,
            incremental_profiling_cost_usd=increment,
            expected_executions=expected_executions,
            estimator_name=estimator_name,
            original_order=original,
            objective=objective,
            evaluation_record_count=evaluation_record_count,
            question_hashes=question_hashes,
            bootstrap=bootstrap,
            correlation=correlation,
            risk=risk,
            economics=economics,
            joint=joint,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            tie_relative_tolerance=tie_relative_tolerance,
            seed=joint.seed + size,
            sparse_posterior_predictive=sparse_posterior_predictive,
        )
        action, selected, reasons = choose_joint_action(
            checkpoint,
            original_order=original,
            profiling_cost_incurred_usd=incurred,
            profiling_budget_usd=profiling_budget_usd,
            maximum_profiling_decisions=maximum_profiling_decisions,
            minimum_voi_usd=joint.minimum_voi_usd,
        )
        checkpoint.update(
            {
                "profiling_cost_incurred_usd": incurred,
                "controller_action": action,
                "controller_selected_order": selected,
                "controller_reasons": reasons,
                "controller_explanation": {
                    "action": action,
                    "reason_codes": reasons,
                    "minimum_net_voi_usd": joint.minimum_voi_usd,
                    "estimated_net_voi_usd": (
                        (checkpoint.get("value_of_additional_profiling") or {}).get(
                            "net_value_of_information_usd"
                        )
                    ),
                    "continuation_value_basis": (
                        (checkpoint.get("value_of_additional_profiling") or {}).get(
                            "continuation_value_basis"
                        )
                    ),
                    "continuation_value_usd": (
                        (checkpoint.get("value_of_additional_profiling") or {}).get(
                            "continuation_value_usd"
                        )
                    ),
                    "profiling_budget_usd": profiling_budget_usd,
                    "maximum_profiling_decisions": maximum_profiling_decisions,
                },
            }
        )
        trajectory.append(checkpoint)
        if action != "PROFILE_MORE":
            final_action, final_order, final_reasons = action, selected, reasons
            break
    return {
        "schema_version": JOINT_SCHEMA_VERSION,
        "action": final_action,
        "selected_order": final_order,
        "reason": ";".join(final_reasons),
        "start_decision": start,
        "profiling_reviews_consumed": trajectory[-1]["profiled_reviews"],
        "profiling_decisions_consumed": trajectory[-1]["profiling_decisions_consumed"],
        "profiling_cost_usd": trajectory[-1]["profiling_cost_incurred_usd"],
        "trajectory": trajectory,
        "sampling": {
            "design": "uniform_without_replacement_nested_complete_rows",
            "seed": sampling_seed,
        },
        "expected_executions": expected_executions,
        "live_provider_calls": 0,
        "held_out_evaluation_evidence_accessible_to_controller": False,
    }

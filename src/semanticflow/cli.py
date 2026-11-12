from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from semanticflow.optimizer.models import load_optimizer_settings
from semanticflow.optimizer.workflow import (
    compare_experiment,
    execute_experiment_strategy,
    plan_experiment,
    profile_experiment,
    run_offline_experiment,
)
from semanticflow.providers.jev import JevProvider
from semanticflow.providers.mock import MockProvider
from semanticflow.query.models import load_query
from semanticflow.runtime import load_settings, run_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SemanticFlow baseline semantic query")
    parser.add_argument("--query", type=Path, default=Path("configs/queries/product_quality.yaml"))
    parser.add_argument("--settings", type=Path, default=Path("configs/settings.yaml"))
    parser.add_argument("--provider", choices=("mock", "jev"), default="mock")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache", choices=("on", "off"), default="off")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-calls",
        type=int,
        help="Override the retry-inclusive provider request-attempt budget",
    )
    return parser


def build_optimizer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SemanticFlow Milestone 2 optimizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, needs_provider: bool = False) -> None:
        command.add_argument(
            "--query", type=Path, default=Path("configs/queries/product_quality.yaml")
        )
        command.add_argument("--settings", type=Path, default=Path("configs/settings.yaml"))
        command.add_argument(
            "--optimizer-config", type=Path, default=Path("configs/optimizer.yaml")
        )
        if needs_provider:
            command.add_argument("--provider", choices=("mock", "jev"), default="mock")
            command.add_argument(
                "--authorize-paid",
                action="store_true",
                help="Required acknowledgement before a command may contact Jev",
            )

    profile = subparsers.add_parser("profile", help="Profile all predicates independently")
    common(profile, needs_provider=True)
    profile.add_argument("--experiment-dir", type=Path, required=True)

    plan = subparsers.add_parser("plan", help="Generate and rank candidate orders")
    common(plan)
    plan.add_argument("--experiment-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run fixed or optimized held-out execution")
    common(run, needs_provider=True)
    run.add_argument("--experiment-dir", type=Path, required=True)
    run.add_argument("--strategy", choices=("fixed", "optimized"), required=True)

    compare = subparsers.add_parser("compare", help="Compare completed held-out runs")
    compare.add_argument("--experiment-dir", type=Path, required=True)

    experiment = subparsers.add_parser(
        "experiment", help="Run profiling, planning, both executions, and comparison"
    )
    common(experiment, needs_provider=True)
    experiment.add_argument("--output-dir", type=Path, default=Path("results/milestone_2"))
    experiment.add_argument("--experiment-id")

    replay = subparsers.add_parser(
        "replay", help="Offline counterfactual replay of predicate orders over stored decisions"
    )
    replay.add_argument(
        "--query", type=Path, default=Path("configs/queries/product_quality.yaml")
    )
    replay.add_argument(
        "--decisions",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Stored decisions.jsonl; repeat for several sources",
    )
    replay.add_argument("--split-manifest", type=Path, required=True)
    replay.add_argument("--split-part", choices=("profiling", "evaluation"), required=True)
    replay.add_argument("--model", required=True, help="Resolved model version to match")
    replay.add_argument(
        "--order",
        action="append",
        required=True,
        help="Comma-separated predicate order; give two to compare orders",
    )
    replay.add_argument(
        "--source-priority", help="Comma-separated source labels (default: load order)"
    )
    replay.add_argument(
        "--conflict-mode", choices=("prefer_priority", "treat_as_unknown"),
        default="prefer_priority",
    )
    replay.add_argument("--settings", type=Path, help="Settings YAML supplying pricing")
    replay.add_argument("--output", type=Path, help="Write the JSON report to this new file")

    analyze = subparsers.add_parser(
        "analyze", help="Offline Milestone 3 analysis over stored Jev artifacts"
    )
    analyze.add_argument(
        "--query", type=Path, default=Path("configs/queries/product_quality.yaml")
    )
    analyze.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_3.yaml")
    )
    analyze.add_argument("--experiment-dir", type=Path, required=True)
    analyze.add_argument("--historical-baseline-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument(
        "--extra-decisions", action="append", default=[], metavar="LABEL=PATH",
        help="Additional stored decisions used only for variability analysis",
    )
    analyze.add_argument("--cv-folds", type=int, default=5)

    plan_b = subparsers.add_parser(
        "benchmark-plan", help="Select an order for every strategy from a stored profile"
    )
    plan_b.add_argument("--query", type=Path, required=True)
    plan_b.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_3.yaml")
    )
    plan_b.add_argument("--experiment-dir", type=Path, required=True)
    plan_b.add_argument("--output", type=Path, required=True)
    plan_b.add_argument("--prior-cost-per-decision", type=float, required=True)
    plan_b.add_argument("--prior-cost-source", required=True)

    schedule_b = subparsers.add_parser(
        "benchmark-schedule", help="Build a counterbalanced ABBA execution schedule"
    )
    schedule_b.add_argument("--strategies", type=Path, required=True)
    schedule_b.add_argument("--output", type=Path, required=True)
    schedule_b.add_argument("--orders", choices=("selected", "all"), default="selected")
    schedule_b.add_argument("--abba-blocks", type=int, default=1)
    schedule_b.add_argument("--seed", type=int, default=5051)

    preflight_b = subparsers.add_parser(
        "benchmark-preflight", help="Offline cost and attempt bounds for a schedule"
    )
    preflight_b.add_argument("--strategies", type=Path, required=True)
    preflight_b.add_argument("--schedule", type=Path, required=True)
    preflight_b.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_3.yaml")
    )
    preflight_b.add_argument("--cost-per-decision", type=float, required=True)
    preflight_b.add_argument("--output", type=Path)

    session_b = subparsers.add_parser(
        "benchmark-session",
        help="Execute a schedule of predicate orders in one contemporaneous session",
    )
    common(session_b, needs_provider=True)
    session_b.add_argument("--experiment-dir", type=Path, required=True)
    session_b.add_argument("--schedule", type=Path, required=True)
    session_b.add_argument("--session-dir", type=Path, required=True)
    session_b.add_argument("--max-session-attempts", type=int)

    analyze_b = subparsers.add_parser(
        "benchmark-analyze", help="Paired statistics for completed benchmark sessions"
    )
    analyze_b.add_argument("--workload", required=True)
    analyze_b.add_argument("--strategies", type=Path, required=True)
    analyze_b.add_argument("--session-dir", action="append", type=Path, required=True)
    analyze_b.add_argument("--output", type=Path, required=True)
    analyze_b.add_argument("--iterations", type=int, default=2000)
    analyze_b.add_argument("--seed", type=int, default=6067)

    compare_b = subparsers.add_parser(
        "benchmark-compare", help="Pool strategy statistics across workload analysis files"
    )
    compare_b.add_argument("--analysis", action="append", type=Path, required=True)
    compare_b.add_argument("--output", type=Path, required=True)

    risk_plan = subparsers.add_parser(
        "risk-plan",
        help="Select a plan from stored profiling evidence using the risk-aware policy",
    )
    risk_plan.add_argument(
        "--query", type=Path, default=Path("configs/queries/product_quality.yaml")
    )
    risk_plan.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_4.yaml")
    )
    risk_plan.add_argument("--experiment-dir", type=Path, required=True)
    risk_plan.add_argument("--output", type=Path)
    risk_plan.add_argument(
        "--statistics-state", choices=("available", "none"), default="available"
    )
    risk_plan.add_argument("--first-checkpoint-cost", type=float)
    risk_plan.add_argument("--savings-upper-bound", type=float)
    risk_plan.add_argument(
        "--robust",
        action="store_true",
        help="Use risk_aware_robust (both estimators) instead of risk_aware",
    )

    risk_bench = subparsers.add_parser(
        "risk-benchmark",
        help="Offline Milestone 4 strategy comparison over stored Milestone 3 traces",
    )
    risk_bench.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_4.yaml")
    )
    risk_bench.add_argument("--output-dir", type=Path, required=True)
    risk_bench.add_argument("--iterations", type=int)
    risk_bench.add_argument("--seed", type=int)
    risk_bench.add_argument("--skip-adaptive", action="store_true")

    joint_bench = subparsers.add_parser(
        "joint-risk-benchmark",
        help="Offline joint adaptive-profiling and risk-selection sequential replay",
    )
    joint_bench.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_4.yaml")
    )
    joint_bench.add_argument("--output-dir", type=Path, required=True)
    joint_bench.add_argument("--joint-iterations", type=int)
    joint_bench.add_argument("--voi-scenarios", type=int)

    voi_validation = subparsers.add_parser(
        "voi-validation",
        help="Offline VOI calibration and sensitivity analysis over stored Milestone 4 data",
    )
    voi_validation.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_4.yaml")
    )
    voi_validation.add_argument("--output-dir", type=Path, required=True)
    voi_validation.add_argument(
        "--canonical-dir",
        type=Path,
        default=Path("results/milestone_4/joint_adaptive_risk_20261106_2000"),
    )

    calibrate = subparsers.add_parser(
        "calibrate-m45",
        help="Milestone 4.5 profiling-only calibration, tuning, freeze, and retrospective",
    )
    calibrate.add_argument(
        "--optimizer-config", type=Path, default=Path("configs/optimizer.milestone_4.yaml")
    )
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--folds", type=int, default=4)
    calibrate.add_argument("--fold-seed", type=int, default=7073)
    calibrate.add_argument("--predictive-iterations", type=int, default=40)
    calibrate.add_argument(
        "--skip-retrospective",
        action="store_true",
        help="Stop after freezing the configuration; do not score held-out live traces",
    )
    return parser


def _labelled_paths(values: list[str]) -> list[tuple[str, Path]]:
    pairs = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        pairs.append((label, Path(path)))
    return pairs


def _run_replay(args: argparse.Namespace) -> int:
    import json

    from semanticflow.optimizer.evidence import DecisionEvidence
    from semanticflow.optimizer.models import SplitManifest
    from semanticflow.optimizer.planner import predicate_hash
    from semanticflow.optimizer.replay import (
        ReplayPricing,
        ReplayTarget,
        ResolutionPolicy,
        compare_orders,
        replay_order,
    )
    from semanticflow.storage.results import ResultStore

    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    query = load_query(args.query)
    evidence = DecisionEvidence()
    sources = _labelled_paths(args.decisions)
    for label, path in sources:
        evidence.load_jsonl(path, label)
    split = SplitManifest.model_validate_json(args.split_manifest.read_text(encoding="utf-8"))
    review_ids = split.profiling_ids if args.split_part == "profiling" else split.evaluation_ids
    priority = (
        [item.strip() for item in args.source_priority.split(",")]
        if args.source_priority
        else [label for label, _ in sources]
    )
    policy = ResolutionPolicy(
        name="cli", source_priority=priority, conflict_mode=args.conflict_mode
    )
    target = ReplayTarget(
        model_identifier=args.model,
        threshold=query.threshold,
        question_hashes={
            item.id: predicate_hash(item.id, item.instructions) for item in query.predicates
        },
    )
    pricing = ReplayPricing()
    if args.settings is not None:
        rates = load_settings(args.settings).get("pricing", {})
        pricing = ReplayPricing(
            input_per_million_usd=rates.get("input_per_million_usd"),
            output_per_million_usd=rates.get("output_per_million_usd"),
        )
    orders = [[item.strip() for item in value.split(",")] for value in args.order]
    if len(orders) == 1:
        report = replay_order(evidence, target, review_ids, orders[0], policy, pricing=pricing)
        print(f"Order: {' -> '.join(orders[0])}")
        print(f"Observability: {report['observability']}")
        print(f"Logical evaluations: {report['logical_evaluations']}")
    elif len(orders) == 2:
        report = compare_orders(
            evidence, target, review_ids, orders[0], orders[1], policy, pricing=pricing
        )
        print(f"Evaluation difference: {report['evaluation_difference_first_minus_second']}")
    else:
        raise ValueError("replay accepts one order, or two orders to compare")
    print("Evidence: offline replay estimate over stored decisions; not a causal measurement")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ResultStore.write_json(args.output, report)
        print(f"Report: {args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    from semanticflow.optimizer.analysis import AnalysisInputs, run_offline_analysis

    optimizer = load_optimizer_settings(args.optimizer_config)
    inputs = AnalysisInputs(
        experiment_dir=args.experiment_dir,
        historical_baseline_dir=args.historical_baseline_dir,
        query=load_query(args.query),
        optimizer=optimizer,
        extra_sources=[
            (label, path, "additional stored decisions")
            for label, path in _labelled_paths(args.extra_decisions)
        ],
        cross_validation_folds=args.cv_folds,
        config_paths={
            "query": str(args.query),
            "optimizer_config": str(args.optimizer_config),
        },
    )
    result = run_offline_analysis(inputs, args.output_dir)
    for item in result["counterfactual"]["decomposition"]:
        effect = item["replay_estimated_ordering_effect_evaluations"]
        print(
            f"Replay-estimated ordering effect ({item['policy']}): "
            f"{effect['lower']}..{effect['upper']} calls (offline estimate, not causal)"
        )
    print(f"Artifacts: {args.output_dir}")
    return 0


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")


def _load_json_object(path: Path) -> dict:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _print_risk_decision(order: list[str], explanation: dict) -> None:
    print(f"Decision: {explanation['decision']}")
    print(f"Reason: {explanation['reason']}")
    print(f"Selected order: {' -> '.join(order)}")
    print(
        "Bootstrap P(plan cheaper) is a resampling diagnostic, "
        "not a calibrated production probability."
    )


def _run_risk_plan(args: argparse.Namespace) -> int:
    from semanticflow.optimizer.models import ProfilingResult, SplitManifest
    from semanticflow.optimizer.planner import CostBasedPlanner
    from semanticflow.optimizer.risk import select_risk_aware_plan
    from semanticflow.optimizer.workflow import (
        _load_json,
        _profiling_report_path,
        load_profile_matrix,
    )
    from semanticflow.storage.results import ResultStore

    if args.output is not None:
        _refuse_existing(args.output)
    optimizer = load_optimizer_settings(args.optimizer_config)
    policy = "risk_aware_robust" if args.robust else "risk_aware"
    if optimizer.planner.selection_policy != policy:
        optimizer = optimizer.model_copy(
            update={
                "planner": optimizer.planner.model_copy(update={"selection_policy": policy})
            }
        )
    query = load_query(args.query)
    if args.statistics_state == "none":
        original = [item.id for item in query.predicates]
        order, explanation = select_risk_aware_plan(
            None,
            original_order=original,
            objective=optimizer.planner.objective,
            evaluation_record_count=optimizer.split.evaluation_size,
            question_hashes={},
            bootstrap=optimizer.bootstrap,
            correlation=optimizer.correlation,
            risk=optimizer.risk,
            economics=optimizer.economics,
            input_price_per_million=None,
            output_price_per_million=None,
            tie_relative_tolerance=optimizer.planner.tie_relative_tolerance,
            primary_estimator=optimizer.planner.cardinality_estimator,
            robust=args.robust,
            statistics_state="none",
            first_checkpoint_decisions=optimizer.adaptive.checkpoints[0] * len(original),
            first_checkpoint_cost=args.first_checkpoint_cost,
            savings_upper_bound_per_execution=args.savings_upper_bound,
        )
        _print_risk_decision(order, explanation)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            ResultStore.write_json(
                args.output, {"selected_order": order, "explanation": explanation}
            )
        return 0
    profile = ProfilingResult.model_validate(
        _load_json(_profiling_report_path(args.experiment_dir))
    )
    split = SplitManifest.model_validate(
        _load_json(args.experiment_dir / "split_manifest.json")
    )
    plan = CostBasedPlanner(optimizer).plan(
        query,
        profile,
        split,
        matrix=load_profile_matrix(args.experiment_dir, profile, split),
    )
    explanation = plan.selection_explanation or {}
    _print_risk_decision(plan.selected_predicate_order, explanation)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ResultStore.write_json(args.output, plan.artifact_dict())
    return 0


def _run_risk_benchmark(args: argparse.Namespace) -> int:
    from semanticflow.optimizer.risk_benchmark import run_offline_benchmark, write_text_summary

    _refuse_existing(args.output_dir)
    optimizer = load_optimizer_settings(args.optimizer_config)
    updates: dict[str, object] = {}
    if args.iterations is not None:
        updates["bootstrap"] = optimizer.bootstrap.model_copy(
            update={"iterations": args.iterations}
        )
    if args.seed is not None:
        bootstrap = updates.get("bootstrap", optimizer.bootstrap)
        updates["bootstrap"] = bootstrap.model_copy(update={"seed": args.seed})
    if updates:
        optimizer = optimizer.model_copy(update=updates)
    report = run_offline_benchmark(
        optimizer,
        args.output_dir,
        include_adaptive=not args.skip_adaptive,
        config_path=str(args.optimizer_config),
    )
    write_text_summary(report, args.output_dir / "summary.md")
    print(f"Workloads: {len(report['workloads'])}")
    print(f"Live Jev calls: {report['live_jev_calls']}")
    print(f"Artifacts: {args.output_dir}")
    return 0


def _run_joint_risk_benchmark(args: argparse.Namespace) -> int:
    from semanticflow.optimizer.joint_benchmark import run_joint_benchmark

    _refuse_existing(args.output_dir)
    optimizer = load_optimizer_settings(args.optimizer_config)
    updates = {}
    if args.joint_iterations is not None:
        updates["bootstrap_iterations"] = args.joint_iterations
    if args.voi_scenarios is not None:
        updates["voi_scenarios"] = args.voi_scenarios
    if updates:
        optimizer = optimizer.model_copy(
            update={"joint": optimizer.joint.model_copy(update=updates)}
        )
    report = run_joint_benchmark(
        optimizer,
        args.output_dir,
        config_path=str(args.optimizer_config),
    )
    print(f"Workloads: {len(report['workloads'])}")
    print(f"Live Jev calls: {report['live_jev_calls']}")
    print(f"Artifacts: {args.output_dir}")
    return 0


def _run_calibrate_m45(args: argparse.Namespace) -> int:
    from semanticflow.optimizer.m45_calibration import run_milestone_45

    _refuse_existing(args.output_dir)
    optimizer = load_optimizer_settings(args.optimizer_config)
    report = run_milestone_45(
        optimizer,
        args.output_dir,
        folds=args.folds,
        fold_seed=args.fold_seed,
        predictive_iterations=args.predictive_iterations,
        skip_retrospective=args.skip_retrospective,
        config_path=str(args.optimizer_config),
    )
    print(f"Baseline reproduction: {report['baseline_reproduction']}")
    print(f"Recommended default: {report['tuning_selection']['recommended_default']}")
    print(f"Live Jev calls: {report['live_jev_calls']}")
    print(f"Artifacts: {args.output_dir}")
    return 0


def _run_voi_validation(args: argparse.Namespace) -> int:
    from semanticflow.optimizer.voi_validation import run_voi_validation

    _refuse_existing(args.output_dir)
    optimizer = load_optimizer_settings(args.optimizer_config)
    report = run_voi_validation(
        optimizer,
        args.output_dir,
        canonical_dir=args.canonical_dir,
        config_path=str(args.optimizer_config),
    )
    print(f"Workloads: {len(report['workloads'])}")
    print(f"Live Jev calls: {report['live_jev_calls']}")
    print(f"Artifacts: {args.output_dir}")
    return 0


def _run_benchmark_offline(args: argparse.Namespace) -> int:
    import json

    from semanticflow.optimizer.benchmark import (
        build_schedule,
        plan_strategies,
        schedule_preflight,
    )
    from semanticflow.optimizer.benchmark_stats import analyze_workload, compare_workloads
    from semanticflow.optimizer.models import SplitManifest, load_optimizer_settings
    from semanticflow.storage.results import ResultStore

    if args.command == "benchmark-plan":
        _refuse_existing(args.output)
        payload = plan_strategies(
            query=load_query(args.query),
            optimizer=load_optimizer_settings(args.optimizer_config),
            profile_dir=args.experiment_dir,
            prior_cost_per_decision=args.prior_cost_per_decision,
            prior_cost_source=args.prior_cost_source,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ResultStore.write_json(args.output, payload)
        print(f"Distinct selected orders: {len(payload['distinct_selected_orders'])}")
        print(f"Strategies: {args.output}")
        return 0
    if args.command == "benchmark-schedule":
        _refuse_existing(args.output)
        payload = build_schedule(
            _load_json_object(args.strategies),
            orders=args.orders,
            abba_blocks=args.abba_blocks,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ResultStore.write_json(args.output, payload)
        print(f"Scheduled runs: {len(payload['runs'])}")
        print(f"Comparator: {' -> '.join(payload['comparator_order'])}")
        print(f"Schedule: {args.output}")
        return 0
    if args.command == "benchmark-preflight":
        if args.output is not None:
            _refuse_existing(args.output)
        payload = schedule_preflight(
            _load_json_object(args.strategies),
            _load_json_object(args.schedule),
            load_optimizer_settings(args.optimizer_config),
            cost_per_decision=args.cost_per_decision,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            ResultStore.write_json(args.output, payload)
        return 0
    if args.command == "benchmark-analyze":
        _refuse_existing(args.output)
        strategies = _load_json_object(args.strategies)
        split = SplitManifest.model_validate(
            _load_json_object(Path(strategies["profile_dir"]) / "split_manifest.json")
        )
        payload = analyze_workload(
            args.workload,
            strategies,
            args.session_dir,
            evaluation_ids=split.evaluation_ids,
            iterations=args.iterations,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ResultStore.write_json(args.output, payload)
        print(f"Observed cheapest: {payload['observed_cheapest_order']}")
        print(f"Analysis: {args.output}")
        return 0
    _refuse_existing(args.output)
    payload = compare_workloads([_load_json_object(path) for path in args.analysis])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ResultStore.write_json(args.output, payload)
    print(f"Workloads: {payload['workload_count']}")
    print(f"Comparison: {args.output}")
    return 0


def _provider(
    name: str,
    settings: dict[str, object],
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> MockProvider | JevProvider:
    providers = settings.get("providers", {})
    provider_config = providers.get(name, {}) if isinstance(providers, dict) else {}
    if not isinstance(provider_config, dict):
        raise ValueError(f"Invalid provider settings for {name}")
    if name == "mock":
        return MockProvider(
            model=str(provider_config.get("model", "mock-deterministic-v1")),
            input_tokens_per_evaluation=input_tokens,
            output_tokens_per_evaluation=output_tokens,
        )
    return JevProvider.from_env(
        model=str(provider_config.get("model", "jev-latest")),
        api_url=str(provider_config.get("api_url", "https://api.typesafe.ai/v1/systemone")),
        timeout_seconds=float(provider_config.get("timeout_seconds", 30)),
        max_retries=int(provider_config.get("max_retries", 2)),
        backoff_seconds=float(provider_config.get("backoff_seconds", 0.5)),
    )


async def _run_optimizer(args: argparse.Namespace) -> int:
    if args.command == "compare":
        comparison = compare_experiment(args.experiment_dir)
        print(f"Comparison: {args.experiment_dir / 'comparison.json'}")
        print(f"Equivalent outputs: {comparison['output_equivalence']['equivalent']}")
        return 0

    settings = load_settings(args.settings)
    optimizer = load_optimizer_settings(args.optimizer_config)
    query = load_query(args.query)
    if args.command == "plan":
        plan = plan_experiment(query=query, optimizer=optimizer, experiment_dir=args.experiment_dir)
        print(f"Original order: {' -> '.join(plan.original_predicate_order)}")
        print(f"Selected order: {' -> '.join(plan.selected_predicate_order)}")
        return 0

    if args.provider == "jev" and not args.authorize_paid:
        raise ValueError(
            "Jev execution requires explicit --authorize-paid after reviewing the request budget"
        )
    provider = _provider(
        args.provider,
        settings,
        input_tokens=optimizer.mock_usage.input_tokens_per_evaluation,
        output_tokens=optimizer.mock_usage.output_tokens_per_evaluation,
    )
    try:
        if args.command == "profile":
            profile, _ = await profile_experiment(
                query=query,
                settings=settings,
                optimizer=optimizer,
                provider=provider,
                experiment_dir=args.experiment_dir,
                research_layout=True,
            )
            print(f"Completed profiling decisions: {profile.completed_decisions}")
            print(f"Artifacts: {args.experiment_dir}")
            return 0 if profile.complete else 2
        if args.command == "benchmark-session":
            from semanticflow.optimizer.benchmark import run_session

            manifest = await run_session(
                query=query,
                settings=settings,
                optimizer=optimizer,
                provider=provider,
                profile_dir=args.experiment_dir,
                schedule=_load_json_object(args.schedule),
                schedule_path=args.schedule,
                session_dir=args.session_dir,
                max_session_attempts=args.max_session_attempts,
            )
            print(f"Session status: {manifest['status']}")
            print(f"Attempts used: {manifest['attempts_used']}")
            print(f"Artifacts: {args.session_dir}")
            return 0 if manifest["status"] == "complete" else 2
        if args.command == "run":
            outcome = await execute_experiment_strategy(
                strategy=args.strategy,
                query=query,
                settings=settings,
                optimizer=optimizer,
                provider=provider,
                experiment_dir=args.experiment_dir,
            )
            print(f"Completion: {outcome.report['completion_status']}")
            print(f"Artifacts: {outcome.run_dir}")
            return 0 if outcome.report["completion_status"] == "complete" else 2
        experiment_id = args.experiment_id or (
            f"{query.query_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        experiment_dir, comparison = await run_offline_experiment(
            query_path=args.query,
            settings=settings,
            optimizer=optimizer,
            provider=provider,
            output_root=args.output_dir,
            experiment_id=experiment_id,
        )
        print(f"Original order: {' -> '.join(comparison['original_predicate_order'])}")
        print(f"Selected order: {' -> '.join(comparison['selected_predicate_order'])}")
        print(f"Equivalent outputs: {comparison['output_equivalence']['equivalent']}")
        print(f"Artifacts: {experiment_dir}")
        return 0
    finally:
        await provider.aclose()


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    provider_config = settings.get("providers", {}).get(args.provider, {})
    if args.provider == "mock":
        provider = MockProvider(model=provider_config.get("model", "mock-deterministic-v1"))
    else:
        provider = JevProvider.from_env(
            model=provider_config.get("model", "jev-latest"),
            api_url=provider_config.get("api_url", "https://api.typesafe.ai/v1/systemone"),
            timeout_seconds=float(provider_config.get("timeout_seconds", 30)),
            max_retries=int(provider_config.get("max_retries", 2)),
            backoff_seconds=float(provider_config.get("backoff_seconds", 0.5)),
        )
    print(f"Provider: {provider.name}; workload: {args.query}")
    try:
        outcome = await run_query(
            query_path=args.query,
            settings=settings,
            provider=provider,
            output_dir=args.output_dir,
            cache_enabled=args.cache == "on",
            labels_path=args.labels,
            limit_override=args.limit,
            max_calls_override=args.max_calls,
        )
    finally:
        await provider.aclose()
    report = outcome.report
    for stage in report["predicates"]:
        print(
            f"Predicate {stage['predicate_id']}: {stage['input_record_count']} input, "
            f"{stage['passing_record_count']} surviving"
        )
    print(f"Actual provider calls: {report['actual_provider_calls']}")
    print(f"Final output count: {report['final_record_count']}")
    print(f"Total execution time: {report['wall_time_seconds']:.3f}s")
    print(f"Results directory: {outcome.run_dir}")
    return 0 if report["completion_status"] == "complete" else 2


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] in {"replay", "analyze"}:
            args = build_optimizer_parser().parse_args(arguments)
            return _run_replay(args) if args.command == "replay" else _run_analyze(args)
        if arguments and arguments[0] == "risk-plan":
            return _run_risk_plan(build_optimizer_parser().parse_args(arguments))
        if arguments and arguments[0] == "risk-benchmark":
            return _run_risk_benchmark(build_optimizer_parser().parse_args(arguments))
        if arguments and arguments[0] == "joint-risk-benchmark":
            return _run_joint_risk_benchmark(build_optimizer_parser().parse_args(arguments))
        if arguments and arguments[0] == "voi-validation":
            return _run_voi_validation(build_optimizer_parser().parse_args(arguments))
        if arguments and arguments[0] == "calibrate-m45":
            return _run_calibrate_m45(build_optimizer_parser().parse_args(arguments))
        if arguments and arguments[0] in {
            "benchmark-plan",
            "benchmark-schedule",
            "benchmark-preflight",
            "benchmark-analyze",
            "benchmark-compare",
        }:
            return _run_benchmark_offline(build_optimizer_parser().parse_args(arguments))
        if arguments and arguments[0] in {
            "profile",
            "plan",
            "run",
            "compare",
            "experiment",
            "benchmark-session",
        }:
            return asyncio.run(_run_optimizer(build_optimizer_parser().parse_args(arguments)))
        return asyncio.run(_run(build_parser().parse_args(arguments)))
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

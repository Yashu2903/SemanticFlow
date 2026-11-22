# SemanticFlow Milestone 5.2 System Design

## Problem and integration boundary

M5.2 adds uncertainty-aware quality qualification and active profiling around the unchanged M5.1 physical plan representation and `existing_adaptive` baseline. It does not change executor semantics, frozen M5.1 plans, the M5.1 point gate, or provider integration.

## Components

- `m52_evidence.py`: immutable paired evidence, discovery/certification isolation, confusion counts, and qualification states.
- `m52_quality_uncertainty.py`: Wilson diagnostics plus exact one-sided sequential bounds with family-wise spending.
- `m52_cost_model.py`: selection-partition-locked predictions with input measurement provenance.
- `m52_replay.py`: reveal-on-acquisition historical matrix and exact recorded charging.
- `m52_active_profiler.py`: random, positive-enriched, threshold, disagreement, cost-aware, and hybrid policies.
- `m52_controller.py`: traceable state machine, budgets, hard quality constraint, EVSI approximation, and stopping.
- `m52_planner.py`: strict, risk-budget, cost-only, M5.1-gate, and existing-adaptive modes.

## Evidence lifecycle

Discovery actions may screen a record using acquired separate inference and then acquire a candidate measurement. Certification actions use an independently frozen representative sequence. A record cannot occur in both tracks. Diagnostic observations never enter ordinary certification counts. Every request ID is charged once and remains invisible until acquired.

## State transitions

The controller records INITIALIZE, EVALUATE_PRIOR, SELECT_ACTION, ACQUIRE_EVIDENCE, UPDATE_QUALITY_ESTIMATES, CHECK_QUALIFICATION, CHECK_ECONOMIC_VALUE, CONTINUE, SELECT_PLAN/FALLBACK, and TERMINATE transitions. Stop reasons distinguish certification, rejection, insufficient support, budget exhaustion, unsupported evidence, and uneconomic information.

## Expected value of information

Semantic quality is a hard constraint. No dollar penalty is assigned to semantic errors. The tractable EVSI approximation multiplies the remaining optimistic reuse-horizon saving by an evidence-gap-based probability of changing the decision, then subtracts predicted acquisition cost. It uses only frozen selection statistics and currently visible counts. It is a stopping heuristic, not a Bayesian optimal-design solution.

## Failure handling and auditability

Missing historical arms are UNSUPPORTED. Provider failures are distinct. Prediction, action, evidence, plan, configuration, and policy fingerprints are persisted. The offline runner contains no provider initialization path.

# SemanticFlow Milestone 5.2.1 System Design

## Purpose

M5.2.1 adds a qualification feasibility gate ahead of the unchanged M5.2 controller. It separates mathematical possibility, statistical power, economic feasibility, and exploratory risk estimation.

## Components and lifecycle

`m521_error_budget.py` versions the preserved legacy and prospective nonredundant policies. `m521_feasibility.py` computes exact best-case attainability from current confusion counts. `m521_statistical_power.py` computes positive-support and certification probabilities. `m521_economic_feasibility.py` enforces optimistic-benefit and expected-value stops. `m521_controller.py` checks feasibility before acquisition and through the M5.2 continuation guard.

Evidence provenance is validated first. The controller then checks qualification, exact attainability, measurement coverage, power, remaining expenditure, and economic upside. Hard impossibility cannot be overridden by an EVSI heuristic. Discovery evidence remains isolated from representative certification.

## Policy types

Formal certification retains frequentist assumptions. Exploratory Jeffreys beta-binomial risk estimates are explicitly `ESTIMATED_RISK_ONLY`; they never override strict deployment constraints. `existing_adaptive`, M5.1, and legacy M5.2 remain unchanged baselines.

## Failure handling

Distinct reasons cover record, call, token, positive-support, structural, and economic infeasibility. All certificates and policy configurations are fingerprinted. Offline execution never imports or initializes a provider.

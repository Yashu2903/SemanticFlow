# SemanticFlow Milestone 5.2 Small Live Pilot Proposal

## Status

Proposal only. No live execution is authorized.

## Scope

- Three workloads: Banking77 card/unauthorized/refund, Beauty failure/usability, and SMS prize/urgency/contact.
- Per workload: 60 representative paired records, up to 20 paid discovery screens, and up to 20 corresponding candidate measurements.
- Screening inference is explicitly counted; no free likely-positive or threshold ranking is assumed.

## Budget

- Expected physical requests: 576.
- Physical request ceiling: 1200.
- Expected named-question evaluations: 980.
- Expected input tokens: 204,446.
- Expected cost proxy: $0.008587.
- Conservative input tokens: 532,412.
- Conservative cost proxy: $0.022361.
- Retry-inclusive attempt ceiling: 3600 with at most two retries per request.

## Independent evaluation and positive feasibility

The pilot is development profiling, not the final independent test. Before scheduling, prevalence estimates must show whether representative certification is feasible; 400 records is not presumed sufficient. A later frozen evaluation needs new records, at least one new domain, and independently double-annotated representative and diagnostic strata.

## Authorization checkpoint

Freeze the policy hash, qualification and sampling rules, candidate catalog, cost estimator, budgets, schedule, model, and dataset fingerprints. Validate them fail-closed before provider initialization. Historical authorization cannot be reused. Execution requires the exact new phrase `AUTHORIZE M5.2 SMALL LIVE PILOT <FROZEN_SCOPE_FINGERPRINT>`. Attempts must be reserved atomically, interruption must not auto-resume, and no automatic paid continuation is allowed.

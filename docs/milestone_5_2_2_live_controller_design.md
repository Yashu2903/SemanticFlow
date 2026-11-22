# Milestone 5.2.2 Live Controller Design

## Objective boundary

`PRODUCTION_QUALIFICATION` retains M5.2.1 feasibility, confidence, and economic guards.
`RESEARCH_INSTRUMENTATION` permits only actions in a frozen universe and sets both formal
certification and production deployment permissions to false. There is no generic disable-safety
flag.

## Components

`PilotAProtocol` -> `AuthorizedActionUniverse` -> `LiveProfilingController` ->
`LiveAcquisitionAdapter` -> existing `InferenceProvider`/`JevProvider` interface. The provider is
created only after authorization validation. `BudgetLedger` reserves one unique identity and three
attempt slots before dispatch. `AppendOnlyEventLog` hash-chains every decision and receipt.

```text
manifest -> validate scope -> initialize provider -> select allowed action
         -> reserve -> dispatch -> persist receipt -> charge -> expose evidence
         -> select next action or terminate
```

```text
CREATED -> VALIDATED -> AUTHORIZED -> EXECUTING -> INSTRUMENTATION_COMPLETE
                                           |-> INTERRUPTED
                                           |-> RECONCILIATION_REQUIRED
                                           |-> FAILED_CLOSED
```

Representative and discovery receipts retain different track identifiers. A discovery screen may
enable a follow-up only after its durable response is visible. Unseen outcomes are not available to
the policy.

## Recovery

An undispatched reservation can be explicitly released when the durable log proves dispatch never
started. A persisted receipt can be accepted without reissuing its request. Dispatch without a
receipt is ambiguous, consumes attempt exposure, and blocks further execution pending external
reconciliation.

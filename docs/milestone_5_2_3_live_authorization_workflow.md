# M5.2.3 live authorization workflow

M5.2.3 uses an experiment-scoped activation manifest. An explicit user instruction to
**perform the live authorization** for the named experiment is the approval event that permits
the runner to create the active manifest from the frozen proposal. The user does not edit JSON,
provide a countersignature, or repeat a protocol-specific phrase.

Activation is a local, offline operation. It validates the proposal against the current frozen
protocol and action universe, records the exact experiment identifier and authorization-scope
fingerprint, and writes a separate active manifest. It does not initialize the provider or make
an API request. Authorization by itself therefore cannot incur paid usage.

For Pilot A, the activation command is:

```powershell
.\.venv\Scripts\python.exe -m experiments.milestone_5_2_3.run_pilot_a authorize `
  --manifest results/milestone_5_2_3/pilot_a_preflight/authorization_manifest_proposal.json `
  --output results/milestone_5_2_3/pilot_a_preflight/authorization_manifest_active.json `
  --experiment semanticflow-m5.2.3-live-pilot-a-instrumentation
```

Execution remains a distinct command. It is permitted only when the user explicitly requests
execution as well as authorization, or makes a later explicit execution request. Before provider
initialization, the runner revalidates the active manifest, protocol fingerprint, action-universe
fingerprint, acquisition-policy fingerprint, model, output directory, 360-request ceiling,
1,080-attempt ceiling, and two-retry limit.

An active manifest is invalid for a different experiment, a different output directory, or any
materially changed protocol or action universe. It cannot be reused as historical authorization.
The unsigned proposal is preserved, and activation refuses to overwrite it.

Sessions in `FAILED_CLOSED` are terminal and cannot be executed or resumed. Provider responses
and usage metadata are durably recorded before semantic response validation; a restart can
validate an already-persisted response without resending the paid request. Ambiguous dispatches
remain subject to conservative accounting and reconciliation and are never retried implicitly.

This workflow changes only how explicit approval is materialized. It does not relax the frozen
scientific protocol, provider accounting, request ceilings, evidence isolation, or the prohibition
on treating research instrumentation as formal quality certification.

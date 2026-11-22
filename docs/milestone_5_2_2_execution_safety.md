# Milestone 5.2.2 Execution Safety

Live execution requires an exact countersigned manifest whose scope binds protocol, action universe,
policy, model, output directory, unique-request ceiling, and attempt ceiling. Historical
authorizations and mock authorization objects are rejected.

The event log is append-only and hash chained. Request identity equals the frozen action identity;
retries never create a new logical action. Capacity is reserved atomically. Actual attempts are
charged even when responses fail. The Pilot-A-only `PilotAJevProvider` retries definitive 429/5xx
responses but never retries an ambiguous transport failure; it enters `RECONCILIATION_REQUIRED`.

Provider responses must contain exactly the expected named decisions, the frozen model, bounded
finite probabilities, consistent attempt metadata, and complete usage. Receipt persistence precedes
accounting/evidence acceptance so recovery can finish without another paid request.

Concurrent reservations are protected by a ledger lock. The Pilot A runner is a single durable
event-log writer; multi-process distributed execution is outside this pilot and remains unsupported.

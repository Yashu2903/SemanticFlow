# Milestone 5.2.3 Research Conclusions

Pilot A demonstrated that scope validation, authorization, action reservation, attempt charging,
hash-chained persistence, and fail-closed termination operate on a real request. It did not answer
the scientific question about real response-adaptive profiling because no provider evidence was
accepted.

The blocking defect is an internal response-shape contract mismatch. The established Jev fused
implementation populates both a group probability and a named predicate probability for single
mode, while the Pilot A adapter requires the named predicate map to be empty. The deterministic
mock follows the latter contract, so the mismatch was not covered by preflight.

No restart or protocol repair is permitted under the current frozen run. A future attempt requires
a versioned infrastructure repair, a real-shape contract regression, preservation of usage and raw
evidence on post-response validation failure, a fresh preflight, and new scope-matched
authorization. M5.3 must not begin from this incomplete experiment.

## Post-incident repair

The required infrastructure repair is now implemented: the established single-response shape is
accepted, raw responses and usage are durably logged before semantic validation, and failed-closed
sessions cannot be reactivated. Targeted offline regression tests cover those guarantees. The
scientific conclusions above remain unchanged because no additional Jev call was made and the
failed session was not modified.

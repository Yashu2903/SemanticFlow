# Milestone 5.2.3 Research Conclusions

## Preserved failed session conclusion

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

## Repaired fresh session conclusion

The repaired Pilot A answered the instrumentation question affirmatively for this bounded run.
SemanticFlow executed a real response-adaptive Jev acquisition policy, persisted every raw response
and usage record before evidence acceptance, kept all actions inside the exact authorization
universe, respected request and retry ceilings, and terminated at the frozen objective. One of 30
real screens changed the acquisition sequence by triggering one diagnostic follow-up; 29 follow-ups
were avoided. Independent replay found that every decision used only evidence available at its
checkpoint.

Reliability findings were strong within the observed run: 251/251 requests completed on their first
HTTP attempt; all used `jev-1.14.0`; every response mapped to one request and one accepted receipt;
all token counts reconciled; representative and discovery tracks remained separate; and the final
event log and 21 derived artifacts passed hash verification. The previous `FAILED_CLOSED` session
remains a separate, unmodified incident artifact.

Descriptive representative comparison found 60/60 query agreements and no predicate threshold
crossings. This is not a quality-certification result. Only two separate-query positives were
observed across the three workloads, SMS Spam had zero, and 20 representative records per workload
are far below both historical and prospective support requirements. Separate Jev decisions are
comparison references, not human ground truth.

The main research lessons for a later, separately authorized M5.3 are to calibrate screening value
against its 11.70% share of pilot cost, add risk features for probability proximity and batch-minus-
separate shifts, model dataset/request-type token error separately, and collect enough independent
positive support to estimate quality risk. This handoff is a recommendation only; no M5.3 policy
was implemented or tuned during Pilot A.

The supported milestone outcome is `RESEARCH_INSTRUMENTATION_COMPLETE`, not `QUALITY_CERTIFIED`.

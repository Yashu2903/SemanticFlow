from __future__ import annotations

from dataclasses import dataclass

from semanticflow.dataset.loader import ReviewRecord
from semanticflow.executor.fused import FusedExecutionResult, FusedExecutor
from semanticflow.optimizer.fusion_planner import FusionPhysicalPlan


@dataclass(frozen=True, slots=True)
class HybridFallbackResult:
    primary: FusedExecutionResult
    fallback: FusedExecutionResult | None
    fallback_authorized: bool
    total_provider_attempts: int
    total_input_tokens: int
    total_output_tokens: int
    complete: bool


class HybridExecutor(FusedExecutor):
    """Mixed physical execution plus explicit, separately billed fallback."""

    async def execute_with_fallback(
        self,
        records: list[ReviewRecord],
        plan: FusionPhysicalPlan,
        *,
        run_id: str,
        maximum_http_attempts: int,
        fallback_plan: FusionPhysicalPlan | None = None,
        fallback_authorized: bool = False,
        fallback_maximum_http_attempts: int = 0,
    ) -> HybridFallbackResult:
        primary = await self.execute(
            records,
            plan,
            run_id=run_id,
            maximum_http_attempts=maximum_http_attempts,
        )
        fallback: FusedExecutionResult | None = None
        if not primary.complete and fallback_plan is not None:
            if not fallback_authorized:
                return HybridFallbackResult(
                    primary=primary,
                    fallback=None,
                    fallback_authorized=False,
                    total_provider_attempts=primary.provider_attempts,
                    total_input_tokens=primary.input_tokens,
                    total_output_tokens=primary.output_tokens,
                    complete=False,
                )
            if fallback_maximum_http_attempts <= 0:
                raise ValueError("authorized fallback requires a positive attempt budget")
            fallback = await self.execute(
                records,
                fallback_plan,
                run_id=f"{run_id}__explicit_fallback",
                maximum_http_attempts=fallback_maximum_http_attempts,
            )
        return HybridFallbackResult(
            primary=primary,
            fallback=fallback,
            fallback_authorized=fallback_authorized,
            total_provider_attempts=primary.provider_attempts
            + (fallback.provider_attempts if fallback else 0),
            total_input_tokens=primary.input_tokens + (fallback.input_tokens if fallback else 0),
            total_output_tokens=primary.output_tokens
            + (fallback.output_tokens if fallback else 0),
            complete=primary.complete or bool(fallback and fallback.complete),
        )

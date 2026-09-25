"""Choose who labels an example: potion for easy ones, a teacher LLM for hard ones."""

from __future__ import annotations

from dataclasses import dataclass

from deck.routing import Route, route_example
from deck.taxonomy import Taxonomy


@dataclass(frozen=True)
class LabelPlan:
    model: str
    score: int
    reasons: tuple[str, ...]
    decision_model: str = "deck4b"


def plan_label(
    state: str,
    taxonomy: Taxonomy,
    *,
    threshold: int = 4,
) -> LabelPlan:
    route: Route = route_example(
        state,
        taxonomy.question,
        option_count=len(taxonomy.classes),
        threshold=threshold,
    )
    return LabelPlan(model=route.model, score=route.score, reasons=route.reasons)

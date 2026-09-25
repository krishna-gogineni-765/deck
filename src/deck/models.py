"""Models in the deck framework. deck4b is the decision model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    id: str
    role: str
    runtime: str
    summary: str
    weights: str = ""


MODELS: dict[str, Model] = {
    "deck4b": Model(
        id="deck4b",
        role="decide",
        runtime="gpu-fp8",
        summary="Rank-8 LoRA decision model. Native A-P probabilities, zero generated tokens.",
        weights="krishna765/deck-4b-v1.0@81e9f5e2cc701f86b1be700dfb49f6549ad83f39",
    ),
    "potion": Model(
        id="potion",
        role="label-easy",
        runtime="cpu",
        summary="Small CPU model for easy examples and short taxonomies.",
    ),
    "teacher": Model(
        id="teacher",
        role="label-hard",
        runtime="llm",
        summary="Powerful LLM used to write class descriptions and label hard examples.",
    ),
}


def get_model(model_id: str) -> Model:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"unknown model {model_id!r}; known models: {known}") from exc

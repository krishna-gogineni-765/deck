"""Ask a teacher model to write class descriptions, then apply the ones that parse."""

from __future__ import annotations

import json
from typing import Any

from deck.taxonomy import Class, Taxonomy, description_is_weak


def description_prompt(taxonomy: Taxonomy) -> str:
    """Prompt a teacher LLM to write concise, mutually exclusive class descriptions."""
    classes = [
        {"id": item.id, "current": item.description}
        for item in taxonomy.classes
        if description_is_weak(item)
    ]
    if not classes:
        classes = [{"id": item.id, "current": item.description} for item in taxonomy.classes]
    return (
        "You are writing a zero-shot classification taxonomy. Preserve every id. "
        "Return JSON only with key `classes`. Each id maps to an object with "
        "`description`, `exclusions`, and one synthetic positive `example`. "
        "Do not quote benchmarks or real records. Make the classes mutually clear.\n\n"
        f"Taxonomy: {taxonomy.id}\n"
        f"Question: {taxonomy.question}\n"
        f"Classes: {json.dumps(classes, ensure_ascii=False)}"
    )


def apply_descriptions(taxonomy: Taxonomy, payload: Any) -> Taxonomy:
    """Apply a teacher JSON object. Unknown ids and empty drafts are ignored."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("description payload must be a JSON object")
    drafts = payload.get("classes") or payload.get("options") or {}
    if not isinstance(drafts, dict):
        raise ValueError("classes must be an object keyed by id")
    updated = taxonomy
    for item in taxonomy.classes:
        draft = drafts.get(item.id)
        if isinstance(draft, str):
            draft = {"description": draft}
        if not isinstance(draft, dict):
            continue
        description = str(draft.get("description") or draft.get("definition") or "").strip()
        exclusions = str(draft.get("exclusions") or "").strip()
        example = str(draft.get("example") or "").strip()
        if not any((description, exclusions, example)):
            continue
        updated = updated.replace(
            Class(
                id=item.id,
                description=description or item.description,
                exclusions=exclusions or item.exclusions,
                example=example or item.example,
            )
        )
    question = str(payload.get("question") or "").strip()
    if question:
        updated = Taxonomy(
            id=updated.id,
            question=question,
            version=updated.version,
            classes=updated.classes,
        )
    return updated

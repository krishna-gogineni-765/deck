"""Taxonomies: class ids, descriptions, and examples."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Class:
    id: str
    description: str = ""
    exclusions: str = ""
    example: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Taxonomy:
    id: str
    question: str
    classes: tuple[Class, ...]
    version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "question": self.question,
            "classes": [item.to_dict() for item in self.classes],
        }

    def get(self, class_id: str) -> Class:
        for item in self.classes:
            if item.id == class_id:
                return item
        raise KeyError(class_id)

    def replace(self, updated: Class) -> "Taxonomy":
        self.get(updated.id)
        return Taxonomy(
            id=self.id,
            question=self.question,
            version=self.version,
            classes=tuple(
                updated if item.id == updated.id else item for item in self.classes
            ),
        )


def normalize(value: str) -> str:
    return _WORD.sub(" ", value.lower()).strip()


def description_is_weak(item: Class) -> bool:
    """A class still needs a written description when it only restates its id."""
    description = normalize(item.description)
    name = normalize(item.id)
    return not description or description == name or len(description.split()) <= 2


def load_taxonomy(path: Path | str) -> Taxonomy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = tuple(
        Class(
            id=str(item["id"]).strip(),
            description=str(item.get("description") or "").strip(),
            exclusions=str(item.get("exclusions") or "").strip(),
            example=str(item.get("example") or "").strip(),
        )
        for item in payload["classes"]
    )
    if not classes or any(not item.id for item in classes):
        raise ValueError("a taxonomy needs classes with ids")
    if len({item.id for item in classes}) != len(classes):
        raise ValueError("class ids must be unique")
    return Taxonomy(
        id=str(payload["id"]).strip(),
        question=str(payload["question"]).strip(),
        version=str(payload.get("version") or "1"),
        classes=classes,
    )


def save_taxonomy(taxonomy: Taxonomy, path: Path | str) -> None:
    Path(path).write_text(
        json.dumps(taxonomy.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

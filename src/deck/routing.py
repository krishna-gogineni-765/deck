"""Send easy examples to potion and hard examples to the teacher."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_OR_DATE = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_RULE_CUES = re.compile(
    r"\b(?:amend(?:ed|ment)?|exception|unless|except|precedence|overrides?|"
    r"supersed(?:e|es|ed)|notwithstanding|provided that|subject to|waiver)\b",
    re.IGNORECASE,
)
_REVISION_CUES = re.compile(
    r"\b(?:however|correction|corrected|revised|updated|cancelled|reversed|later section)\b",
    re.IGNORECASE,
)
_REASONING_CUES = re.compile(
    r"\b(?:calculate|compute|deadline|business days?|time zone|required approval|"
    r"under the policy|after applying|highest[- ]ranked|determine whether)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Route:
    model: str
    score: int
    reasons: tuple[str, ...]


def route_example(
    state: str,
    question: str = "",
    *,
    option_count: int = 0,
    threshold: int = 4,
) -> Route:
    """Score how hard an example is. Easy goes to potion; hard goes to the teacher."""
    state_text = str(state)
    question_text = str(question)
    joined = f"{question_text}\n{state_text}"
    score = 0
    reasons: list[str] = []

    if len(state_text) >= 4_000:
        score += 3
        reasons.append("state>=4000_chars")
    elif len(state_text) >= 1_500:
        score += 2
        reasons.append("state>=1500_chars")

    numeric_count = len(_NUMBER_OR_DATE.findall(joined))
    if numeric_count >= 8:
        score += 2
        reasons.append("dense_dates_or_numbers")
    elif numeric_count >= 4:
        score += 1
        reasons.append("dates_or_numbers")

    rule_count = len(_RULE_CUES.findall(joined))
    if rule_count >= 2:
        score += 2
        reasons.append("rule_precedence_or_exceptions")
    elif rule_count == 1:
        score += 1
        reasons.append("single_rule_exception")

    if _REASONING_CUES.search(question_text):
        score += 2
        reasons.append("explicit_reasoning_question")
    if len(_REVISION_CUES.findall(state_text)) >= 2:
        score += 1
        reasons.append("multiple_revisions")
    if option_count >= 8:
        score += 1
        reasons.append("many_options")
    if option_count > 16:
        score += 2
        reasons.append("above_single_pass_options")

    model = "teacher" if score >= threshold else "potion"
    return Route(model=model, score=score, reasons=tuple(reasons))

from __future__ import annotations

import re

from app.schemas import QualityIssue, Requirement
from app.services.generator import is_project_activity


VAGUE = {
    "adequate", "appropriate", "efficient", "easy", "fast", "reliable",
    "robust", "secure", "user-friendly",
}
STOP_WORDS = {
    "about", "after", "before", "could", "from", "have", "shall", "should",
    "that", "their", "there", "these", "this", "those", "will", "with",
}
ACTOR = re.compile(
    r"^\s*(?:(?:the\s+)?(?:system|software|service|application|app|portal|"
    r"platform|scheduler|module|patient|patients|user|users|clinician|"
    r"clinicians|staff|administrator|administrators|scheduler|schedulers)\b)",
    re.I,
)
MODAL = re.compile(r"\b(?:shall|must|should|will|can|cannot|needs? to|has to)\b", re.I)
COMPOUND = re.compile(
    r"\b(?:and|also)\s+(?:(?:the\s+)?(?:system|service|portal|platform|app)\s+)?"
    r"(?:(?:shall|must|should|will)\s+)?"
    r"(?:allow|book|cancel|capture|confirm|create|delete|display|encrypt|"
    r"export|generate|import|integrate|log|notify|prevent|provide|record|"
    r"register|reject|retain|restrict|route|schedule|search|send|store|"
    r"support|sync|track|transmit|update|validate|verify)\b",
    re.I,
)


def _tokens(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]{3,}", text.lower())
        if word not in STOP_WORDS
    }


def _new_issue(
    issues: list[QualityIssue],
    requirement: Requirement | None,
    check: str,
    severity: str,
    message: str,
    recommendation: str,
) -> None:
    issues.append(QualityIssue(
        id=f"QI-{len(issues) + 1:03d}",
        requirement_id=requirement.id if requirement else None,
        check=check,
        severity=severity,
        message=message,
        recommendation=recommendation,
    ))


def _polarity(statement: str) -> set[str]:
    lowered = statement.lower()
    markers: set[str] = set()
    for positive, negative in (
        ("allow", "not allow"),
        ("permit", "prohibit"),
        ("enable", "disable"),
        ("retain", "delete"),
        ("accept", "reject"),
    ):
        if positive in lowered:
            markers.add(positive)
        if negative in lowered or negative == "prohibit" and "prohibit" in lowered:
            markers.add(negative)
    if re.search(r"\b(?:shall|must|should|will|can)\s+not\b|\bcannot\b", lowered):
        markers.add("negated")
    elif MODAL.search(lowered):
        markers.add("positive")
    return markers


def _contradiction(first: Requirement, second: Requirement) -> bool:
    first_tokens = _tokens(first.statement)
    second_tokens = _tokens(second.statement)
    shared = first_tokens & second_tokens
    union = first_tokens | second_tokens
    if not union or len(shared) / len(union) < 0.3:
        return False
    first_polarity = _polarity(first.statement)
    second_polarity = _polarity(second.statement)
    opposite_pairs = (
        ("allow", "not allow"), ("permit", "prohibit"),
        ("enable", "disable"), ("retain", "delete"),
        ("accept", "reject"),
    )
    if any((a in first_polarity and b in second_polarity)
           or (b in first_polarity and a in second_polarity)
           for a, b in opposite_pairs):
        return True
    return ("negated" in first_polarity) != ("negated" in second_polarity)


def analyse(requirements: list[Requirement]) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for req in requirements:
        statement = req.statement.strip()
        lowered = statement.lower()
        if any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", lowered) for word in VAGUE):
            _new_issue(
                issues, req, "Ambiguity", "Medium",
                "The requirement uses a qualitative term without a measurable threshold.",
                "Define an observable metric, limit, or acceptance test.",
            )
        if len(_tokens(statement)) < 5:
            _new_issue(
                issues, req, "Potential incompleteness", "Medium",
                "The requirement is too short to establish a clear system behavior.",
                "Specify the actor, behavior, and relevant outcome.",
            )
        if not ACTOR.search(statement):
            _new_issue(
                issues, req, "Missing actor", "Medium",
                "The requirement does not identify a system or stakeholder actor.",
                "Name the system component or role responsible for the behavior.",
            )
        if not MODAL.search(statement):
            _new_issue(
                issues, req, "Incomplete requirement statement", "Medium",
                "The statement does not express a required or permitted behavior.",
                "State the expected behavior with a clear requirement verb.",
            )
        if is_project_activity(statement):
            _new_issue(
                issues, req, "Project activity, not software requirement", "Low",
                "The statement describes testing, rollout, training, or approval work rather than system behavior.",
                "Track it as a project activity or convert only the supported system behavior into a requirement.",
            )
        if COMPOUND.search(statement):
            _new_issue(
                issues, req, "Multiple behaviors", "Medium",
                "The statement may combine independently verifiable behaviors.",
                "Split the behaviors into separate requirements and retain the same source references.",
            )
        if not req.evidence:
            _new_issue(
                issues, req, "Missing evidence", "High",
                "No source evidence is attached.",
                "Link an approved source or record an explicit stakeholder decision.",
            )
        elif len(_tokens(statement)) >= 4:
            evidence_tokens = _tokens(" ".join(item.excerpt for item in req.evidence))
            if len(_tokens(statement) & evidence_tokens) < 2:
                _new_issue(
                    issues, req, "Weak evidence alignment", "Medium",
                    "The statement has little lexical overlap with its cited excerpt; semantic support has not been established.",
                    "Review the cited text and either revise the requirement or attach a source that directly supports it.",
                )
        if not req.acceptance_criteria or any(
            re.search(r"\b(?:reviewer|stakeholder)\s+(?:must|should|needs? to)\b", item, re.I)
            for item in req.acceptance_criteria
        ):
            _new_issue(
                issues, req, "Lack of testability", "Medium",
                "Acceptance criteria are absent or defer measurable outcomes to a reviewer.",
                "Add pass/fail criteria, synthetic test data, and an expected result.",
            )
        evidence_tokens = _tokens(" ".join(item.excerpt for item in req.evidence))
        for assumption in req.assumptions:
            assumption_tokens = _tokens(assumption)
            if assumption_tokens and len(assumption_tokens & evidence_tokens) < 2:
                _new_issue(
                    issues, req, "Potentially unsupported assumption", "Medium",
                    "An assumption has little lexical support in the attached evidence; this heuristic cannot determine entailment.",
                    "Confirm the assumption with a stakeholder or link an approved source before baselining.",
                )

    for index, first in enumerate(requirements):
        first_tokens = _tokens(first.statement)
        for second in requirements[index + 1:]:
            second_tokens = _tokens(second.statement)
            union = first_tokens | second_tokens
            similarity = len(first_tokens & second_tokens) / len(union) if union else 0
            if _contradiction(first, second):
                _new_issue(
                    issues, first, "Potential contradiction", "High",
                    f"The requirement may conflict with {second.id}.",
                    "Resolve the conflicting behavior with the responsible stakeholder and retain both source references.",
                )
            elif similarity > 0.72:
                _new_issue(
                    issues, first, "Duplication", "Medium",
                    f"Potential overlap with {second.id}.",
                    "Merge, differentiate scope, or link the requirements.",
                )
    return issues

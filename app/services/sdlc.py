from __future__ import annotations

from collections.abc import Callable
import re
from typing import Literal

from app.schemas import (
    Category,
    ProjectContext,
    QualityIssue,
    Requirement,
    SdlcFactor,
    SdlcOption,
    SdlcRecommendation,
    WorkflowPhase,
)

SDLC_FACTOR_NAMES = [
    "Requirement stability",
    "Change volatility",
    "Project complexity",
    "Risk and criticality",
    "Stakeholder availability",
    "Iteration need",
    "Verification and validation",
    "Documentation burden",
    "Time pressure",
    "Dependency coupling",
    "Requirement clarity",
    "Patient-data sensitivity",
]

MODEL_SUITABILITY = {
    "Waterfall": "Sequential stages with stable scope and controlled changes.",
    "V-Shape": "Planned verification and validation mapped to each development stage.",
    "Prototyping": "Early workflow or interface learning where needs remain unclear.",
    "RAD": "Time-boxed delivery with available stakeholders and lower criticality.",
    "Spiral": "Iterative risk reduction for complex or high-criticality work.",
    "Incremental": "Small, validated releases for separable scope and dependencies.",
    "Agile": "Frequent stakeholder feedback and adaptation across iterations.",
}

MODEL_WEIGHTS: dict[str, dict[str, int]] = {
    "Waterfall": {
        "Requirement stability": 3, "Change volatility": -3, "Project complexity": -1,
        "Documentation burden": 2, "Time pressure": -1, "Requirement clarity": 3,
    },
    "V-Shape": {
        "Requirement stability": 2, "Risk and criticality": 3,
        "Verification and validation": 3, "Documentation burden": 2,
        "Requirement clarity": 2, "Patient-data sensitivity": 2,
    },
    "Prototyping": {
        "Change volatility": 2, "Stakeholder availability": 3,
        "Iteration need": 3, "Requirement clarity": -3, "Project complexity": 1,
    },
    "RAD": {
        "Time pressure": 3, "Stakeholder availability": 3,
        "Project complexity": -2, "Risk and criticality": -3,
        "Patient-data sensitivity": -2, "Requirement clarity": 1,
    },
    "Spiral": {
        "Project complexity": 2, "Risk and criticality": 3,
        "Verification and validation": 2, "Dependency coupling": 2,
        "Patient-data sensitivity": 2, "Change volatility": 1,
    },
    "Incremental": {
        "Project complexity": 2, "Change volatility": 2, "Iteration need": 3,
        "Dependency coupling": 2, "Verification and validation": 2,
        "Stakeholder availability": 1,
    },
    "Agile": {
        "Change volatility": 3, "Stakeholder availability": 3, "Iteration need": 3,
        "Requirement clarity": -1, "Verification and validation": 1,
        "Documentation burden": -1,
    },
}


def _requirement_ids(
    requirements: list[Requirement],
    predicate: Callable[[Requirement], bool],
) -> list[str]:
    return [item.id for item in requirements if predicate(item)]


def _factor(
    name: str,
    score: int,
    value: str,
    rationale: str,
    requirement_ids: list[str] | None = None,
) -> SdlcFactor:
    return SdlcFactor(
        name=name,
        score=max(1, min(5, score)),
        value=value,
        rationale=rationale,
        requirement_ids=requirement_ids or [],
    )


def _all_text(requirements: list[Requirement], context: ProjectContext) -> str:
    context_text = " ".join([
        context.objective,
        context.stakeholders,
        context.constraints,
        context.limitations,
        *context.interview_responses.values(),
    ])
    requirement_text = " ".join(
        " ".join([
            item.statement,
            item.business_justification,
            *item.assumptions,
            *item.healthcare_concerns,
            *item.risk_notes,
        ])
        for item in requirements
    )
    return f"{context_text} {requirement_text}".lower()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def fallback_factors(
    requirements: list[Requirement],
    context: ProjectContext,
    quality_issues: list[QualityIssue] | None = None,
) -> list[SdlcFactor]:
    """Derive conservative Agent 2 factors from the validated requirement set."""
    quality_issues = quality_issues or []
    categories = {category for item in requirements for category in item.categories}
    all_text = _all_text(requirements, context)
    context_text = " ".join([
        context.objective, context.stakeholders, context.constraints,
        context.limitations, *context.interview_responses.values(),
    ]).lower()
    issue_names: dict[str, set[str]] = {}
    for issue in quality_issues:
        if issue.requirement_id:
            issue_names.setdefault(issue.requirement_id, set()).add(issue.check.lower())
    clarity_issue_names = {
        "ambiguity", "potential incompleteness", "lack of testability",
        "missing actor", "incomplete requirement statement", "multiple behaviors",
        "weak evidence alignment",
    }
    unclear_ids = {
        requirement_id
        for requirement_id, checks in issue_names.items()
        if checks & clarity_issue_names
    }
    unclear_ids.update(
        item.id for item in requirements
        if not item.acceptance_criteria
        or any("reviewer" in criterion.lower() for criterion in item.acceptance_criteria)
    )
    unclear = [item for item in requirements if item.id in unclear_ids]
    integrations = _requirement_ids(
        requirements,
        lambda item: Category.INTEGRATION in item.categories,
    )
    risk_ids = _requirement_ids(
        requirements,
        lambda item: item.risk_level == "High"
        or bool({Category.SECURITY, Category.PRIVACY, Category.REGULATORY} & set(item.categories))
        or any(term in " ".join(item.risk_notes).lower() for term in ("high risk", "critical", "safety")),
    )
    privacy_ids = _requirement_ids(
        requirements,
        lambda item: Category.PRIVACY in item.categories
        or _contains_any(
            f"{item.statement} {' '.join(item.healthcare_concerns)}",
            (r"\bpatient health information\b", r"\bpersonal health information\b",
             r"\bmedical record\b", r"\bpatient data\b", r"\bmrn\b"),
        ),
    )
    dependency_ids = [item.id for item in requirements if item.dependencies]
    change_patterns = (
        r"\b(?:requirements?|scope|project needs?)\b.{0,40}\b(?:change|vary|evolv|shift)\w*\b",
        r"\b(?:change|vary|evolv|shift)\w*\b.{0,40}\b(?:requirements?|scope|project needs?)\b",
        r"\b(?:frequently|frequent|often|regularly)\s+(?:changing|change|changes)\b",
    )
    explicit_change = _contains_any(all_text, change_patterns)
    explicit_stability = _contains_any(all_text, (
        r"\b(?:stable|fixed|unlikely to change|frozen)\s+(?:requirements?|scope)\b",
        r"\b(?:requirements?|scope)\s+(?:are|remain)\s+(?:stable|fixed)\b",
    ))
    deadline_patterns = (
        r"\b(?:deadline|due date|deliver(?:y)?|release|launch|go[- ]live|complete)\b"
        r".{0,50}\b(?:\d+|one|two|three|four|five|six|eight|ten|twelve)\s*"
        r"(?:days?|weeks?|months?)\b",
        r"\b(?:urgent|tight deadline|time[- ]boxed|timebox)\b",
    )
    has_deadline = _contains_any(context_text, deadline_patterns)
    availability_patterns = (
        r"\b(?:available|availability|review|feedback|involvement)\b.{0,35}"
        r"\b(?:daily|weekly|fortnightly|biweekly|every sprint)\b",
        r"\b(?:daily|weekly|fortnightly|biweekly|every sprint)\b.{0,35}"
        r"\b(?:stakeholders?|customers?|users?)\b",
        r"\b(?:limited|unavailable|cannot attend)\b.{0,25}\b(?:stakeholders?|customers?)\b",
    )
    weekly_availability = _contains_any(all_text, availability_patterns[:2])
    limited_availability = _contains_any(all_text, availability_patterns[2:])
    explicit_iteration = _contains_any(all_text, (
        r"\b(?:prototype|prototype early|pilot|iteration|iterative|incremental|"
        r"feedback loop|frequent releases?)\b",
    ))
    explicit_complexity = _contains_any(all_text, (
        r"\b(?:multiple systems|legacy integration|complex workflow|many dependencies)\b",
    ))
    standalone = _contains_any(all_text, (
        r"\b(?:standalone|single system|no integrations?|no external dependencies)\b",
    ))
    explicit_low_risk = all(item.risk_level == "Low" for item in requirements) and not risk_ids
    explicit_sensitive_data = bool(privacy_ids)
    stability_score = 2 if explicit_change else 4 if explicit_stability else 3
    clarity_score = 2 if len(unclear) * 2 >= len(requirements) else 4 if not unclear else 3
    complexity_score = 4 if integrations or dependency_ids or explicit_complexity else 2 if standalone else 3
    risk_score = (
        5 if any(item.risk_level == "High" for item in requirements)
        else 4 if risk_ids
        else 2 if explicit_low_risk
        else 3
    )
    volatility_score = 4 if explicit_change else 2 if explicit_stability else 3
    change_or_prototype = explicit_change or explicit_iteration
    iteration_score = 4 if change_or_prototype or unclear else 2 if explicit_stability else 3
    validation_ids = {
        issue.requirement_id
        for issue in quality_issues
        if issue.requirement_id
        and issue.check.lower() in {"lack of testability", "weak evidence alignment"}
    }
    validation_score = 4 if risk_ids or validation_ids or categories & {Category.AVAILABILITY, Category.AUDIT} else 3
    documentation_score = 4 if categories & {Category.REGULATORY, Category.AUDIT} else 3
    time_score = 4 if has_deadline else 3
    stakeholder_score = 4 if weekly_availability else 2 if limited_availability else 3
    privacy_score = 5 if explicit_sensitive_data else 3
    dependency_score = 4 if dependency_ids else 3 if integrations else 2 if standalone else 3

    def unknown_value(known: bool, value: str) -> str:
        return value if known else "Insufficient evidence; neutral score used"

    volatility_ids = _requirement_ids(
        requirements, lambda item: _contains_any(item.statement, change_patterns)
    )
    iterative_ids = _requirement_ids(
        requirements, lambda item: _contains_any(item.statement, (
            r"\b(?:prototype|pilot|iteration|iterative|incremental|frequent releases?)\b",
        ))
    )
    time_ids = _requirement_ids(
        requirements, lambda item: _contains_any(item.statement, deadline_patterns)
    )

    return [
        _factor("Requirement stability", stability_score, unknown_value(explicit_change or explicit_stability, "Frequent change reported" if explicit_change else "Stable scope stated"), "Derived from explicit scope/change evidence; draft status alone does not imply volatility.", volatility_ids),
        _factor("Change volatility", volatility_score, unknown_value(explicit_change or explicit_stability, "Frequent change reported" if explicit_change else "Stable scope stated"), "Unknown change rate uses a neutral score; only explicit change/stability evidence changes it.", volatility_ids),
        _factor("Project complexity", complexity_score, unknown_value(bool(integrations or dependency_ids or explicit_complexity or standalone), "Integration/dependency signals" if integrations or dependency_ids or explicit_complexity else "Standalone scope stated"), "Derived from explicit integrations, dependencies, or standalone scope; general data handling alone is insufficient.", integrations + dependency_ids),
        _factor("Risk and criticality", risk_score, unknown_value(bool(risk_ids) or explicit_low_risk, "Requirement-level risk signals present" if risk_ids else "Low risk explicitly stated"), "Uses explicit high-risk labels and security/privacy/regulatory evidence; it is not a clinical risk assessment.", risk_ids),
        _factor("Stakeholder availability", stakeholder_score, unknown_value(weekly_availability or limited_availability, "Availability stated" if weekly_availability else "Availability limited"), "Roles alone do not prove availability; uses explicit meeting/availability evidence.", []),
        _factor(
            "Iteration need",
            iteration_score,
            "Change/prototype evidence present"
            if change_or_prototype
            else "Agent 1 clarity findings suggest iteration"
            if unclear
            else "Stable scope stated"
            if explicit_stability
            else "Insufficient evidence; neutral score used",
            "Uses explicit change/prototype signals and Agent 1 clarity findings, not draft status alone.",
            volatility_ids + iterative_ids + [item.id for item in unclear],
        ),
        _factor("Verification and validation", validation_score, "Elevated validation need" if validation_score >= 4 else "Insufficient evidence; neutral score used", "Requirement-level risk, availability, audit, and relevant quality findings inform verification needs.", risk_ids + sorted(validation_ids)),
        _factor("Documentation burden", documentation_score, "Audit or policy documentation signals" if documentation_score >= 4 else "Insufficient evidence; neutral score used", "Derived from explicit audit/regulatory categories, not an assumption of applicable law.", _requirement_ids(requirements, lambda item: bool({Category.REGULATORY, Category.AUDIT} & set(item.categories)))),
        _factor("Time pressure", time_score, "Deadline evidence present" if has_deadline else "Insufficient evidence; neutral score used", "Explicit deadline/timeline evidence changes the score; otherwise neutral.", time_ids),
        _factor("Dependency coupling", dependency_score, unknown_value(bool(dependency_ids or integrations or standalone), "Explicit dependencies or integrations" if dependency_ids or integrations else "Standalone scope stated"), "Derived from explicit requirement dependencies and integration categories.", dependency_ids + integrations),
        _factor("Requirement clarity", clarity_score, "Needs refinement" if clarity_score <= 2 else "Mixed evidence" if clarity_score == 3 else "Clear acceptance evidence", "Uses Agent 1 quality findings and measurable acceptance-criteria presence; it is a proxy, not a full quality score.", [item.id for item in unclear]),
        _factor("Patient-data sensitivity", privacy_score, "Sensitive-data requirements present" if explicit_sensitive_data else "Insufficient evidence; neutral score used", "Derived from privacy classification and explicit sensitive-data terms, not generic healthcare context.", privacy_ids),
    ]


def _score_option(name: str, factors: list[SdlcFactor]) -> SdlcOption:
    by_name = {item.name: item for item in factors}
    weights = MODEL_WEIGHTS[name]
    denominator = sum(abs(weight) * 5 for weight in weights.values())
    breakdown: dict[str, int] = {}
    for factor_name, weight in weights.items():
        level = by_name[factor_name].score
        fit = level if weight > 0 else 6 - level
        breakdown[factor_name] = abs(weight) * fit
    score = round(sum(breakdown.values()) * 100 / denominator)
    ordered_factors = sorted(breakdown, key=breakdown.get, reverse=True)
    strongest = ordered_factors[:2]
    limiting = ordered_factors[-1]
    linked_ids = sorted({
        requirement_id
        for factor_name in strongest
        for requirement_id in by_name[factor_name].requirement_ids
    })
    rationale = (
        "Strongest relative fit: "
        + "; ".join(
            f"{factor_name} ({breakdown[factor_name]} weighted fit points; assessed {by_name[factor_name].score}/5)"
            for factor_name in strongest
        )
        + f". Limiting fit: {limiting} ({breakdown[limiting]} weighted fit points; assessed {by_name[limiting].score}/5)."
    )
    return SdlcOption(
        name=name,
        score=score,
        suitability=MODEL_SUITABILITY[name],
        rationale=rationale,
        score_breakdown=breakdown,
        score_weights=weights,
        requirement_ids=linked_ids,
    )


def recommend(
    requirements: list[Requirement],
    factors: list[SdlcFactor] | None = None,
    context: ProjectContext | None = None,
    analysis_mode: Literal["llm-assisted", "rules-fallback"] = "rules-fallback",
    quality_issues: list[QualityIssue] | None = None,
) -> SdlcRecommendation:
    """Rank SDLC models transparently from Agent 2 factor assessments."""
    if not requirements:
        raise ValueError("At least one Agent 1 requirement is required for SDLC selection.")
    context = context or ProjectContext()
    factors = factors or fallback_factors(requirements, context, quality_issues)
    factor_names = {item.name for item in factors}
    if factor_names != set(SDLC_FACTOR_NAMES):
        raise ValueError("SDLC factors must contain exactly the supported factor names.")
    valid_ids = {item.id for item in requirements}
    if any(not set(item.requirement_ids) <= valid_ids for item in factors):
        raise ValueError("SDLC factors may reference only Agent 1 requirement IDs.")

    options = [_score_option(name, factors) for name in MODEL_WEIGHTS]
    options.sort(key=lambda option: (-option.score, option.name))
    choice = options[0].name
    top_factors = sorted(factors, key=lambda item: item.score, reverse=True)[:3]
    factor_summary = "; ".join(
        f"{item.name} is {item.value.lower()} ({item.score}/5)"
        for item in top_factors
    )
    traceable_ids = sorted({
        requirement_id
        for item in top_factors
        for requirement_id in item.requirement_ids
    })
    justification = (
        f"{choice} ranks highest under the documented weighted fit matrix. "
        f"The strongest assessed characteristics are: {factor_summary}. "
        "Model scores are comparative decision support, not probabilities or a compliance determination."
    )
    if traceable_ids:
        justification += f" Supporting Agent 1 requirement IDs: {', '.join(traceable_ids)}."
    else:
        justification += " The leading factor assessments are based on project context rather than linked requirement IDs."
    if options[1].score == options[0].score:
        justification += " The top score is tied; stakeholders should review the tied alternatives."

    risks_and_limitations = [
        f"{factor.name}: evidence is insufficient; the neutral score may affect the ranking."
        for factor in factors
        if "insufficient evidence" in factor.value.lower()
    ]
    risks_and_limitations.extend([
        "The fixed weighted matrix is a comparative heuristic, not a probability or calibrated outcome prediction.",
        "Requirement-level risk factors are not a clinical safety assessment or regulatory compliance determination.",
    ])
    requirement_ids = [item.id for item in requirements]
    workflow = [
        WorkflowPhase(
            phase="Validate the requirement baseline",
            activities=[
                "Resolve Agent 1 quality findings",
                "Confirm constraints, assumptions, and stakeholder availability",
                "Review patient-data, access-control, reliability, and audit requirements",
            ],
            roles=["Healthcare stakeholder", "Business analyst", "Security/privacy reviewer"],
            deliverables=["Reviewed requirements", "Quality disposition", "Assumption and risk review"],
            gate="Human approval of requirement scope and unresolved risks",
        ),
        WorkflowPhase(
            phase=f"Plan {choice} delivery and verification",
            activities=[
                f"Organise work using {choice} planning and review points",
                "Map requirement IDs to design decisions and verification evidence",
                "Plan integration, privacy, availability, and recovery checks where requirements call for them",
            ],
            roles=["Project lead", "Architect", "QA", "Clinical operations representative"],
            deliverables=["Traceability plan", "Verification and validation plan", "Delivery increments or stages"],
            gate="Engineering and stakeholder approval of the delivery plan",
        ),
        WorkflowPhase(
            phase="Build and verify in controlled scope",
            activities=[
                "Implement approved requirements",
                "Test acceptance criteria and requirement-level controls",
                "Review exceptions, data integrity, access boundaries, and audit events",
            ],
            roles=["Developers", "QA", "Security/privacy reviewer", "Domain stakeholder"],
            deliverables=["Working increment or verified stage", "Test results linked to requirement IDs"],
            gate="Pass required verification and obtain stakeholder review before progression",
        ),
        WorkflowPhase(
            phase="Pilot, release, and reassess",
            activities=[
                "Use a synthetic-data pilot and controlled release",
                "Monitor reliability and workflow outcomes",
                "Feed validated changes back into requirements and the SDLC assessment",
            ],
            roles=["Operations", "Product owner", "Clinical operations representative"],
            deliverables=["Release decision", "Monitoring plan", "Updated requirements and risk notes"],
            gate="Human release approval; no clinical or regulatory suitability is implied",
        ),
    ]
    return SdlcRecommendation(
        recommended_sdlc=choice,
        recommended_score=options[0].score,
        factors=factors,
        options=options,
        justification=justification,
        risks_and_limitations=risks_and_limitations,
        workflow=workflow,
        analysis_mode=analysis_mode,
        input_requirement_ids=requirement_ids,
    )

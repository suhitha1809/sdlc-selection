from __future__ import annotations

import re

from app.schemas import Category, Evidence, ProjectActivity, Requirement


KEYWORDS: list[tuple[Category, tuple[str, ...]]] = [
    (Category.SECURITY, ("authentication", "authorisation", "authorization", "encrypt", "security", "access", "mfa", "role-based", "role based")),
    (Category.PRIVACY, ("patient data", "personal health information", "health information", "privacy", "consent", "retention", "confidentiality")),
    (Category.REGULATORY, ("regulation", "compliance", "policy", "legal", "governance", "clinical policy")),
    (Category.PERFORMANCE, ("response time", "performance", "latency", "throughput", "within seconds", "wait time")),
    (Category.AVAILABILITY, ("appointment", "schedule", "booking", "double booking", "reschedule", "cancel", "recovery", "uptime", "reliable", "availability")),
    (Category.INTEGRATION, ("api", "integration", "legacy", "interface", "ehr", "lab system", "telemedicine")),
    (Category.AUDIT, ("audit", "report", "logging", "trace", "record every decision")),
    (Category.DATA, ("data", "database", "record", "document", "medical record", "patient record", "clinical workflow")),
    (Category.USABILITY, ("accessibility", "accessible", "screen reader", "usability", "user interface", "easy to use")),
    (Category.OPERATIONAL, ("monitor", "maintenance", "deployment", "operation", "support")),
]

SOFTWARE_ACTOR = re.compile(
    r"\b(?:the\s+)?(?:system|software|service|application|app|portal|platform|"
    r"scheduler|module|lims|lis|laboratory information management system|"
    r"laboratory information system|laboratory management system|lab system)\b",
    re.I,
)
STAKEHOLDER_CAPABILITY = re.compile(
    r"^\s*(?:patients?|users?|clinicians?|physicians?|nurses?|staff|"
    r"administrators?|schedulers?|lab(?:oratory)?\s+(?:staff|technicians?|"
    r"technologists?))\s+(?:can|cannot|may|must be able to|should be able to)\b",
    re.I,
)
MODAL = re.compile(
    r"\b(?:shall|must|should|will|needs? to|has to|can|cannot|must not|shall not|should not)\b",
    re.I,
)
ACTION_STARTS = (
    "access", "allow", "assign", "book", "calculate", "cancel", "capture",
    "compare", "confirm", "connect", "create", "delete", "display", "download",
    "encrypt", "export", "flag", "generate", "identify", "import", "integrate",
    "log", "maintain", "measure", "notify", "prevent", "process", "provide",
    "receive", "record", "register", "reject", "retain", "restrict", "route",
    "schedule", "search", "send", "store", "support", "sync", "track",
    "transmit", "update", "upload", "validate", "verify", "view",
)
ACTIVITY = re.compile(
    r"^\s*(?:(?:(?:(?:before|after|during|prior to|following)\s+)?"
    r"(?:go[- ]live|launch|deployment|release)|post[- ](?:launch|deployment))"
    r"[^.;:]{0,80}\b"
    r"(?:test|testing|pilot|roll ?out|deploy|train|validate|monitor|"
    r"approve|review|conduct|perform|complete)\w*\b|"
    r"(?:the\s+)?(?:project\s+)?(?:team|developers?|testers?|qa|staff|"
    r"stakeholders?|operations team|project manager)\s+"
    r"(?:(?:shall|must|should|will|needs? to|need|to)\s+)?"
    r"(?:(?:run|conduct|perform|complete|carry out|hold|schedule)\s+)?"
    r"(?:test|testing|user acceptance testing|uat|pilot|roll out|rollout|"
    r"deploy|deployment|release|train|training|validate|validation|"
    r"monitor|monitoring|approve|approval|review|reviewing|sign off)\b|"
    r"(?:(?:run|conduct|perform|complete|carry out|hold|schedule)\s+)?"
    r"(?:test|testing|user acceptance testing|uat|pilot|roll out|rollout|"
    r"deploy|deployment|release|train|training|validate|validation|"
    r"monitor|monitoring|approve|approval|review|reviewing|sign off)\b)",
    re.I,
)
SYSTEM_ACTIVITY = re.compile(
    r"^\s*(?:the\s+)?(?:system|service|application|app|portal|platform)\s+"
    r"(?:shall|must|should|will)\s+be\s+(?:tested|piloted|deployed|released|"
    r"validated|approved|rolled out)\b",
    re.I,
)
DOMAIN_TERMS = {
    "laboratory": (
        r"\blaborator(?:y|ies)\b", r"\blims\b", r"\blis\b",
        r"\bspecimen\b", r"\bassay\b", r"\blab(?:oratory)?\s+(?:sample|"
        r"result|system|information|management|technician|staff)\b",
    ),
    "appointments": (
        r"\bappointments?\b", r"\bbookings?\b", r"\bschedul(?:e|ing)\b",
        r"\breschedul(?:e|ing)\b", r"\bdouble booking\b",
    ),
    "telemedicine": (
        r"\btelemedicine\b", r"\bvirtual visits?\b", r"\bvideo consultations?\b",
    ),
    "pharmacy": (
        r"\bpharmac(?:y|ies)\b", r"\bmedication inventory\b",
        r"\bdrug inventory\b", r"\bdispens(?:e|ing)\b",
    ),
    "emergency intake": (
        r"\bemergency intake\b", r"\bemergency department\b",
        r"\btriage\b", r"\b(?:ed|er)\s+intake\b",
    ),
}
MODAL_WORD = re.compile(
    r"\b(?:shall|must|should|will|needs? to|has to|can|cannot)\b",
    re.I,
)
VAGUE_TERM = re.compile(
    r"\b(?:adequate|appropriate|efficient|easy|fast|reliable|robust|"
    r"secure|user-friendly)\b",
    re.I,
)
CLINICAL_ACTION = re.compile(
    r"\b(?:take|administer|prescribe|diagnose|treat|dose|medicate)\b", re.I
)
STOP_WORDS = {
    "about", "after", "before", "could", "from", "have", "shall", "should",
    "that", "their", "there", "these", "this", "those", "will", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in STOP_WORDS
    }


def classify(text: str) -> list[Category]:
    found = [category for category, terms in KEYWORDS if any(term in text.lower() for term in terms)]
    return found or [Category.FUNCTIONAL]


def _source_statements(chunk: dict) -> list[str]:
    text = chunk["text"]
    if chunk.get("document_name") == "Stakeholder interview":
        text = re.sub(r"^[^:]{1,120}:\s*", "", text, count=1)
    segments = re.split(r"(?<=[.!?])\s+|[;\n•]+", text)
    activity_start = (
        r"(?:(?:the\s+)?(?:project\s+)?(?:team|developers?|testers?|qa|staff|"
        r"stakeholders?|operations team|project manager)\s+"
        r"(?:(?:shall|must|should|will|needs? to|need|to)\s+)?"
        r"(?:(?:run|conduct|perform|complete|carry out|hold|schedule)\s+)?"
        r"(?:test|testing|user acceptance testing|uat|pilot|roll out|rollout|"
        r"deploy|deployment|release|train|training|validate|validation|"
        r"monitor|monitoring|approve|approval|review|reviewing|sign off)|"
        r"(?:(?:(?:before|after|during|prior to|following)\s+)?"
        r"(?:go[- ]live|launch|deployment|release)|post[- ](?:launch|deployment))"
        r"[^.;:]{0,80}\b(?:test|testing|pilot|roll ?out|deploy|train|"
        r"validate|monitor|approve|review|conduct|perform|complete))"
    )
    statements = []
    for segment in segments:
        segment = segment.strip(" \t-•")
        if not segment:
            continue
        clauses = re.split(
            rf"\s+(?:and|also)\s+(?={activity_start}\b)",
            segment,
            flags=re.I,
        )
        statements.extend(clause.strip(" \t-•") for clause in clauses if clause.strip(" \t-•"))
    return statements


def _canonicalize_conversation(text: str) -> str:
    actor = (
        r"(?:laboratory information management system|laboratory information "
        r"system|laboratory management system|lab system|lims|lis|system|"
        r"service|application|app|portal|platform)"
    )
    text = re.sub(
        rf"^\s*(?:we|i)\s+need\s+(?:the\s+)?({actor})\s+to\s+",
        lambda match: f"The {match.group(1)} shall ",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"^\s*(?:the\s+)?({actor})\s+"
        r"(?:needs|need|has)\s+to\s+",
        lambda match: f"The {match.group(1)} shall ",
        text,
        flags=re.I,
    )
    return text


def _decompose(statement: str) -> list[str]:
    statement = _canonicalize_conversation(statement).strip()
    match = re.match(
        r"^(?P<subject>.+?\b(?:shall|must|should|will|can|cannot|needs? to|has to))"
        r"\s+(?P<body>.+)$",
        statement,
        re.I,
    )
    if not match:
        return [statement]
    subject = match.group("subject")
    body = match.group("body")
    actions = "|".join(sorted(ACTION_STARTS, key=len, reverse=True))
    parts = re.split(
        rf"(?:,\s*(?:and\s+)?|\s+(?:and|also)\s+)"
        rf"(?=(?:(?:shall|must|should|will)\s+)?(?:{actions})\b)",
        body,
        flags=re.I,
    )
    results: list[str] = []
    for part in parts:
        part = re.sub(r"^(?:shall|must|should|will)\s+", "", part, flags=re.I).strip()
        if part:
            results.append(f"{subject} {part}")
    return results or [statement]


def _is_requirement(statement: str) -> bool:
    if is_project_activity(statement):
        return False
    if not MODAL.search(statement):
        return False
    if SOFTWARE_ACTOR.search(statement) or STAKEHOLDER_CAPABILITY.search(statement):
        return True
    if re.match(r"^\s*(?:patients?|clinicians?|users?)\b", statement, re.I) and CLINICAL_ACTION.search(statement):
        return False
    return False


def is_project_activity(statement: str) -> bool:
    return bool(ACTIVITY.search(statement) or SYSTEM_ACTIVITY.search(statement))


def _domains(text: str) -> set[str]:
    return {
        domain
        for domain, patterns in DOMAIN_TERMS.items()
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    }


def domain_mismatch(functionality: str, statement: str) -> bool:
    project_domains = _domains(functionality)
    statement_domains = _domains(statement)
    return bool(
        project_domains
        and statement_domains
        and project_domains.isdisjoint(statement_domains)
    )


def partition_project_evidence(
    functionality: str,
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    project_domains = _domains(functionality)
    relevant: list[dict] = []
    mismatches: list[dict] = []
    for chunk in chunks:
        for statement in _source_statements(chunk):
            source_domains = _domains(statement)
            if project_domains and source_domains and project_domains.isdisjoint(source_domains):
                mismatches.append({
                    **chunk,
                    "text": statement,
                    "project_domains": sorted(project_domains),
                    "source_domains": sorted(source_domains),
                })
            else:
                relevant.append({**chunk, "text": statement})
    return relevant, mismatches


def _draft_acceptance_criteria(statement: str) -> list[str]:
    if VAGUE_TERM.search(statement):
        return []
    match = MODAL_WORD.search(statement)
    if not match:
        return []
    behavior = f"{statement[:match.start()]}{statement[match.end():]}".strip(" .")
    if not behavior:
        return []
    return [f"Using synthetic data, verify that {behavior[0].lower() + behavior[1:]}."] 


def _activity_category(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("test", "testing", "qa", "acceptance")):
        return "Testing"
    if any(term in lowered for term in ("validate", "validation", "approve", "approval", "uat", "sign off")):
        return "Validation"
    if any(term in lowered for term in ("pilot", "rollout", "roll out", "deploy", "release", "train")):
        return "Rollout"
    if any(term in lowered for term in ("monitor", "review", "go-live", "launch")):
        return "Operations"
    return "Project management"


def extract_project_activities(chunks: list[dict]) -> list[ProjectActivity]:
    activities: list[ProjectActivity] = []
    seen: set[str] = set()
    for chunk in chunks:
        for statement in _source_statements(chunk):
            if not is_project_activity(statement):
                continue
            normalized = " ".join(statement.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            activities.append(ProjectActivity(
                id=f"ACT-{len(activities) + 1:03d}",
                activity=statement,
                category=_activity_category(statement),
                evidence=[Evidence(
                    document_id=chunk["document_id"],
                    document_name=chunk["document_name"],
                    chunk_id=chunk["id"],
                    excerpt=statement[:500],
                    relevance=chunk["relevance"],
                )],
            ))
    return activities


def generate_evidence_drafts(functionality: str, chunks: list[dict]) -> list[Requirement]:
    candidates: list[tuple[dict, str]] = []
    for chunk in chunks:
        for statement in _source_statements(chunk):
            for requirement in _decompose(statement):
                if _is_requirement(requirement) and len(_tokens(requirement)) >= 3:
                    candidates.append((chunk, requirement))

    requirements: list[Requirement] = []
    seen: set[str] = set()
    for chunk, statement in candidates:
        normalized = " ".join(_tokens(statement))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        evidence = Evidence(
            document_id=chunk["document_id"],
            document_name=chunk["document_name"],
            chunk_id=chunk["id"],
            excerpt=chunk["text"][:500],
            relevance=chunk["relevance"],
        )
        requirements.append(Requirement(
            id=f"REQ-{len(requirements) + 1:03d}",
            statement=statement,
            categories=classify(statement),
            evidence=[evidence],
            business_justification=f"Supports the documented {functionality} need.",
            priority="Should",
            acceptance_criteria=_draft_acceptance_criteria(statement),
            assumptions=["Priority defaults to Should because no source priority was stated."],
            confidence=min(0.9, max(0.35, chunk["relevance"])),
            reasoning=(
                "Evidence-only draft with a source-bounded acceptance check where the behavior is clear; "
                "confirm scope, priority, assumptions, preconditions, thresholds, and edge cases with stakeholders."
            ),
            approval_status="Needs review",
        ))
        if len(requirements) == 12:
            break
    return requirements

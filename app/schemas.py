from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Category(StrEnum):
    BUSINESS = "Business"
    STAKEHOLDER = "Stakeholder"
    FUNCTIONAL = "Functional"
    SECURITY = "Security"
    PRIVACY = "Privacy"
    REGULATORY = "Regulatory"
    PERFORMANCE = "Performance"
    AVAILABILITY = "Availability / reliability"
    USABILITY = "Usability"
    DATA = "Data management"
    INTEGRATION = "Integration"
    AUDIT = "Audit / reporting"
    OPERATIONAL = "Operational / maintenance"
class Evidence(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    excerpt: str
    relevance: float = Field(ge=0, le=1)
class Requirement(BaseModel):
    id: str
    statement: str = Field(min_length=1)
    categories: list[Category]
    evidence: list[Evidence] = Field(default_factory=list)
    business_justification: str
    priority: Literal["Must", "Should", "Could"] = "Should"
    dependencies: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    applicable_policy: str | None = None
    risk_level: Literal["Low", "Medium", "High"] = "Medium"
    risk_notes: list[str] = Field(default_factory=list)
    healthcare_concerns: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    approval_status: Literal["Draft", "Approved", "Rejected", "Needs review"] = "Draft"


class QualityIssue(BaseModel):
    id: str
    requirement_id: str | None = None
    check: str
    severity: Literal["Low", "Medium", "High"]
    message: str
    recommendation: str


class ProjectActivity(BaseModel):
    id: str
    activity: str = Field(min_length=1)
    category: Literal["Testing", "Validation", "Rollout", "Operations", "Project management"]
    evidence: list[Evidence] = Field(min_length=1)


class ProjectContext(BaseModel):
    objective: str = ""
    stakeholders: str = ""
    constraints: str = ""
    limitations: str = ""
    interview_responses: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    id: str
    text: str
    document_id: str
    document_name: str
    relevance: float = Field(ge=0, le=1)


class RequirementsAgentInput(BaseModel):
    project_id: str = Field(min_length=1)
    functionality: str = Field(min_length=3)
    context: ProjectContext = Field(default_factory=ProjectContext)
    evidence: list[RetrievedChunk] = Field(default_factory=list)


class RequirementsAgentOutput(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)
    project_activities: list[ProjectActivity] = Field(default_factory=list)
    analysis_gaps: list[str] = Field(default_factory=list)
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    generation_mode: Literal["llm", "fallback"]
    retrieved_chunk_count: int = Field(ge=1)
    validation_passed: bool = True

    @model_validator(mode="after")
    def validate_requirement_set(self) -> RequirementsAgentOutput:
        requirement_ids = [item.id for item in self.requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Agent 1 requirement IDs must be unique.")
        if self.validation_passed and (
            not self.requirements or any(not item.evidence for item in self.requirements)
        ):
            raise ValueError("Validated Agent 1 output requires evidenced requirements.")
        return self


class SdlcFactor(BaseModel):
    name: str
    value: str
    score: int = Field(ge=1, le=5)
    rationale: str
    requirement_ids: list[str] = Field(default_factory=list)
class SdlcOption(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    suitability: str
    rationale: str = ""
    score_breakdown: dict[str, int] = Field(default_factory=dict)
    score_weights: dict[str, int] = Field(default_factory=dict)
    requirement_ids: list[str] = Field(default_factory=list)


class WorkflowPhase(BaseModel):
    phase: str
    activities: list[str]
    roles: list[str]
    deliverables: list[str]
    gate: str


class SdlcRecommendation(BaseModel):
    recommended_sdlc: str
    recommended_score: int = Field(default=0, ge=0, le=100)
    factors: list[SdlcFactor]
    options: list[SdlcOption]
    justification: str
    risks_and_limitations: list[str] = Field(default_factory=list)
    workflow: list[WorkflowPhase]
    analysis_mode: Literal["llm-assisted", "rules-fallback"] = "rules-fallback"
    input_requirement_ids: list[str] = Field(default_factory=list)
    human_approval_required: bool = True


class SdlcApiResponse(SdlcRecommendation):
    factors: list[SdlcFactor] = Field(min_length=1)
    options: list[SdlcOption] = Field(min_length=7, max_length=7)
    recommended_score: int = Field(ge=0, le=100)
    workflow: list[WorkflowPhase] = Field(min_length=1)
    input_requirement_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_recommendation_traceability(self) -> SdlcApiResponse:
        if self.recommended_sdlc != self.options[0].name:
            raise ValueError("The recommended model must be the highest-ranked option.")
        if self.recommended_score != self.options[0].score:
            raise ValueError("The recommended score must match the highest-ranked option.")
        if len({option.name for option in self.options}) != len(self.options):
            raise ValueError("SDLC option names must be unique.")
        if len({factor.name for factor in self.factors}) != len(self.factors):
            raise ValueError("SDLC factor names must be unique.")
        if len(set(self.input_requirement_ids)) != len(self.input_requirement_ids):
            raise ValueError("Agent 1 requirement IDs must be unique.")
        return self


class SdlcAgentInput(BaseModel):
    requirements_agent_output: RequirementsAgentOutput
    context: ProjectContext = Field(default_factory=ProjectContext)


class SdlcAgentOutput(BaseModel):
    recommendation: SdlcRecommendation
    validation_passed: bool = True


class AgentExecution(BaseModel):
    agent_name: Literal["Requirements Engineering Agent", "SDLC Selection Agent"]
    mode: Literal["llm", "fallback", "llm-assisted", "rules-fallback"]
    status: Literal["completed"] = "completed"
    validation_passed: bool = True
    requirement_count: int = 0
    retrieved_chunk_count: int = 0


class Project(BaseModel):
    id: str
    functionality: str
    created_at: datetime
    context: ProjectContext = Field(default_factory=ProjectContext)
    documents: list[dict] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    project_activities: list[ProjectActivity] = Field(default_factory=list)
    analysis_gaps: list[str] = Field(default_factory=list)
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    sdlc: SdlcRecommendation | None = None
    agent_runs: dict[str, AgentExecution] = Field(default_factory=dict)
    audit_log: list[dict] = Field(default_factory=list)

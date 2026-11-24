from __future__ import annotations

import logging

from pydantic import ValidationError

from app.schemas import (
    AgentExecution,
    ProjectContext,
    RequirementsAgentInput,
    RequirementsAgentOutput,
    Requirement,
    RetrievedChunk,
    QualityIssue,
    SdlcAgentInput,
    SdlcAgentOutput,
    SdlcFactor,
)
from app.services import llm
from app.services.generator import (
    domain_mismatch,
    extract_project_activities,
    generate_evidence_drafts,
    partition_project_evidence,
)
from app.services.quality import analyse
from app.services.rag import VectorStore
from app.services.sdlc import SDLC_FACTOR_NAMES, fallback_factors, recommend

logger = logging.getLogger(__name__)


class MissingRetrievedEvidence(ValueError):
    """Raised when Agent 1 cannot obtain source evidence for the project."""


class RequirementsEngineeringAgent:
    """Agent 1: retrieve-grounded requirement drafting and deterministic validation."""

    name = "Requirements Engineering Agent"

    def run(
        self,
        agent_input: RequirementsAgentInput,
        vector_store: VectorStore | None = None,
    ) -> RequirementsAgentOutput:
        retrieved = agent_input.evidence
        if not retrieved and vector_store is not None:
            query = " ".join(
                value for value in [
                    agent_input.functionality,
                    agent_input.context.objective,
                    agent_input.context.stakeholders,
                    agent_input.context.constraints,
                    agent_input.context.limitations,
                    *agent_input.context.interview_responses.values(),
                ] if value
            )[:2000]
            retrieved = [
                RetrievedChunk.model_validate(item)
                for item in vector_store.retrieve(agent_input.project_id, query)
            ]
        if not retrieved:
            raise MissingRetrievedEvidence("No retrievable evidence was found for Agent 1.")

        evidence = [item.model_dump() for item in retrieved]
        project_evidence, domain_mismatches = partition_project_evidence(
            agent_input.functionality,
            evidence,
        )
        usable_evidence = [
            RetrievedChunk.model_validate(item)
            for item in project_evidence
        ]
        mode = "fallback"
        requirements = None
        if llm.enabled() and usable_evidence:
            try:
                requirements = llm.generate_requirements(
                    agent_input.functionality,
                    project_evidence,
                    agent_input.context,
                )
                mode = "llm"
            except (RuntimeError, ValueError, ValidationError) as exc:
                logger.warning(
                    "Agent 1 LLM generation failed; using evidence-only drafts: %s",
                    exc,
                )

        if requirements is None:
            mode = "fallback"
            requirements = generate_evidence_drafts(agent_input.functionality, project_evidence)
        try:
            requirements = self._validate_requirements(
                requirements,
                usable_evidence,
                agent_input.functionality,
            )
        except ValueError as exc:
            if mode != "llm":
                raise
            logger.warning(
                "Agent 1 output validation failed; using evidence-only drafts: %s",
                exc,
            )
            mode = "fallback"
            requirements = generate_evidence_drafts(agent_input.functionality, project_evidence)
            requirements = self._validate_requirements(
                requirements,
                usable_evidence,
                agent_input.functionality,
            )

        project_activities = extract_project_activities(project_evidence)
        quality_issues = analyse(requirements)
        analysis_gaps = []
        for mismatch in domain_mismatches:
            quality_issues.append(QualityIssue(
                id=f"QI-{len(quality_issues) + 1:03d}",
                check="Project-domain inconsistency",
                severity="High",
                message=(
                    f"Excluded a source statement from {mismatch['document_name']} "
                    f"({mismatch['document_id']}/{mismatch['id']}): it indicates "
                    f"{', '.join(mismatch['source_domains'])}, while the selected "
                    f"functionality indicates {', '.join(mismatch['project_domains'])}."
                ),
                recommendation=(
                    "Confirm whether this response or source belongs to the selected project. "
                    "Correct the project functionality or source responses, then regenerate requirements."
                ),
            ))
        if domain_mismatches:
            analysis_gaps.append(
                f"Excluded {len(domain_mismatches)} source statement(s) whose detected domain conflicts "
                "with the selected functionality. Resolve the project/source mismatch before treating "
                "the requirement set as validated."
            )
        if not requirements:
            analysis_gaps.append(
                "No sufficiently specific software behavior could be extracted from the retrieved evidence. "
                "Add a clear system/user action, condition, and expected outcome."
            )
            quality_issues.append(
                QualityIssue(
                    id=f"QI-{len(quality_issues) + 1:03d}",
                    check="Insufficient requirement evidence",
                    severity="Medium",
                    message="The retrieved evidence contained no testable software requirement.",
                    recommendation="Clarify the system behavior and its expected outcome; keep project activities separate.",
                )
            )
        return RequirementsAgentOutput(
            requirements=requirements,
            project_activities=project_activities,
            analysis_gaps=analysis_gaps,
            quality_issues=quality_issues,
            generation_mode=mode,
            retrieved_chunk_count=len(retrieved),
            validation_passed=bool(requirements) and not domain_mismatches,
        )

    @staticmethod
    def _validate_requirements(
        requirements: object,
        retrieved: list[RetrievedChunk],
        functionality: str | None = None,
    ) -> list[Requirement]:
        if not isinstance(requirements, list):
            raise ValueError("Agent 1 returned a non-list requirement structure.")
        try:
            validated = [Requirement.model_validate(item) for item in requirements]
        except ValidationError as exc:
            raise ValueError("Agent 1 returned an invalid requirement structure.") from exc
        evidence_ids = {(item.document_id, item.id) for item in retrieved}
        requirement_ids = [item.id for item in validated]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Agent 1 generated duplicate requirement IDs.")
        for requirement in validated:
            if functionality and domain_mismatch(functionality, requirement.statement):
                raise ValueError(
                    f"{requirement.id} is inconsistent with the selected project functionality."
                )
            if not requirement.evidence:
                raise ValueError(f"{requirement.id} has no evidence reference.")
            if any(
                (source.document_id, source.chunk_id) not in evidence_ids
                for source in requirement.evidence
            ):
                raise ValueError(f"{requirement.id} refers to evidence outside Agent 1's retrieved input.")
        return validated

    @staticmethod
    def execution(output: RequirementsAgentOutput) -> AgentExecution:
        return AgentExecution(
            agent_name="Requirements Engineering Agent",
            mode=output.generation_mode,
            validation_passed=output.validation_passed,
            requirement_count=len(output.requirements),
            retrieved_chunk_count=output.retrieved_chunk_count,
        )


class SdlcSelectionAgent:
    """Agent 2: assess Agent 1 requirements and apply a transparent model score matrix."""

    name = "SDLC Selection Agent"

    def run(self, agent_input: SdlcAgentInput) -> SdlcAgentOutput:
        first_agent = agent_input.requirements_agent_output
        if not first_agent.validation_passed:
            raise ValueError("Agent 2 requires validated Agent 1 output.")
        if not first_agent.requirements:
            raise ValueError("Agent 2 requires at least one Agent 1 requirement.")

        valid_ids = {item.id for item in first_agent.requirements}
        mode = "rules-fallback"
        factors: list[SdlcFactor] | None = None
        if llm.enabled():
            try:
                raw_factors = llm.analyse_sdlc(
                    first_agent.requirements,
                    agent_input.context,
                    SDLC_FACTOR_NAMES,
                    first_agent.quality_issues,
                )
                factors = self._validate_factors(raw_factors, valid_ids)
                mode = "llm-assisted"
            except (RuntimeError, ValueError, ValidationError) as exc:
                logger.warning(
                    "Agent 2 LLM analysis failed; using rules-based factors: %s",
                    exc,
                )

        if factors is None:
            factors = fallback_factors(
                first_agent.requirements,
                agent_input.context,
                first_agent.quality_issues,
            )

        recommendation = recommend(
            first_agent.requirements,
            factors=factors,
            context=agent_input.context,
            analysis_mode=mode,
            quality_issues=first_agent.quality_issues,
        )
        return SdlcAgentOutput(recommendation=recommendation, validation_passed=True)

    @staticmethod
    def _validate_factors(raw_factors: list[dict], valid_ids: set[str]) -> list[SdlcFactor]:
        if not isinstance(raw_factors, list):
            raise ValueError("Agent 2 returned an invalid factor structure.")
        factors = [SdlcFactor.model_validate(item) for item in raw_factors]
        names = [item.name for item in factors]
        if len(names) != len(set(names)) or set(names) != set(SDLC_FACTOR_NAMES):
            raise ValueError("Agent 2 must return exactly one assessment per supported SDLC factor.")
        if any(not set(item.requirement_ids) <= valid_ids for item in factors):
            raise ValueError("Agent 2 referenced a requirement ID that Agent 1 did not generate.")
        return factors

    @staticmethod
    def execution(output: SdlcAgentOutput) -> AgentExecution:
        return AgentExecution(
            agent_name="SDLC Selection Agent",
            mode=output.recommendation.analysis_mode,
            validation_passed=output.validation_passed,
            requirement_count=len(output.recommendation.input_requirement_ids),
        )


class AgentOrchestrator:
    """Explicit, sequential handoff between the two domain agents."""

    def __init__(
        self,
        requirements_agent: RequirementsEngineeringAgent | None = None,
        sdlc_agent: SdlcSelectionAgent | None = None,
    ) -> None:
        self.requirements_agent = requirements_agent or RequirementsEngineeringAgent()
        self.sdlc_agent = sdlc_agent or SdlcSelectionAgent()

    def run_requirements(
        self,
        agent_input: RequirementsAgentInput,
        vector_store: VectorStore,
    ) -> RequirementsAgentOutput:
        return self.requirements_agent.run(agent_input, vector_store)

    def run_sdlc(
        self,
        requirements_output: RequirementsAgentOutput,
        context: ProjectContext,
    ) -> SdlcAgentOutput:
        handoff = SdlcAgentInput(
            requirements_agent_output=requirements_output,
            context=context,
        )
        return self.sdlc_agent.run(handoff)

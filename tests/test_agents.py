from __future__ import annotations

import json
from io import BytesIO

import pytest
from pydantic import ValidationError

from app.schemas import (
    Category,
    Evidence,
    ProjectContext,
    QualityIssue,
    Requirement,
    RequirementsAgentInput,
    RequirementsAgentOutput,
    RetrievedChunk,
    SdlcApiResponse,
)
from app.services import agents
from app.services import llm as llm_service
from app.services.agents import (
    AgentOrchestrator,
    RequirementsEngineeringAgent,
    SdlcSelectionAgent,
)
from app.services.sdlc import SDLC_FACTOR_NAMES


def _input() -> RequirementsAgentInput:
    return RequirementsAgentInput(
        project_id="synthetic-project",
        functionality="Patient Appointment Scheduling",
        context=ProjectContext(
            objective="Avoid duplicate bookings.",
            stakeholders="Patients and clinic schedulers.",
            constraints="Use synthetic data for demonstrations.",
        ),
        evidence=[
            RetrievedChunk(
                id="chunk-001",
                text="The scheduling service shall prevent double booking for each clinician and record every decision in an auditable event log.",
                document_id="DOC-001",
                document_name="synthetic_policy.txt",
                relevance=0.91,
            )
        ],
    )


def _requirement(source_document: str = "DOC-001") -> Requirement:
    return Requirement(
        id="REQ-001",
        statement="The scheduling service shall prevent double booking for each clinician.",
        categories=[Category.FUNCTIONAL, Category.AVAILABILITY],
        evidence=[
            Evidence(
                document_id=source_document,
                document_name="synthetic_policy.txt",
                chunk_id="chunk-001",
                excerpt="The scheduling service shall prevent double booking for each clinician.",
                relevance=0.91,
            )
        ],
        business_justification="Prevents conflicting appointment assignments.",
        priority="Must",
        acceptance_criteria=["Given a booked slot, a second booking for that slot is rejected."],
        confidence=0.85,
        reasoning="Directly supported by the retrieved scheduling policy.",
        approval_status="Needs review",
    )


def _agent1_output() -> RequirementsAgentOutput:
    return RequirementsAgentOutput(
        requirements=[_requirement()],
        quality_issues=[
            QualityIssue(
                id="QI-001",
                requirement_id="REQ-001",
                check="Lack of testability",
                severity="Medium",
                message="Synthetic finding for handoff validation.",
                recommendation="Add measurable acceptance criteria.",
            )
        ],
        generation_mode="llm",
        retrieved_chunk_count=1,
    )


def test_requirements_agent_uses_llm_and_validates_retrieved_citations(monkeypatch):
    monkeypatch.setattr(agents.llm, "enabled", lambda: True)
    monkeypatch.setattr(agents.llm, "generate_requirements", lambda *_args: [_requirement()])

    output = RequirementsEngineeringAgent().run(_input())

    assert output.generation_mode == "llm"
    assert output.validation_passed
    assert output.requirements[0].id == "REQ-001"
    assert output.requirements[0].evidence[0].chunk_id == "chunk-001"


def test_requirements_agent_falls_back_on_invalid_llm_citation(monkeypatch):
    monkeypatch.setattr(agents.llm, "enabled", lambda: True)
    monkeypatch.setattr(agents.llm, "generate_requirements", lambda *_args: [_requirement("DOC-OTHER")])

    output = RequirementsEngineeringAgent().run(_input())

    assert output.generation_mode == "fallback"
    assert output.requirements
    assert all(
        source.document_id == "DOC-001"
        for requirement in output.requirements
        for source in requirement.evidence
    )


def test_lims_fallback_extracts_atomic_cited_requirements_and_project_activities(monkeypatch):
    source = RetrievedChunk(
        id="chunk-lims-001",
        text=(
            "Workflow: We need the LIMS to receive specimens, validate specimen identifiers, "
            "track processing status, and notify laboratory staff. Before go-live, the team "
            "should conduct user acceptance testing. After launch, the operations team should "
            "review failed sample handoffs weekly. Laboratory staff need training before deployment."
        ),
        document_id="DOC-LIMS",
        document_name="Stakeholder interview",
        relevance=0.94,
    )
    agent_input = RequirementsAgentInput(
        project_id="synthetic-lims",
        functionality="Hospital Laboratory Information Management System (LIMS)",
        evidence=[source],
    )
    monkeypatch.setattr(agents.llm, "enabled", lambda: False)

    output = RequirementsEngineeringAgent().run(agent_input)

    assert output.generation_mode == "fallback"
    assert output.validation_passed
    assert len(output.requirements) == 4
    assert all(requirement.acceptance_criteria for requirement in output.requirements)
    assert all(
        requirement.evidence[0].document_id == "DOC-LIMS"
        and requirement.evidence[0].chunk_id == "chunk-lims-001"
        for requirement in output.requirements
    )
    assert {activity.category for activity in output.project_activities} == {
        "Testing", "Operations", "Rollout"
    }
    assert not any(
        any(term in requirement.statement.lower() for term in ("test", "review", "training", "deploy"))
        for requirement in output.requirements
    )


def test_mixed_lims_and_appointment_evidence_is_excluded_and_flagged(monkeypatch):
    agent_input = RequirementsAgentInput(
        project_id="synthetic-mixed-lims",
        functionality="Hospital Laboratory Information Management System (LIMS)",
        evidence=[
            RetrievedChunk(
                id="chunk-lims-001",
                text="The LIMS shall receive specimens and track sample status.",
                document_id="DOC-LIMS",
                document_name="LIMS interview",
                relevance=0.96,
            ),
            RetrievedChunk(
                id="chunk-appointments-001",
                text=(
                    "The scheduling service shall allow patients to book appointments "
                    "and reschedule appointments."
                ),
                document_id="DOC-MIXED",
                document_name="Stakeholder interview",
                relevance=0.95,
            ),
        ],
    )
    monkeypatch.setattr(agents.llm, "enabled", lambda: False)

    output = RequirementsEngineeringAgent().run(agent_input)

    assert output.requirements
    assert all("appointment" not in requirement.statement.lower() for requirement in output.requirements)
    assert all(
        source.document_id == "DOC-LIMS"
        for requirement in output.requirements
        for source in requirement.evidence
    )
    assert not output.validation_passed
    assert any("conflicts" in gap for gap in output.analysis_gaps)
    mismatch = next(
        issue for issue in output.quality_issues
        if issue.check == "Project-domain inconsistency"
    )
    assert mismatch.severity == "High"
    assert "DOC-MIXED/chunk-appointments-001" in mismatch.message


def test_requirements_agent_logs_sanitized_llm_failure_before_fallback(monkeypatch, caplog):
    monkeypatch.setattr(agents.llm, "enabled", lambda: True)

    def fail_generation(*_args):
        raise RuntimeError("The LLM provider returned HTTP 401 (request ID synthetic-request-id).")

    monkeypatch.setattr(agents.llm, "generate_requirements", fail_generation)

    output = RequirementsEngineeringAgent().run(_input())

    assert output.generation_mode == "fallback"
    assert output.requirements
    assert "HTTP 401" in caplog.text
    assert "synthetic-request-id" in caplog.text
    assert "Bearer " not in caplog.text


def test_sdlc_agent_receives_agent1_requirements_and_returns_ranked_models(monkeypatch):
    captured: dict = {}

    def fake_analysis(requirements, _context, factor_names, _quality):
        captured["requirements"] = requirements
        captured["quality"] = _quality
        assert factor_names == SDLC_FACTOR_NAMES
        return [
            {
                "name": name,
                "value": "Requirement evidence indicates elevated need",
                "score": 4,
                "rationale": "Assessed from the supplied Agent 1 requirement set.",
                "requirement_ids": ["REQ-001"],
            }
            for name in factor_names
        ]

    monkeypatch.setattr(agents.llm, "enabled", lambda: True)
    monkeypatch.setattr(agents.llm, "analyse_sdlc", fake_analysis)
    first_output = _agent1_output()

    output = AgentOrchestrator().run_sdlc(first_output, ProjectContext())

    assert captured["requirements"] == first_output.requirements
    assert captured["quality"] == first_output.quality_issues
    assert output.recommendation.analysis_mode == "llm-assisted"
    assert output.recommendation.input_requirement_ids == ["REQ-001"]
    assert len(output.recommendation.options) == 7
    assert output.recommendation.options[0].score >= output.recommendation.options[-1].score
    assert all(option.score_breakdown for option in output.recommendation.options)


def test_sdlc_agent_uses_rules_fallback_for_incomplete_llm_output(monkeypatch):
    monkeypatch.setattr(agents.llm, "enabled", lambda: True)
    monkeypatch.setattr(agents.llm, "analyse_sdlc", lambda *_args: [])

    output = SdlcSelectionAgent().run(
        agents.SdlcAgentInput(
            requirements_agent_output=_agent1_output(),
            context=ProjectContext(),
        )
    )

    assert output.recommendation.analysis_mode == "rules-fallback"
    assert output.recommendation.input_requirement_ids == ["REQ-001"]
    assert len(output.recommendation.factors) == len(SDLC_FACTOR_NAMES)


def test_sdlc_api_contract_rejects_missing_agent1_requirement_ids(monkeypatch):
    monkeypatch.setattr(agents.llm, "enabled", lambda: False)
    recommendation = SdlcSelectionAgent().run(
        agents.SdlcAgentInput(
            requirements_agent_output=_agent1_output(),
            context=ProjectContext(),
        )
    ).recommendation.model_dump(mode="json")
    recommendation.pop("input_requirement_ids")

    with pytest.raises(ValidationError) as error:
        SdlcApiResponse.model_validate(recommendation)

    assert any(item["loc"] == ("input_requirement_ids",) for item in error.value.errors())


def test_sdlc_agent_rejects_unvalidated_or_missing_agent1_output():
    output = _agent1_output().model_copy(update={"validation_passed": False})
    try:
        SdlcSelectionAgent().run(
            agents.SdlcAgentInput(requirements_agent_output=output, context=ProjectContext())
        )
    except ValueError as exc:
        assert "validated Agent 1 output" in str(exc)
    else:
        raise AssertionError("Agent 2 must reject unvalidated Agent 1 output.")


def test_llm_client_uses_configured_chat_model_and_system_prompt(monkeypatch):
    captured = {}
    monkeypatch.setenv("LLM_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        response = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        return BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(llm_service, "urlopen", fake_urlopen)

    result = llm_service._post_chat("synthetic user content", system_prompt="synthetic agent instruction")

    assert result == {"ok": True}
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "synthetic agent instruction"},
        {"role": "user", "content": "synthetic user content"},
    ]
    assert captured["authorization"] == "Bearer synthetic-test-key"
    assert "synthetic-test-key" not in str(result)


def test_llm_parser_accepts_fenced_json_and_reports_invalid_content():
    response = {
        "choices": [{
            "message": {"content": '```json\n{"requirements": []}\n```'}
        }]
    }

    assert llm_service._json_content(json.dumps(response).encode()) == {"requirements": []}
    with pytest.raises(RuntimeError, match="not valid JSON"):
        llm_service._json_content(
            b'{"choices":[{"message":{"content":"not structured output"}}]}'
        )


def test_llm_provider_error_does_not_expose_key_or_error_body(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "synthetic-secret-key")

    def rejected_request(*_args, **_kwargs):
        raise llm_service.HTTPError(
            "https://example.invalid/v1/chat/completions",
            401,
            "Unauthorized",
            {"x-request-id": "synthetic-request-id"},
            BytesIO(b"synthetic-secret-key must not be logged"),
        )

    monkeypatch.setattr(llm_service, "urlopen", rejected_request)
    with pytest.raises(RuntimeError) as error:
        llm_service._post_chat("synthetic prompt")

    assert "HTTP 401" in str(error.value)
    assert "synthetic-request-id" in str(error.value)
    assert "synthetic-secret-key" not in str(error.value)


def test_api_runs_both_agents_preserving_approval_and_export(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import main
    from app.services.storage import ProjectStore

    class FakeVectorStore:
        last_query = ""

        def configuration(self):
            return {"vector_database": "ChromaDB", "embedding_model": "local test vectors"}

        def index(self, _project_id, documents):
            return sum(len(document["chunks"]) for document in documents)

        def retrieve(self, _project_id, _query, limit=6):
            self.last_query = _query
            return [{
                "id": "chunk-001",
                "text": "The scheduling service shall prevent double booking for each clinician and record every decision in an auditable event log.",
                "document_id": "DOC-001",
                "document_name": "Stakeholder interview",
                "relevance": 0.9,
            }][:limit]

    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "chroma"))
    fake_vector = FakeVectorStore()
    monkeypatch.setattr(main, "store", ProjectStore())
    monkeypatch.setattr(main, "vector_store", fake_vector)
    monkeypatch.setattr(main, "agent_orchestrator", AgentOrchestrator())
    monkeypatch.setattr(agents.llm, "enabled", lambda: False)

    with TestClient(main.app) as client:
        created = client.post(
            "/api/projects",
            data={
                "functionality": "Patient Appointment Scheduling",
                "stakeholder_input": '{"workflow":"Patients and schedulers use the appointment process."}',
                "project_context": (
                    '{"objective":"Avoid duplicate bookings.",'
                    '"stakeholders":"Patients and clinic schedulers.",'
                    '"constraints":"Use synthetic data.",'
                    '"limitations":"Prototype only.",'
                    '"interview_responses":{"workflow":"Patients use the scheduling service."}}'
                ),
            },
        )
        assert created.status_code == 200
        project_id = created.json()["project_id"]

        generated = client.post(f"/api/projects/{project_id}/requirements")
        assert generated.status_code == 200
        assert "Avoid duplicate bookings." in fake_vector.last_query
        assert "Use synthetic data." in fake_vector.last_query
        agent1 = generated.json()
        assert agent1["agent_runs"]["requirements"]["agent_name"] == "Requirements Engineering Agent"
        assert agent1["agent_runs"]["requirements"]["mode"] == "fallback"
        assert agent1["requirements"]

        selected = client.post(f"/api/projects/{project_id}/sdlc")
        assert selected.status_code == 200
        agent2 = selected.json()
        assert agent2["analysis_mode"] == "rules-fallback"
        assert agent2["input_requirement_ids"] == [item["id"] for item in agent1["requirements"]]
        saved = client.get(f"/api/projects/{project_id}").json()
        assert saved["agent_runs"]["sdlc"]["agent_name"] == "SDLC Selection Agent"
        assert saved["agent_runs"]["sdlc"]["requirement_count"] == len(agent1["requirements"])

        requirement_id = agent1["requirements"][0]["id"]
        approval = client.post(
            f"/api/projects/{project_id}/requirements/{requirement_id}/approval",
            data={"status": "Approved"},
        )
        assert approval.status_code == 200
        reopened = client.get(f"/api/projects/{project_id}").json()
        assert reopened["sdlc"] is None
        assert "sdlc" not in reopened["agent_runs"]

        exported = client.get(f"/api/projects/{project_id}/export")
        assert exported.status_code == 200
        assert exported.json()["agent_runs"]["requirements"]["mode"] == "fallback"


def test_api_reports_evidence_gap_and_rejects_sdlc_without_requirements(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app import main
    from app.services.storage import ProjectStore

    class ActivityOnlyVectorStore:
        def configuration(self):
            return {"vector_database": "ChromaDB", "embedding_model": "local test vectors"}

        def index(self, _project_id, documents):
            self.documents = documents
            return sum(len(document["chunks"]) for document in documents)

        def retrieve(self, _project_id, _query, limit=6):
            document = self.documents[0]
            return [{
                **document["chunks"][0],
                "document_id": document["id"],
                "document_name": document["name"],
                "relevance": 0.9,
            }][:limit]

    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(main, "store", ProjectStore())
    monkeypatch.setattr(main, "vector_store", ActivityOnlyVectorStore())
    monkeypatch.setattr(agents.llm, "enabled", lambda: False)

    with TestClient(main.app) as client:
        created = client.post(
            "/api/projects",
            data={
                "functionality": "Synthetic appointment workflow",
                "stakeholder_input": '{"rollout":"The team should test the pilot before release."}',
            },
        )
        assert created.status_code == 200
        project_id = created.json()["project_id"]

        generated = client.post(f"/api/projects/{project_id}/requirements")
        assert generated.status_code == 200
        payload = generated.json()
        assert payload["requirements"] == []
        assert payload["analysis_gaps"]
        assert payload["project_activities"]
        assert payload["agent_runs"]["requirements"]["validation_passed"] is False

        selection = client.post(f"/api/projects/{project_id}/sdlc")
        assert selection.status_code == 422
        assert "Generate requirements" in selection.json()["detail"]

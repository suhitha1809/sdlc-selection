import json
from app.schemas import Category
from app.services.documents import chunk_text, mask_sensitive, retrieve
from app.services.generator import extract_project_activities, generate_evidence_drafts
from app.services.quality import analyse
from io import BytesIO
from urllib.error import HTTPError
from app.services import rag
from app.services.rag import RagConfigurationError, VectorStore, _api_key, _embed, provider_configured
from app.schemas import Evidence, ProjectContext, QualityIssue, Requirement, SdlcFactor
from app.services.sdlc import SDLC_FACTOR_NAMES, fallback_factors, recommend
import pytest
def test_pipeline_creates_grounded_requirements_and_recommendation():
    text = "The system shall allow patients to book appointments with doctors. The system must prevent double booking and record every scheduling decision in an audit log."
    chunks = [chunk.__dict__ for chunk in chunk_text(mask_sensitive(text))]
    documents = [{"id": "DOC-001", "name": "appointment_policy.txt", "chunks": chunks}]
    evidence = retrieve("appointment scheduling and audit log", documents)

    requirements = generate_evidence_drafts("Patient Appointment Scheduling", evidence)

    assert requirements
    assert all(requirement.evidence for requirement in requirements)
    assert Category.SECURITY in requirements[0].categories or Category.PRIVACY in requirements[0].categories or Category.AVAILABILITY in requirements[0].categories
    assert isinstance(analyse(requirements), list)
    recommendation = recommend(requirements)
    assert recommendation.options[0].score >= recommendation.options[-1].score
    assert {option.name for option in recommendation.options} == {"Waterfall", "V-Shape", "Prototyping", "RAD", "Spiral", "Incremental", "Agile"}
    assert {factor.name for factor in recommendation.factors} == {
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
    }


def test_masking_replaces_healthcare_sensitive_values():
    source = (
        "Patient name: Jane Doe, Patient ID: P-123456, MRN: MRN-98765, "
        "DOB 1990-05-12, Dr. Alex Smith, phone 555-123-4567, "
        "email jane.doe@example.com. Clinical note: diabetes."
    )

    masked = mask_sensitive(source)

    assert "[MASKED_PATIENT_NAME]" in masked
    assert "[MASKED_MRN]" in masked
    assert "[MASKED_DOB]" in masked
    assert "[MASKED_PHONE]" in masked
    assert "[MASKED_EMAIL]" in masked
    assert "Jane Doe" not in masked
    assert "Alex Smith" not in masked
    assert "P-123456" not in masked
    assert "MRN-98765" not in masked
    assert "1990-05-12" not in masked
    assert "555-123-4567" not in masked
    assert "jane.doe@example.com" not in masked
    assert "diabetes" in masked


def test_masking_does_not_mask_project_titles_or_ordinary_phrases():
    source = "Patient Appointment Scheduling. Patient name: Jane Doe."

    masked = mask_sensitive(source)

    assert "Patient Appointment Scheduling" in masked
    assert "Jane Doe" not in masked
    assert "[MASKED_PATIENT_NAME]" in masked


def test_chunking_splits_long_multiline_source_into_bounded_chunks():
    chunks = chunk_text(
        "First documented behavior. " + "second " * 120 + "\n"
        "Another independent behavior.",
        size=80,
    )

    assert len(chunks) > 2
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert "First documented behavior." in chunks[0].text
    assert chunks[-1].text == "Another independent behavior."


def test_fallback_decomposes_conversation_and_keeps_activities_separate():
    source = {
        "id": "chunk-001",
        "text": (
            "Workflow: We need the system to let patients book appointments and notify clinic staff, "
            "and record each booking. The team should test the workflow before release. "
            "Patients should take medication as prescribed."
        ),
        "document_id": "DOC-001",
        "document_name": "Stakeholder interview",
        "relevance": 0.9,
    }

    requirements = generate_evidence_drafts("Appointment scheduling", [source])
    activities = extract_project_activities([source])

    assert len(requirements) == 3
    assert all(item.evidence[0].chunk_id == "chunk-001" for item in requirements)
    assert any("book appointments" in item.statement for item in requirements)
    assert any("notify clinic staff" in item.statement for item in requirements)
    assert any("record each booking" in item.statement for item in requirements)
    assert all("take medication" not in item.statement for item in requirements)
    assert len(activities) == 1
    assert activities[0].category == "Testing"


def test_fallback_does_not_invent_requirement_from_clinical_or_activity_text():
    source = {
        "id": "chunk-001",
        "text": "The care team should administer medication as prescribed. The team should test the pilot.",
        "document_id": "DOC-001",
        "document_name": "Stakeholder interview",
        "relevance": 0.8,
    }

    assert generate_evidence_drafts("Clinical workflow", [source]) == []
    assert len(extract_project_activities([source])) == 1


def test_masking_does_not_apply_legacy_financial_patterns():
    source = "Example card number 4111111111111111 and identifier ABCDE1234F."

    assert mask_sensitive(source) == source


def test_rag_requires_explicit_provider_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    try:
        _api_key()
    except RagConfigurationError:
        return
    raise AssertionError("RAG must not run without an embedding-provider key")


def test_rag_configuration_never_exposes_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "not-for-the-client")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    configuration = VectorStore.configuration()

    assert configuration["vector_database"] == "ChromaDB"
    assert configuration["llm_model"] == "test-model"
    assert "not-for-the-client" not in str(configuration)


def test_rag_uses_local_embeddings_without_provider_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    vectors = _embed(["identity verification requires an audit record"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert any(vectors[0])


def test_rag_falls_back_to_local_embeddings_when_provider_rejects_request(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def rejected_request(*_args, **_kwargs):
        raise HTTPError("https://example.invalid/v1/embeddings", 404, "Not Found", {}, BytesIO())

    monkeypatch.setattr(rag, "urlopen", rejected_request)

    vectors = rag._embed(["Identity verification requires an audit record."])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert any(vectors[0])


def test_rag_filters_irrelevant_cosine_matches_in_isolated_chroma(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "chroma"))
    store = VectorStore()
    store.index("synthetic-rag-test", [{
        "id": "DOC-001",
        "name": "synthetic.txt",
        "chunks": [
            {"id": "chunk-001", "text": "Patient appointment scheduling prevents double booking."},
            {"id": "chunk-002", "text": "Volcanic weather satellite orbit measurements."},
        ],
    }])

    matches = store.retrieve("synthetic-rag-test", "Patient appointment scheduling")

    assert matches
    assert matches[0]["id"] == "chunk-001"
    assert all(item["relevance"] >= rag.MIN_RELEVANCE for item in matches)


def test_openai_api_key_environment_variable_is_supported(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert provider_configured()
    assert _api_key() == "test-key"


def test_project_creation_works_without_provider_key(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.services.storage import ProjectStore

    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "projects"))
    from app import main

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "vector_store", VectorStore())
    monkeypatch.setattr(main, "store", ProjectStore())
    client = TestClient(main.app)
    response = client.post(
        "/api/projects",
        data={"functionality": "Patient Appointment Scheduling", "stakeholder_input": '{"workflow": "Patients can book and reschedule appointments while clinicians maintain availability."}'},
    )

    assert response.status_code == 200
    assert response.json()["rag"]["vector_database"] == "ChromaDB"


def test_fallback_factors_reflect_context_and_agent1_findings():
    requirement = Requirement(
        id="REQ-001",
        statement="The scheduling service shall prevent double booking for each clinician.",
        categories=[Category.FUNCTIONAL],
        evidence=[Evidence(
            document_id="DOC-001",
            document_name="synthetic.txt",
            chunk_id="chunk-001",
            excerpt="The scheduling service shall prevent double booking for each clinician.",
            relevance=0.9,
        )],
        business_justification="Prevents conflicting bookings.",
        confidence=0.9,
        reasoning="Directly evidenced.",
        acceptance_criteria=["A duplicate booking is rejected."],
    )
    stable_context = ProjectContext(
        interview_responses={
            "Delivery": "Requirements are stable and fixed. Stakeholders review weekly. Delivery is due in two months."
        }
    )
    changing_context = ProjectContext(
        interview_responses={
            "Delivery": "Requirements change frequently. Stakeholders review weekly. Build a prototype early."
        }
    )

    stable = {item.name: item for item in fallback_factors([requirement], stable_context)}
    changing = {item.name: item for item in fallback_factors([requirement], changing_context)}

    assert stable["Requirement stability"].score == 4
    assert stable["Change volatility"].score == 2
    assert stable["Time pressure"].score == 4
    assert stable["Stakeholder availability"].score == 4
    assert changing["Requirement stability"].score == 2
    assert changing["Change volatility"].score == 4
    assert changing["Iteration need"].score == 4
    with_quality_findings = fallback_factors(
        [requirement],
        stable_context,
        [QualityIssue(
            id="QI-001",
            requirement_id="REQ-001",
            check="Lack of testability",
            severity="Medium",
            message="Acceptance evidence is incomplete.",
            recommendation="Add measurable criteria.",
        )],
    )
    iteration = next(item for item in with_quality_findings if item.name == "Iteration need")
    assert iteration.score == 4
    assert iteration.value == "Agent 1 clarity findings suggest iteration"


def test_recommendation_scoring_is_normalized_and_ties_are_stable():
    requirement = Requirement(
        id="REQ-001",
        statement="The scheduling service shall prevent double booking for each clinician.",
        categories=[Category.FUNCTIONAL],
        evidence=[Evidence(
            document_id="DOC-001",
            document_name="synthetic.txt",
            chunk_id="chunk-001",
            excerpt="The scheduling service shall prevent double booking for each clinician.",
            relevance=0.9,
        )],
        business_justification="Prevents conflicting bookings.",
        confidence=0.9,
        reasoning="Directly evidenced.",
    )
    neutral = [
        SdlcFactor(
            name=name,
            value="Insufficient evidence; neutral score used",
            score=3,
            rationale="Neutral synthetic test assessment.",
        )
        for name in SDLC_FACTOR_NAMES
    ]

    result = recommend([requirement], factors=neutral)

    assert [item.score for item in result.options] == [60] * 7
    assert [item.name for item in result.options] == sorted(item.name for item in result.options)
    assert result.recommended_score == result.options[0].score
    assert "tied" in result.justification
    assert result.input_requirement_ids == ["REQ-001"]


def test_quality_checks_project_activities_and_avoids_generic_security_findings():
    requirement = Requirement(
        id="REQ-001",
        statement="The scheduling service shall allow patients to book appointments.",
        categories=[Category.FUNCTIONAL],
        evidence=[Evidence(
            document_id="DOC-001",
            document_name="synthetic.txt",
            chunk_id="chunk-001",
            excerpt="The scheduling service shall allow patients to book appointments.",
            relevance=0.9,
        )],
        business_justification="Supports appointment scheduling.",
        confidence=0.9,
        reasoning="Directly evidenced.",
        acceptance_criteria=["A patient can book an available appointment."],
    )

    issues = analyse([requirement])

    assert not any("security" in item.message.lower() for item in issues)
    assert not any(item.check == "Project activity, not software requirement" for item in issues)


@pytest.mark.parametrize(
    ("functionality", "answer", "expected_factors"),
    [
        (
            "Patient Appointment Scheduling",
            "The scheduling service shall allow patients to book appointments and prevent double bookings. "
            "Requirements are stable and fixed. Stakeholders review weekly. Delivery is due in two months.",
            {"Requirement stability": 4, "Change volatility": 2, "Time pressure": 4},
        ),
        (
            "Laboratory Management System",
            "The laboratory platform shall verify specimen identifiers and integrate with the legacy lab system. "
            "Requirements change frequently and the system has several dependencies.",
            {"Requirement stability": 2, "Project complexity": 4},
        ),
        (
            "Telemedicine Platform",
            "The telemedicine platform shall start video sessions and notify clinicians when a connection fails. "
            "Stakeholders review weekly, want an early prototype, and requirements change frequently.",
            {"Iteration need": 4, "Stakeholder availability": 4},
        ),
        (
            "Hospital Pharmacy Inventory",
            "The inventory system shall track stock levels and alert pharmacy staff when stock is low. "
            "Requirements are stable and fixed. The system is standalone with no external dependencies. "
            "Delivery is due in eight months.",
            {"Project complexity": 2, "Dependency coupling": 2, "Time pressure": 4},
        ),
        (
            "Emergency Intake Management",
            "The emergency intake system shall register incoming requests and restrict access to patient data. "
            "Requirements change frequently and access must be audited.",
            {"Requirement stability": 2, "Patient-data sensitivity": 5},
        ),
    ],
)
def test_synthetic_healthcare_projects_complete_two_agent_api_flow(
    monkeypatch, tmp_path, functionality, answer, expected_factors
):
    from fastapi.testclient import TestClient

    from app import main
    from app.services.agents import AgentOrchestrator
    from app.services.storage import ProjectStore

    class SyntheticVectorStore:
        documents = []

        def configuration(self):
            return {"vector_database": "ChromaDB", "embedding_model": "local synthetic vectors"}

        def index(self, _project_id, documents):
            self.documents = documents
            return sum(len(document["chunks"]) for document in documents)

        def retrieve(self, _project_id, _query, limit=6):
            results = []
            for document in self.documents:
                for chunk in document["chunks"]:
                    results.append({
                        **chunk,
                        "document_id": document["id"],
                        "document_name": document["name"],
                        "relevance": 0.9,
                    })
            return results[:limit]

    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "chroma"))
    vector_store = SyntheticVectorStore()
    monkeypatch.setattr(main, "store", ProjectStore())
    monkeypatch.setattr(main, "vector_store", vector_store)
    monkeypatch.setattr(main, "agent_orchestrator", AgentOrchestrator())
    monkeypatch.setattr("app.services.agents.llm.enabled", lambda: False)

    context = {
        "objective": f"Synthetic operational workflow for {functionality}; no real patient data.",
        "stakeholders": "Synthetic patients, staff, and administrators.",
        "interview_responses": {"workflow and outcomes": answer},
    }
    with TestClient(main.app) as client:
        created = client.post(
            "/api/projects",
            data={
                "functionality": functionality,
                "stakeholder_input": "{}",
                "project_context": json.dumps(context),
            },
        )
        assert created.status_code == 200
        project_id = created.json()["project_id"]

        agent1_response = client.post(f"/api/projects/{project_id}/requirements")
        assert agent1_response.status_code == 200
        agent1 = agent1_response.json()
        assert agent1["agent_runs"]["requirements"]["agent_name"] == "Requirements Engineering Agent"
        assert agent1["agent_runs"]["requirements"]["validation_passed"] is True
        assert agent1["requirements"]
        assert all(requirement["evidence"] for requirement in agent1["requirements"])

        agent2_response = client.post(f"/api/projects/{project_id}/sdlc")
        assert agent2_response.status_code == 200
        agent2 = agent2_response.json()
        assert agent2["input_requirement_ids"] == [item["id"] for item in agent1["requirements"]]
        assert agent2["recommended_score"] == agent2["options"][0]["score"]
        assert len(agent2["factors"]) == len(SDLC_FACTOR_NAMES)
        factors = {item["name"]: item for item in agent2["factors"]}
        for name, score in expected_factors.items():
            assert factors[name]["score"] == score

        persisted = client.get(f"/api/projects/{project_id}").json()
        assert persisted["sdlc"]["recommended_sdlc"] == agent2["recommended_sdlc"]
        assert persisted["agent_runs"]["sdlc"]["agent_name"] == "SDLC Selection Agent"
        assert client.get(f"/api/projects/{project_id}/export").status_code == 200

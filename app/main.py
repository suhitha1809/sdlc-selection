from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.schemas import (
    Project,
    ProjectContext,
    RequirementsAgentInput,
    RequirementsAgentOutput,
    SdlcApiResponse,
)
from app.services.documents import chunk_text, extract_text, mask_sensitive
from app.services.agents import (
    AgentOrchestrator,
    MissingRetrievedEvidence,
    RequirementsEngineeringAgent,
    SdlcSelectionAgent,
)
from app.services.rag import RagConfigurationError, VectorStore, provider_configured
from app.services.storage import ProjectStore

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")
store = ProjectStore()
vector_store = VectorStore()
agent_orchestrator = AgentOrchestrator()


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("\n" + "=" * 56)
    print("  HealthReq Studio is running")
    print("  App:      http://127.0.0.1:8000")
    print("  API docs: http://127.0.0.1:8000/docs")
    print("  RAG:      ChromaDB + OpenAI-compatible embeddings")
    print("  Storage:  local JSON project data and ChromaDB vectors")
    print("  Stop:     press Ctrl+C")
    print("=" * 56 + "\n")
    yield
    print("HealthReq Studio stopped.")


app = FastAPI(title="HealthReq Studio", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def audit(project: Project, action: str, detail: str) -> None:
    project.audit_log.append({"at": datetime.now(timezone.utc).isoformat(), "action": action, "detail": detail})


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "templates" / "index.html")


@app.get("/api/rag/status")
def rag_status() -> JSONResponse:
    return JSONResponse({
        **vector_store.configuration(),
        "embedding_provider_configured": provider_configured(),
        "generation_mode": "LLM configured; each agent reports its actual run mode" if provider_configured() else "Both agents use deterministic fallback until an LLM key is configured",
        "agents": ["Requirements Engineering Agent", "SDLC Selection Agent"],
    })


@app.post("/api/projects")
async def create_project(
    functionality: str = Form(...),
    stakeholder_input: str = Form("{}"),
    project_context: str = Form("{}"),
    files: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    functionality = mask_sensitive(functionality.strip())
    if len(functionality) < 3:
        raise HTTPException(422, "Enter a healthcare functionality of at least three characters.")
    try:
        answers = json.loads(stakeholder_input)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "Stakeholder answers could not be processed.") from exc
    if not isinstance(answers, dict):
        raise HTTPException(422, "Stakeholder answers must be a structured set of responses.")
    answers = {str(question): str(answer).strip() for question, answer in answers.items() if str(answer).strip()}
    try:
        context_data = json.loads(project_context)
        if not isinstance(context_data, dict):
            raise ValueError("Project context must be a structured object.")
        submitted_context = ProjectContext.model_validate(context_data)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        raise HTTPException(422, "Project context could not be processed.") from exc
    response_data = submitted_context.interview_responses or answers
    response_data = {
        str(question): str(answer).strip()
        for question, answer in response_data.items()
        if str(answer).strip()
    }
    context = ProjectContext(
        objective=mask_sensitive(submitted_context.objective or answers.get("business objective", "")),
        stakeholders=mask_sensitive(submitted_context.stakeholders or answers.get("users and roles", "")),
        constraints=mask_sensitive(submitted_context.constraints),
        limitations=mask_sensitive(submitted_context.limitations),
        interview_responses={
            str(question): mask_sensitive(str(answer).strip())
            for question, answer in response_data.items()
            if str(answer).strip()
        },
    )
    uploads = [upload for upload in files if upload.filename]
    if not response_data and not uploads:
        raise HTTPException(422, "Answer at least one stakeholder question or upload a supporting document.")
    documents = []
    if response_data:
        interview_chunks = []
        for question, answer in response_data.items():
            masked_answer = mask_sensitive(f"{question}: {answer}")
            for chunk in chunk_text(masked_answer):
                interview_chunks.append({
                    "id": f"chunk-{len(interview_chunks) + 1:03d}",
                    "text": chunk.text,
                })
        documents.append({"id": "DOC-001", "name": "Stakeholder interview", "chunks": interview_chunks})
    for upload in uploads:
        content = await upload.read()
        if len(content) > 5_000_000:
            raise HTTPException(413, f"{upload.filename} exceeds the 5 MB prototype limit.")
        try:
            text = mask_sensitive(extract_text(upload.filename, content))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not text.strip():
            raise HTTPException(422, f"{upload.filename} contains no extractable text.")
        doc_id = f"DOC-{len(documents)+1:03d}"
        documents.append({"id": doc_id, "name": upload.filename, "chunks": [chunk.__dict__ for chunk in chunk_text(text)]})
    project = Project(
        id=uuid4().hex[:12],
        functionality=functionality,
        created_at=datetime.now(timezone.utc),
        context=context,
        documents=documents,
    )
    try:
        indexed_chunks = vector_store.index(project.id, documents)
    except RagConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc
    audit(project, "sources_recorded", f"Stored {len(documents)} masked source(s): {len(answers)} stakeholder response(s) and {len(uploads)} uploaded document(s).")
    audit(project, "rag_indexed", f"Indexed {indexed_chunks} source chunk(s) in ChromaDB.")
    store.save(project)
    return JSONResponse({
        "project_id": project.id,
        "documents": [{"id": doc["id"], "name": doc["name"], "chunks": len(doc["chunks"])} for doc in documents],
        "rag": {**vector_store.configuration(), "indexed_chunks": indexed_chunks},
    })


@app.get("/api/projects/{project_id}/retrieval")
def retrieve_evidence(project_id: str, query: str | None = None) -> JSONResponse:
    try:
        project = store.get(project_id)
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc
    search_query = (query or project.functionality).strip()
    if len(search_query) < 3:
        raise HTTPException(422, "Enter a retrieval query of at least three characters.")
    try:
        evidence = vector_store.retrieve(project.id, search_query)
    except RagConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc
    return JSONResponse({"query": search_query, "matches": evidence, "count": len(evidence)})


@app.post("/api/projects/{project_id}/requirements")
def generate_requirements(project_id: str) -> JSONResponse:
    try:
        project = store.get(project_id)
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc
    try:
        agent1_output = agent_orchestrator.run_requirements(RequirementsAgentInput(
            project_id=project.id,
            functionality=project.functionality,
            context=project.context,
        ), vector_store)
    except RagConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc
    except MissingRetrievedEvidence as exc:
        raise HTTPException(422, "No retrievable evidence was found in the uploaded documents.") from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(502, "The Requirements Engineering Agent could not produce validated output.") from exc
    project.requirements = agent1_output.requirements
    project.project_activities = agent1_output.project_activities
    project.analysis_gaps = agent1_output.analysis_gaps
    project.quality_issues = agent1_output.quality_issues
    project.agent_runs["requirements"] = RequirementsEngineeringAgent.execution(agent1_output)
    project.sdlc = None
    project.agent_runs.pop("sdlc", None)
    audit(
        project,
        "requirements_agent_completed",
        f"Requirements Engineering Agent generated {len(project.requirements)} validated draft(s) in {agent1_output.generation_mode} mode using {agent1_output.retrieved_chunk_count} retrieved chunk(s).",
    )
    store.save(project)
    return JSONResponse(project.model_dump(mode="json"))


@app.post("/api/projects/{project_id}/requirements/{requirement_id}/approval")
def update_approval(project_id: str, requirement_id: str, status: str = Form(...)) -> JSONResponse:
    if status not in {"Approved", "Rejected", "Needs review", "Draft"}:
        raise HTTPException(422, "Invalid approval status.")
    try:
        project = store.get(project_id)
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc
    requirement = next((item for item in project.requirements if item.id == requirement_id), None)
    if not requirement:
        raise HTTPException(404, "Requirement not found.")
    requirement.approval_status = status
    project.sdlc = None
    project.agent_runs.pop("sdlc", None)
    audit(project, "requirement_approval_updated", f"{requirement_id} marked {status}.")
    store.save(project)
    return JSONResponse(requirement.model_dump(mode="json"))


@app.post("/api/projects/{project_id}/sdlc", response_model=SdlcApiResponse)
def sdlc_analysis(project_id: str) -> SdlcApiResponse:
    try:
        project = store.get(project_id)
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc
    if not project.requirements:
        raise HTTPException(422, "Generate requirements before SDLC analysis.")
    previous_run = project.agent_runs.get("requirements")
    if not previous_run or previous_run.agent_name != "Requirements Engineering Agent" or not previous_run.validation_passed:
        raise HTTPException(422, "Regenerate validated Agent 1 requirements before SDLC analysis.")
    source_chunk_ids = {
        (document.get("id"), chunk.get("id"))
        for document in project.documents
        for chunk in document.get("chunks", [])
    }
    if any(
        not requirement.evidence
        or any(
            (source.document_id, source.chunk_id) not in source_chunk_ids
            for source in requirement.evidence
        )
        for requirement in project.requirements
    ):
        raise HTTPException(422, "Saved requirements contain stale or unsupported source references; regenerate Agent 1 output.")

    project.sdlc = None
    project.agent_runs.pop("sdlc", None)
    audit(project, "sdlc_analysis_started", "A new Agent 2 run invalidated any prior SDLC recommendation.")
    store.save(project)

    generation_mode = previous_run.mode if previous_run and previous_run.mode in {"llm", "fallback"} else "fallback"
    evidence_count = previous_run.retrieved_chunk_count if previous_run else max(
        1,
        len({
            (source.document_id, source.chunk_id)
            for requirement in project.requirements
            for source in requirement.evidence
        }),
    )
    try:
        agent1_output = RequirementsAgentOutput(
            requirements=project.requirements,
            project_activities=project.project_activities,
            analysis_gaps=project.analysis_gaps,
            quality_issues=project.quality_issues,
            generation_mode=generation_mode,
            retrieved_chunk_count=evidence_count,
            validation_passed=True,
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, "The SDLC Selection Agent requires valid Requirements Engineering Agent output.") from exc
    try:
        agent2_output = agent_orchestrator.run_sdlc(agent1_output, project.context)
        response = SdlcApiResponse.model_validate(
            agent2_output.recommendation.model_dump(mode="python")
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(502, "The SDLC Selection Agent returned incomplete output for the Agent 1 handoff.") from exc
    project.sdlc = response
    project.agent_runs["sdlc"] = SdlcSelectionAgent.execution(agent2_output)
    audit(
        project,
        "sdlc_agent_completed",
        f"SDLC Selection Agent evaluated {len(agent1_output.requirements)} Agent 1 requirement(s) in {project.sdlc.analysis_mode} mode and recommended {project.sdlc.recommended_sdlc}.",
    )
    store.save(project)
    return response


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> JSONResponse:
    try:
        return JSONResponse(store.get(project_id).model_dump(mode="json"))
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc


@app.get("/api/projects/{project_id}/export")
def export_project(project_id: str) -> JSONResponse:
    try:
        return JSONResponse(store.export(store.get(project_id)), headers={"Content-Disposition": f'attachment; filename="healthreq-{project_id}.json"'})
    except KeyError as exc:
        raise HTTPException(404, "Project not found.") from exc

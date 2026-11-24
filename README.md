# HealthReq Studio

HealthReq Studio is a healthcare requirements engineering prototype with a sequential, two-agent workflow. It turns stakeholder responses and supporting documents into traceable draft requirements, highlights rule-based quality findings, and recommends an SDLC approach for human review.

## Highlights

- Healthcare project intake with an eight-prompt, predefined stakeholder interview; keyword branching changes question order but does not use an LLM to create questions.
- Upload and process `.txt`, `.md`, `.csv`, `.pdf`, and `.docx` source documents.
- Mask common patient, contact, and health-record identifier patterns before indexing.
- Store embeddings in a persistent, per-project ChromaDB collection.
- Inspect retrieved evidence before generating requirements.
- Generate requirements with citations to the retrieved source chunks.
- Review deterministic quality findings and approve, reject, or return requirements for review.
- Run two distinct AI agents sequentially, with per-run LLM/fallback status and a validated requirements handoff.
- Produce an explainable SDLC comparison and export the project audit trail as JSON.

## Two-Agent Architecture and RAG Flow

```text
Project intake + interview + uploaded documents
              |
       Mask selected patterns
              |
       Chunk and index in ChromaDB
              |
     Retrieve evidence for Agent 1
              |
  Agent 1: Requirements Engineering Agent
  - Context- and evidence-grounded draft requirements
  - Structured output and evidence-reference validation
  - Deterministic requirement quality checks
              |
  Validated requirements + quality findings
              |
  Agent 2: SDLC Selection Agent
  - LLM factor assessment from Agent 1 requirements
  - Transparent weighted model scoring and comparison
              |
  Human review, approval, JSON export
```

`AgentOrchestrator` in `app/services/agents.py` is the explicit handoff layer. Agent 2 receives a `RequirementsAgentOutput` containing Agent 1's actual structured requirements and quality findings; it does not select a model from the project description alone.

Agent 1 uses the existing ChromaDB retrieval pipeline and an OpenAI-compatible LLM call when configured and successful. Its structured output is checked for unique IDs and citations to retrieved chunks, then the deterministic quality checker records findings. If the key is absent or the LLM response/network/validation fails, it produces evidence-only drafts and marks the run as `fallback`.

Agent 2 asks the configured LLM to assess twelve requirement-related factors and cite Agent 1 requirement IDs. Its model rankings are then computed by a fixed, inspectable weighted scoring matrix. If factor analysis fails or is invalid, conservative rules derive the factors from Agent 1's requirements and stored project context. Each run records its actual mode (`llm`, `fallback`, `llm-assisted`, or `rules-fallback`) in the saved project. A configured key alone is not reported as proof that a live LLM call succeeded.

Every generated requirement remains a draft and requires human review. The factor and risk outputs are decision support; they are not clinical risk assessments or determinations of regulatory compliance.

The evidence-only fallback decomposes supported system behaviors, keeps testing/rollout activities separate, and returns an explicit evidence gap rather than inventing a requirement when no testable behavior is supported. Retrieval uses ChromaDB cosine similarity and filters matches below a minimum relevance threshold. Citations are checked against retrieved chunks, but neither lexical checks nor source IDs prove semantic entailment.

The quality checker is a deterministic validation component inside Agent 1, not a third agent. It checks ambiguity, missing actors, testability, compound behaviors, activities, evidence alignment, likely duplicates, and possible contradictions. It does not emit a generic security finding for every requirement. The existing `quality_analysis.txt` file is not loaded by the application.

The scoring matrix applies `abs(weight) * score` for positive weights and `abs(weight) * (6 - score)` for negative weights. It normalizes the weighted points to 0–100; these are relative fit scores, not probabilities. Unsupported factors use a neutral score and are called out as limitations. Tied scores are ordered by model name and identified in the justification. The application currently assesses twelve factors; this is the implemented schema and should not be confused with shorter factor lists in older project notes.

## Quick Start

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000.

For Windows PowerShell, from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

For the existing WSL `.venv-linux` environment, open a WSL terminal and run:

```bash
cd /mnt/c/Users/suhi1809/Downloads/software_eng_project-main/software_eng_project-main
source .venv-linux/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

To use a different port, replace `8000` in the start command, for example `--port 8080`, then open `http://127.0.0.1:8080`.

## LLM Configuration

The project works without an API key. In this local mode, it uses deterministic local embeddings for ChromaDB retrieval, creates evidence-only Agent 1 drafts, and uses rules-based Agent 2 factors. If a key is present but an API call fails or returns invalid data, the affected agent uses its fallback. The UI displays the actual mode for each completed agent run.

To enable OpenAI-compatible embeddings and both agents' LLM analysis, configure a `.env` file in the project root:

```text
LLM_API_KEY=your_new_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

APP_DATA_DIR=data
VECTOR_DB_PATH=data/chroma
```

`OPENAI_API_KEY` is also supported as an alternative to `LLM_API_KEY`. Restart the server after changing `.env`.

Keep API keys only in `.env`, which is ignored by Git. Never add a key to source code, JSON exports, screenshots, or documentation. If a key is exposed, revoke it and create a replacement immediately.

## Using the App

1. Enter the healthcare functionality, clinical objective, and stakeholders/roles.
2. Complete the adaptive interview and optionally attach policy or requirement documents.
3. Create the project. The app masks, chunks, and indexes the sources.
4. Review the retrieved evidence or use the evidence search field to query the vector index.
5. Generate evidence-grounded requirements and review their citations, confidence, risk, and acceptance criteria.
6. Review Agent 1 requirements and quality findings; record approval decisions.
7. Run Agent 2. Its panel reports how many Agent 1 requirements it consumed and links the factor assessments to requirement IDs.
8. Review the model ranking and rationale, then export the project JSON.

Try the included [patient appointment scheduling policy](sample_data/digital_onboarding_policy.txt) with the functionality `Patient Appointment Scheduling`.

## Legacy RAG Rebuild Script

`scripts/rebuild_healthcare_rag.py` is a maintenance script for legacy data. It deletes project JSON files whose contents match its legacy financial keywords and deletes matching financial ChromaDB collections before indexing a healthcare demo collection. The keyword filter is heuristic. Do not run it against a workspace containing projects or ChromaDB data you need to preserve; back up and review its effects first.

The application does not require this script for normal startup, project creation, or analysis.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/rag/status` | Safe RAG configuration and provider status; never returns secrets. |
| `POST` | `/api/projects` | Create a project and index its source chunks. |
| `GET` | `/api/projects/{project_id}/retrieval?query=...` | Retrieve the highest-ranked ChromaDB evidence chunks. |
| `POST` | `/api/projects/{project_id}/requirements` | Generate evidence-grounded draft requirements. |
| `POST` | `/api/projects/{project_id}/sdlc` | Generate the SDLC recommendation. |
| `GET` | `/api/projects/{project_id}/export` | Download the project audit trail as JSON. |

Interactive API documentation is available at http://127.0.0.1:8000/docs while the server is running.

## Project Structure

```text
app/
  main.py                 FastAPI routes and application lifecycle
  services/agents.py      Exactly two agents and their sequential orchestrator
  services/documents.py   Extraction, masking, and chunking
  services/rag.py         Embeddings, ChromaDB indexing, and retrieval
  services/llm.py         OpenAI-compatible calls for both agents
  services/quality.py     Agent 1's deterministic requirement checks
  services/sdlc.py        Agent 2 factors and transparent SDLC scoring
  services/storage.py     Local project JSON storage
  prompts/                Requirement-generation prompt templates
sample_data/              Example supporting document
tests/                    Automated checks
data/                     Local JSON projects and ChromaDB data
```

## Tests

Run the automated test suite from the project root. This command disables Python bytecode and pytest cache writes; API integration tests use temporary storage:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-linux/bin/python -m pytest -q -p no:cacheprovider
```

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `Address already in use` | A server is already using the chosen port. Stop it with `Ctrl+C`, or start this app on another port. |
| `embedding_provider_configured: false` | Add a valid replacement key to `.env`, then restart the server. Local fallback mode still works without it. |
| The configured embedding provider rejects a request | The app automatically indexes with local embeddings so project creation remains available. Verify `LLM_BASE_URL`, `EMBEDDING_MODEL`, and the provider key before relying on remote embeddings. |
| `500` while creating a project | Stop the server, ensure the `data/` directory is writable, then restart. A fresh ChromaDB collection is created for each project. |
| Browser appears frozen | In DevTools, select Resume script execution if the page is paused in the debugger, then refresh. |

## Limitations

This is a local prototype, not a clinical, compliance, or legal decision system. Use synthetic data only. Regex masking is partial and does not guarantee de-identification. The SDLC score matrix and quality checks are heuristics; the LLM can make mistakes, and citation validation does not prove semantic entailment. Retrieval relevance thresholds and lexical support checks reduce irrelevant evidence but cannot establish that a source entails a requirement. Before any production use, add authentication, access controls, encrypted durable storage, source versioning, stronger prompt-injection defenses, retention policies, monitoring, clinical safety review, and formal compliance governance.

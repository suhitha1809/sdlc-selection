let projectId = null;
let questionIndex = 0;
let questionQueue = [];
let askedQuestionKeys = new Set();
const interviewAnswers = {};
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
const INTERVIEW_QUESTION_LIMIT = 8;
const coverageQuestions = [
  { key: "business rules", topic: "Decision rules", prompt: "Which rules, limits, approvals, or eligibility conditions decide whether the workflow can continue?" },
  { key: "exceptions", topic: "Exceptions", prompt: "What should happen for failures, missing information, rejected decisions, or manual overrides? Who owns the resolution?" },
  { key: "data and integrations", topic: "Data and integrations", prompt: "Which data is collected or exchanged, which systems or third parties are involved, and what happens if an integration is unavailable?" },
  { key: "security and compliance", topic: "Security and compliance", prompt: "Which access controls, audit records, privacy obligations, fraud checks, or compliance requirements must be met?" },
  { key: "service levels and scale", topic: "Operational scale", prompt: "What volumes, peak periods, response times, availability targets, or recovery expectations should the solution support?" },
  { key: "delivery constraints", topic: "Delivery context", prompt: "What deadlines, budget or team constraints, stakeholder availability, technical dependencies, and change expectations should shape delivery?" },
  { key: "validation and rollout", topic: "Validation and rollout", prompt: "How should this be tested, approved, piloted, released, monitored, and improved after launch?" },
];
const domainOpeners = [
  { terms: ["appointment", "schedule", "booking", "double booking", "reschedule", "clinic"], topic: "Appointment workflow", prompt: "For the appointment journey, what starts the scheduling request, which stakeholders participate, and what outcome should the patient receive?" },
  { terms: ["patient registration", "registration", "check-in", "admission"], topic: "Patient registration", prompt: "For the registration flow, what patient information is collected, which staff roles are involved, and what confirmation or follow-up should the patient receive?" },
  { terms: ["ehr", "electronic health record", "record", "clinical notes", "demographics"], topic: "Clinical records", prompt: "For the records workflow, which patient information is created or updated, which roles access it, and what audit or privacy controls are expected?" },
  { terms: ["telemedicine", "virtual visit", "consultation", "video visit"], topic: "Telemedicine", prompt: "For the virtual-care journey, what starts the clinical interaction, which clinicians and patients participate, and what outcome or follow-up is expected?" },
  { terms: ["prescription", "medication", "lab", "laboratory", "test results"], topic: "Medication and diagnostics", prompt: "For the medication or diagnostics flow, what data is exchanged, which teams act on it, and what approvals or notifications are required?" },
  { terms: ["hospital", "care team", "nurse", "doctor", "clinician", "administrator"], topic: "Care coordination", prompt: "For the clinical workflow, which roles coordinate care, what handoffs occur, and which operational or safety checks must be in place?" },
];

function answerExcerpt(answer) {
  const compact = answer.replace(/\s+/g, " ").trim();
  return compact.length > 150 ? `${compact.slice(0, 147)}...` : compact;
}

function selectOpeningQuestion(context) {
  const opener = domainOpeners.find((rule) => rule.terms.some((term) => context.includes(term)));
  return { key: "workflow and outcomes", ...(opener || { topic: "Workflow", prompt: "Walk through the normal journey. What starts the process, what decisions happen, and what outcome should the user receive?" }) };
}

function selectNextQuestion(previousAnswer) {
  const remaining = coverageQuestions.filter((question) => !askedQuestionKeys.has(question.key));
  if (!remaining.length) return null;
  const answer = previousAnswer.toLowerCase();
  const followUpOrder = [
    { key: "exceptions", terms: ["fail", "error", "reject", "declin", "exception", "manual", "override"] },
    { key: "data and integrations", terms: ["api", "system", "integrat", "third party", "vendor", "data", "document"] },
    { key: "security and compliance", terms: ["risk", "fraud", "security", "audit", "privacy", "compliance", "access"] },
    { key: "business rules", terms: ["rule", "limit", "approv", "eligib", "decision", "threshold"] },
    { key: "service levels and scale", terms: ["volume", "peak", "time", "urgent", "daily", "availability", "customer"] },
  ];
  const followUp = followUpOrder.find((rule) => rule.terms.some((term) => answer.includes(term)));
  const selected = remaining.find((question) => question.key === followUp?.key) || remaining[0];
  const context = previousAnswer ? `You mentioned "${answerExcerpt(previousAnswer)}". ` : "To complete the project picture, ";
  return { ...selected, prompt: `${context}${selected.prompt}` };
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 3200);
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    let detail = "";
    try {
      const body = JSON.parse(text);
      detail = typeof body.detail === "string" ? body.detail : "";
    } catch {
      detail = text.trim().slice(0, 240);
    }
    throw new Error(detail || `Request failed (${response.status} ${response.statusText})`);
  }
  try {
    return await response.json();
  } catch {
    throw new Error("The server returned an invalid JSON response.");
  }
}

function displayDocuments(documents, rag) {
  if (!Array.isArray(documents) || !rag || typeof rag !== "object") {
    throw new Error("The project API returned incomplete source-index information.");
  }
  $("#documentList").innerHTML = documents.map((doc) => `<div class="doc-row"><strong>${escapeHtml(doc.name)}</strong><small>${doc.chunks} processed evidence chunks</small></div>`).join("");
  $("#ragSummary").innerHTML = `<span>${escapeHtml(rag.vector_database)}</span><span>${escapeHtml(rag.embedding_model)}</span><span>${rag.indexed_chunks} vectors indexed</span>`;
  $("#documentsPanel").classList.remove("is-hidden");
}

function displayRetrievedEvidence(result) {
  if (!result || !Array.isArray(result.matches)) {
    throw new Error("The evidence API returned an invalid result.");
  }
  const list = $("#retrievalResults");
  list.innerHTML = result.matches.length
    ? result.matches.map((match) => `<article class="retrieval-match"><div><span class="field-label">${escapeHtml(match.document_name)} / ${escapeHtml(match.id)}</span><p>${escapeHtml(match.text)}</p></div><strong>${Math.round(match.relevance * 100)}%</strong></article>`).join("")
    : "<p class=\"muted\">No matching chunks found.</p>";
  $("#retrievalCount").textContent = `${result.count} match(es)`;
}

function validateRequirementsResponse(project) {
  if (!project || typeof project !== "object" || !Array.isArray(project.requirements) || !Array.isArray(project.quality_issues)) {
    throw new Error("Agent 1 returned incomplete requirements or quality findings.");
  }
  if (!Array.isArray(project.project_activities) || !Array.isArray(project.analysis_gaps)) {
    throw new Error("Agent 1 returned an incomplete activity/gap analysis.");
  }
  if (project.project_activities.some((item) => !item || typeof item.id !== "string"
      || typeof item.activity !== "string" || !Array.isArray(item.evidence))
      || project.analysis_gaps.some((item) => typeof item !== "string")) {
    throw new Error("Agent 1 returned malformed activity or gap data.");
  }
  const validRequirement = (req) => req && typeof req.id === "string"
    && typeof req.statement === "string" && Array.isArray(req.categories)
    && Array.isArray(req.evidence) && Array.isArray(req.acceptance_criteria)
    && Array.isArray(req.dependencies) && Array.isArray(req.assumptions)
    && Array.isArray(req.risk_notes) && Array.isArray(req.healthcare_concerns);
  if (project.requirements.some((req) => !validRequirement(req))) {
    throw new Error("Agent 1 returned a malformed requirement; no partial result was displayed.");
  }
  if (project.quality_issues.some((issue) => !issue || typeof issue.check !== "string" || typeof issue.message !== "string")) {
    throw new Error("Agent 1 returned malformed quality findings.");
  }
  return project;
}

function displayRequirements(project) {
  validateRequirementsResponse(project);
  const list = $("#requirementsList");
  list.innerHTML = project.requirements.length ? project.requirements.map((req) => {
    const riskNotes = (req.risk_notes || []).concat(req.healthcare_concerns || []);
    return `<article class="requirement"><div class="requirement-head"><div><span class="req-id">${escapeHtml(req.id)}</span><h3>${escapeHtml(req.statement)}</h3><div class="tags">${req.categories.map((category) => `<span class="tag">${escapeHtml(category)}</span>`).join("")}</div></div><select class="status" data-id="${escapeHtml(req.id)}" aria-label="Approval status for ${escapeHtml(req.id)}">${["Draft", "Needs review", "Approved", "Rejected"].map((status) => `<option ${req.approval_status === status ? "selected" : ""}>${status}</option>`).join("")}</select></div><div class="requirement-grid"><div><span class="field-label">Clinical justification</span><p>${escapeHtml(req.business_justification)}</p></div><div><span class="field-label">Confidence / risk</span><p>${Math.round(req.confidence * 100)}% confidence · ${escapeHtml(req.risk_level)} risk · ${escapeHtml(req.priority)}</p></div><div><span class="field-label">Acceptance criteria</span><p>${req.acceptance_criteria.map(escapeHtml).join("<br>")}</p></div><div><span class="field-label">Dependencies / assumptions</span><p>${escapeHtml((req.dependencies || []).join(", ") || "None stated")}<br>${escapeHtml((req.assumptions || []).join("; ") || "None stated")}</p></div><div><span class="field-label">Healthcare risk notes</span><p>${riskNotes.map(escapeHtml).join("<br>") || "No specific concern stated"}</p></div><div><span class="field-label">Reasoning</span><p>${escapeHtml(req.reasoning)}</p></div></div>${req.evidence.map((evidence) => `<div class="evidence"><span class="field-label">Evidence · ${escapeHtml(evidence.document_name)} / ${escapeHtml(evidence.chunk_id)}</span><p>${escapeHtml(evidence.excerpt)}</p></div>`).join("")}</article>`;
  }).join("") : "<p class=\"muted\">No testable software requirements could be extracted. Review the gap analysis and provide clearer system behaviors.</p>";
  $("#requirementCount").textContent = `${project.requirements.length} draft(s)`;
  const run = project.agent_runs && project.agent_runs.requirements;
  if (run) {
    const mode = run.mode === "llm"
      ? "LLM-generated requirements; structured output and evidence references validated"
      : "Evidence-only fallback; LLM was not used for requirement generation";
    $("#agent1Summary").innerHTML = `<strong>Agent 1 · Requirements Engineering Agent</strong><p>${escapeHtml(mode)}.</p><p>${run.requirement_count} requirements · ${run.retrieved_chunk_count} RAG chunks · validation ${run.validation_passed ? "passed" : "failed"} · deterministic quality checks: ${project.quality_issues.length} finding(s)</p>`;
    $("#agent1Summary").classList.remove("is-hidden");
  }
  const gap = $("#agent1Gap");
  gap.innerHTML = project.analysis_gaps.length
    ? `<strong>Evidence gaps</strong>${project.analysis_gaps.map((item) => `<p>${escapeHtml(item)}</p>`).join("")}`
    : "";
  gap.classList.toggle("is-hidden", !project.analysis_gaps.length);
  const activities = $("#agent1Activities");
  activities.innerHTML = project.project_activities.length
    ? `<h3>Project activities (not software requirements)</h3>${project.project_activities.map((item) => `<article class="issue"><div><span class="severity">${escapeHtml(item.category)}</span></div><div><strong>${escapeHtml(item.id)}</strong><p>${escapeHtml(item.activity)}</p>${item.evidence.map((source) => `<small>${escapeHtml(source.document_name)} / ${escapeHtml(source.chunk_id)}</small>`).join("<br>")}</div></article>`).join("")}`
    : "";
  $("#requirementsPanel").classList.remove("is-hidden");
  list.querySelectorAll("select").forEach((select) => select.addEventListener("change", async () => {
    const data = new FormData();
    data.append("status", select.value);
    try {
      await request(`/api/projects/${projectId}/requirements/${select.dataset.id}/approval`, { method: "POST", body: data });
      $("#agent2Summary").classList.add("is-hidden");
      $("#sdlcResult").textContent = "";
      toast("Approval status recorded in audit trail.");
    } catch (error) {
      toast(error.message);
    }
  }));
}

function displayIssues(issues) {
  if (!Array.isArray(issues) || issues.some((issue) => !issue || typeof issue.check !== "string")) {
    throw new Error("The quality analysis returned malformed findings.");
  }
  $("#issuesList").innerHTML = issues.length
    ? issues.map((issue) => `<article class="issue"><div><span class="severity">${escapeHtml(issue.severity)}</span><p class="muted">${escapeHtml(issue.check)}</p></div><div><strong>${escapeHtml(issue.requirement_id || "Project")}</strong><p>${escapeHtml(issue.message)}</p><p class="recommendation">${escapeHtml(issue.recommendation)}</p></div></article>`).join("")
    : "<p class=\"muted\">No rule-based issues found. Human review is still required.</p>";
  $("#issueCount").textContent = `${issues.length} issue(s)`;
  $("#qualityPanel").classList.remove("is-hidden");
}

function validateSdlcResponse(sdlc) {
  if (!sdlc || typeof sdlc !== "object") {
    throw new Error("Agent 2 returned no SDLC recommendation data.");
  }
  if (!Array.isArray(sdlc.input_requirement_ids) || !sdlc.input_requirement_ids.length || sdlc.input_requirement_ids.some((id) => typeof id !== "string" || !id)) {
    throw new Error("The SDLC response is missing Agent 1 requirement IDs. The backend may be out of date; restart the FastAPI server and try again.");
  }
  if (!Array.isArray(sdlc.factors) || sdlc.factors.length !== 12 || sdlc.factors.some((factor) => !factor || typeof factor.name !== "string" || typeof factor.rationale !== "string" || !Array.isArray(factor.requirement_ids) || !Number.isInteger(factor.score) || factor.score < 1 || factor.score > 5)) {
    throw new Error("Agent 2 returned incomplete factor data. No recommendation was displayed.");
  }
  if (!Array.isArray(sdlc.options) || sdlc.options.length !== 7 || sdlc.options.some((option) => !option || typeof option.name !== "string" || !Number.isInteger(option.score) || option.score < 0 || option.score > 100 || !option.score_breakdown || !option.score_weights)) {
    throw new Error("Agent 2 returned incomplete SDLC comparison data. No recommendation was displayed.");
  }
  if (!Array.isArray(sdlc.workflow) || !sdlc.workflow.length || sdlc.workflow.some((phase) => !phase || typeof phase.phase !== "string" || !Array.isArray(phase.activities) || !Array.isArray(phase.deliverables))) {
    throw new Error("Agent 2 returned incomplete workflow data. No recommendation was displayed.");
  }
  if (typeof sdlc.recommended_sdlc !== "string" || !sdlc.recommended_sdlc || !Number.isInteger(sdlc.recommended_score) || sdlc.recommended_score !== sdlc.options[0].score || sdlc.recommended_sdlc !== sdlc.options[0].name || typeof sdlc.justification !== "string" || !Array.isArray(sdlc.risks_and_limitations)) {
    throw new Error("Agent 2 returned an incomplete recommendation. No recommendation was displayed.");
  }
  return sdlc;
}

function displaySdlc(sdlc) {
  validateSdlcResponse(sdlc);
  const mode = sdlc.analysis_mode === "llm-assisted"
    ? "LLM factor analysis + transparent scoring"
    : "Rules-based factor fallback + transparent scoring";
  $("#agent2Summary").innerHTML = `<strong>Agent 2 · SDLC Selection Agent</strong><p>${escapeHtml(mode)}. Agent 2 received ${sdlc.input_requirement_ids.length} validated requirement(s) from Agent 1.</p><p>Requirement IDs: ${escapeHtml(sdlc.input_requirement_ids.join(", "))}</p>`;
  $("#agent2Summary").classList.remove("is-hidden");
  $("#sdlcResult").innerHTML = `<div class="sdlc-overview"><div><p class="eyebrow">Recommended model</p><h3>${escapeHtml(sdlc.recommended_sdlc)} · ${sdlc.recommended_score}/100</h3><p>${escapeHtml(sdlc.justification)}</p></div><div class="rankings">${sdlc.options.map((option) => `<div class="ranking"><span>${escapeHtml(option.name)}</span><div class="bar"><span style="width:${option.score}%"></span></div><strong>${option.score}%</strong><small>${escapeHtml(option.rationale)}</small><small>Weighted fit points: ${escapeHtml(Object.entries(option.score_breakdown).map(([name, score]) => `${name} ${score} (weight ${option.score_weights[name]})`).join(" · "))}</small></div>`).join("")}</div></div><div class="workflow"><p class="eyebrow">Decision limits and evidence gaps</p>${sdlc.risks_and_limitations.length ? `<ul>${sdlc.risks_and_limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : "<p>No additional limitations were returned.</p>"}</div><div class="workflow"><p class="eyebrow">Requirement-derived SDLC factors (1–5)</p><div class="factor-grid">${sdlc.factors.map((factor) => `<div class="factor"><h4>${escapeHtml(factor.name)}</h4><strong>${factor.score}/5 · ${escapeHtml(factor.value)}</strong><p>${escapeHtml(factor.rationale)}</p>${factor.requirement_ids.length ? `<small>Linked Agent 1 requirements: ${escapeHtml(factor.requirement_ids.join(", "))}</small>` : ""}</div>`).join("")}</div></div><div class="workflow"><p class="eyebrow">Project-specific workflow</p><div class="workflow-grid">${sdlc.workflow.map((phase) => `<div class="phase"><h4>${escapeHtml(phase.phase)}</h4><p>${escapeHtml(phase.activities.join(" · "))}</p><p><strong>Deliverables:</strong> ${escapeHtml(phase.deliverables.join(" · "))}</p><p><strong>Gate:</strong> ${escapeHtml(phase.gate)}</p></div>`).join("")}</div></div>`;
}

function renderQuestion() {
  const question = questionQueue[questionIndex];
  if (!question) {
    $("#interviewSession").classList.add("is-hidden");
    $("#interviewComplete").classList.remove("is-hidden");
    $("#interviewSummary").textContent = `All ${INTERVIEW_QUESTION_LIMIT} project prompts are complete. ${Object.keys(interviewAnswers).length} responses are ready for requirement and SDLC analysis.`;
    return;
  }
  $("#questionProgress").textContent = `Question ${questionIndex + 1} of ${INTERVIEW_QUESTION_LIMIT}`;
  $("#answerCount").textContent = `${Object.keys(interviewAnswers).length} answers saved`;
  $("#questionTopic").textContent = question.topic;
  $("#interviewQuestion").textContent = question.prompt;
  $("#interviewAnswer").value = interviewAnswers[question.key] || "";
  $("#interviewAnswer").focus();
}

function advanceInterview(saveAnswer) {
  const question = questionQueue[questionIndex];
  const answer = $("#interviewAnswer").value.trim();
  if (saveAnswer && !answer) {
    toast("Add an answer or choose Skip for now.");
    return;
  }
  if (answer) interviewAnswers[question.key] = answer;
  if (questionQueue.length < INTERVIEW_QUESTION_LIMIT) {
    const nextQuestion = selectNextQuestion(answer);
    if (nextQuestion) {
      askedQuestionKeys.add(nextQuestion.key);
      questionQueue.push(nextQuestion);
    }
  }
  questionIndex += 1;
  renderQuestion();
}

$("#interviewStartForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const functionality = $("#functionality").value.trim();
  const objective = $("#businessObjective").value.trim();
  const users = $("#usersRoles").value.trim();

  if (functionality.length < 3 || !objective || !users) {
    toast("Complete the functionality, business objective, and users/roles to begin.");
    return;
  }

  interviewAnswers["business objective"] = objective;
  interviewAnswers["users and roles"] = users;

  const openingQuestion = selectOpeningQuestion(`${functionality} ${objective} ${users}`.toLowerCase());
  questionQueue = [openingQuestion];
  askedQuestionKeys = new Set([openingQuestion.key]);
  questionIndex = 0;

  $("#interviewStartForm").classList.add("is-hidden");
  $("#interviewSession").classList.remove("is-hidden");
  renderQuestion();
});

$("#continueInterview").addEventListener("click", () => advanceInterview(true));
$("#skipInterviewQuestion").addEventListener("click", () => advanceInterview(false));

$("#createProject").addEventListener("click", async () => {
  const button = $("#createProject");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Creating project…";
  try {
    const data = new FormData();
    data.append("functionality", $("#functionality").value.trim());
    data.append("stakeholder_input", JSON.stringify(interviewAnswers));
    data.append("project_context", JSON.stringify({
      objective: $("#businessObjective").value.trim(),
      stakeholders: $("#usersRoles").value.trim(),
      constraints: $("#projectConstraints").value.trim(),
      limitations: $("#projectLimitations").value.trim(),
      interview_responses: interviewAnswers,
    }));
    [...$("#files").files].forEach((file) => data.append("files", file));

    const result = await request("/api/projects", { method: "POST", body: data });
    projectId = result.project_id;
    $("#projectState").textContent = "Project ready";
    $("#exportLink").href = `/api/projects/${projectId}/export`;
    $("#exportLink").classList.remove("is-hidden");
    displayDocuments(result.documents, result.rag);
    displayRetrievedEvidence(await request(`/api/projects/${projectId}/retrieval`));
    $("#interviewComplete").classList.add("is-hidden");
    toast("Sources were indexed and the most relevant evidence is ready for review.");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
});

$("#retrieveButton").addEventListener("click", async () => {
  if (!projectId) return;
  const button = $("#retrieveButton");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Searching…";
  try {
    const query = $("#retrievalQuery").value.trim() || $("#functionality").value.trim();
    displayRetrievedEvidence(await request(`/api/projects/${projectId}/retrieval?query=${encodeURIComponent(query)}`));
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
});

$("#generateButton").addEventListener("click", async () => {
  if (!projectId) return;
  const button = $("#generateButton");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Generating…";
  $("#requirementsList").textContent = "Agent 1 is analyzing the retrieved evidence…";
  $("#issuesList").textContent = "";
  $("#agent1Summary").classList.add("is-hidden");
  $("#agent1Gap").classList.add("is-hidden");
  $("#agent1Activities").textContent = "";
  $("#agent2Summary").classList.add("is-hidden");
  $("#sdlcResult").textContent = "";
  try {
    const project = await request(`/api/projects/${projectId}/requirements`, { method: "POST" });
    displayRequirements(project);
    displayIssues(project.quality_issues);
    $("#sdlcPanel").classList.remove("is-hidden");
    $("#sdlcButton").disabled = project.requirements.length === 0;
    toast("Evidence-grounded drafts are ready for review.");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
});

$("#sdlcButton").addEventListener("click", async () => {
  if (!projectId) return;
  const button = $("#sdlcButton");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Analyzing…";
  $("#agent2Summary").classList.add("is-hidden");
  $("#sdlcResult").textContent = "Agent 2 is assessing Agent 1's validated requirements…";
  try {
    displaySdlc(await request(`/api/projects/${projectId}/sdlc`, { method: "POST" }));
    toast("SDLC recommendation generated. Human approval remains required.");
  } catch (error) {
    $("#sdlcResult").textContent = "";
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
});
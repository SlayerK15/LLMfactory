// Orchestrator UI — submits a pipeline run and renders the SSE event stream.

const form = document.getElementById("pipeline-form");
const logEl = document.getElementById("log");
const startBtn = document.getElementById("start");
const statusPill = document.getElementById("status-pill");
const counterEl = document.getElementById("counter");
const autoscroll = document.getElementById("autoscroll");
const clearBtn = document.getElementById("clear-log");
const summaryEl = document.getElementById("summary");
const summaryList = document.getElementById("summary-list");
const trainKnobs = document.getElementById("train-knobs");

let eventCount = 0;
let currentRunId = null;

// --- Mode picker: show/hide training fields -----------------------------------
function syncModeVisibility() {
  const mode = form.querySelector('input[name="mode"]:checked')?.value || "dataset_only";
  trainKnobs.hidden = mode !== "train";
  const trainStage = document.querySelector('.stage[data-stage="train"]');
  if (trainStage) trainStage.classList.toggle("visible", mode === "train");
}
form.querySelectorAll('input[name="mode"]').forEach((el) => el.addEventListener("change", syncModeVisibility));
syncModeVisibility();

// --- Log filters --------------------------------------------------------------
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    const f = chip.dataset.filter;
    logEl.classList.remove("filter-stage", "filter-log", "filter-err");
    if (f !== "all") logEl.classList.add(`filter-${f}`);
  });
});

clearBtn.addEventListener("click", () => {
  logEl.replaceChildren();
  eventCount = 0;
  counterEl.textContent = "0 events";
});

// --- Stage pill helpers -------------------------------------------------------
function setStage(stage, state) {
  const el = document.querySelector(`.stage[data-stage="${stage}"]`);
  if (!el) return;
  el.classList.remove("running", "done", "failed");
  if (state) el.classList.add(state);
}

function resetStages(mode) {
  ["init", "collect", "clean", "forge", "train", "done"].forEach((s) => setStage(s, null));
  const trainStage = document.querySelector('.stage[data-stage="train"]');
  if (trainStage) trainStage.classList.toggle("visible", mode === "train");
}

function setStatus(state, label) {
  statusPill.className = `status ${state}`;
  statusPill.textContent = label;
}

// --- Log output ---------------------------------------------------------------
function append(line, cls) {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  const ts = new Date().toLocaleTimeString([], { hour12: false });
  const tsEl = document.createElement("span");
  tsEl.className = "ts";
  tsEl.textContent = ts;
  span.appendChild(tsEl);
  span.appendChild(document.createTextNode(line));
  logEl.appendChild(span);
  eventCount += 1;
  counterEl.textContent = `${eventCount} event${eventCount === 1 ? "" : "s"}`;
  if (autoscroll.checked) logEl.scrollTop = logEl.scrollHeight;
}

// --- Summary rendering --------------------------------------------------------
function clearSummary() {
  summaryList.replaceChildren();
  summaryEl.hidden = true;
}

function renderSummary(results) {
  summaryList.replaceChildren();
  if (!results || typeof results !== "object") return;
  const add = (html) => {
    const li = document.createElement("li");
    li.innerHTML = html;
    summaryList.appendChild(li);
  };

  if (currentRunId) add(`Run ID: <code>${currentRunId}</code>`);

  const collect = results.collect;
  if (collect?.total_docs != null) add(`Collected <code>${collect.total_docs}</code> docs`);

  const clean = results.clean;
  if (clean?.docs_kept != null) {
    add(`Cleaned: kept <code>${clean.docs_kept}</code> / ${clean.docs_in ?? "?"}`);
  }

  const forge = results.forge;
  if (forge?.train_path) add(`Train JSONL: <code>${forge.train_path}</code>`);
  if (forge?.eval_path) add(`Eval JSONL: <code>${forge.eval_path}</code>`);

  const train = results.train;
  if (train?.artifact?.local_path) add(`Model artifact: <code>${train.artifact.local_path}</code>`);
  if (train?.artifact?.s3_uri) add(`Uploaded to: <code>${train.artifact.s3_uri}</code>`);
  if (train?.eval_loss_final != null) add(`Final eval loss: <code>${Number(train.eval_loss_final).toFixed(4)}</code>`);

  summaryEl.hidden = summaryList.children.length === 0;
}

// --- Form submission ----------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const mode = fd.get("mode") || "dataset_only";
  const body = {
    topic: fd.get("topic"),
    doc_count: Number(fd.get("doc_count")),
    output_mode: fd.get("output_mode"),
    mode,
  };
  if (mode === "train") {
    body.base_model = fd.get("base_model") || "unsloth/Llama-3.2-3B-Instruct-bnb-4bit";
    body.epochs = Number(fd.get("epochs") || 1);
    body.lora_rank = Number(fd.get("lora_rank") || 16);
    body.output_format = fd.get("output_format") || "gguf_q4_k_m";
    const bucket = fd.get("s3_bucket");
    if (bucket) body.s3_bucket = bucket;
  }

  const endpoint = document.getElementById("dry-run").checked
    ? "/api/pipeline/dry-run"
    : "/api/pipeline";

  logEl.replaceChildren();
  eventCount = 0;
  counterEl.textContent = "0 events";
  resetStages(mode);
  clearSummary();
  setStatus("running", "running");
  startBtn.disabled = true;
  currentRunId = null;

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}${text ? `: ${text}` : ""}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      // sse-starlette uses CRLF line endings, but the SSE spec also permits LF-only.
      // Normalise to LF so a single "\n\n" split handles both.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        if (rawEvent) handleRawEvent(rawEvent);
      }
    }
  } catch (err) {
    append(`error: ${err.message}`, "err");
    setStatus("failed", "failed");
  } finally {
    startBtn.disabled = false;
  }
});

function handleRawEvent(raw) {
  const lines = raw.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  let payload;
  try {
    payload = JSON.parse(data);
  } catch {
    append(data);
    return;
  }
  const { stage, message, kind, run_id, data: payloadData } = payload;

  if (run_id && !currentRunId) currentRunId = run_id;

  if (kind === "log") {
    append(message || "", "log-line");
    return;
  }

  append(`[${stage}] ${kind}: ${message || ""}`, `stage-${kind}`);

  if (kind === "stage_started") setStage(stage, "running");
  else if (kind === "stage_done") setStage(stage, "done");
  else if (kind === "error") {
    setStage(stage, "failed");
    setStatus("failed", "failed");
  } else if (kind === "pipeline_done") {
    setStage("done", "done");
    setStatus("done", "done");
    renderSummary(payloadData);
  }
}

// ---- Config -----------------------------------------------------------
// Flip USE_MOCK_DATA to false once you have a real GEMINI_API_KEY set and
// the backend running, to call the live pipeline instead of the saved
// sample_response.json.
const CONFIG = {
  USE_MOCK_DATA: false,
  API_BASE_URL: "http://localhost:8000",
};

// Cycled through in the typing indicator while a request is in flight —
// purely cosmetic, since the backend returns the whole result in one shot
// rather than streaming per-agent progress yet.
const AGENT_STATUSES = [
  "Planning research approach",
  "Search Agent researching",
  "Paper Agent reviewing literature",
  "Benchmark Agent comparing metrics",
  "Fact Checker cross-referencing",
  "Writing final report",
];

// ---- Elements -----------------------------------------------------------
const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const messagesEl = document.getElementById("messages"); // scroll container
const introHint = document.getElementById("intro-hint");
const chatColumn = messagesEl.querySelector(".chat-column"); // where messages are appended
const resetBtn = document.getElementById("reset-btn");
const modeDot = document.getElementById("mode-dot");

// Prior turns of this conversation, sent with each request so the backend can resolve
// follow-up references ("compare that with X") — see app/history.py on the backend.
let conversationHistory = [];

modeDot.title = CONFIG.USE_MOCK_DATA
  ? "Demo mode — showing saved sample data"
  : `Live mode — calling ${CONFIG.API_BASE_URL}`;

resetBtn.addEventListener("click", () => {
  chatColumn.innerHTML = "";
  chatColumn.appendChild(introHint);
  introHint.hidden = false;
  conversationHistory = [];
});

// ---- Main flow -----------------------------------------------------------
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  introHint.hidden = true;
  input.value = "";
  submitBtn.disabled = true;

  appendUserMessage(query);
  const typingRow = appendTypingIndicator();

  try {
    const data = CONFIG.USE_MOCK_DATA ? await loadMockData() : await callResearchApi(query);
    typingRow.remove();
    const answer = data.final_report || "_No report was generated._";
    appendAssistantMessage(answer);
    conversationHistory.push({ query, answer });
  } catch (err) {
    typingRow.remove();
    appendAssistantMessage(null, err);
  } finally {
    submitBtn.disabled = false;
    input.focus();
  }
});

// ---- Message rendering -----------------------------------------------------
function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg user";
  row.innerHTML = `<div class="bubble-user">${escapeHtml(text)}</div>`;
  chatColumn.appendChild(row);
  scrollToBottom();
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg assistant typing-row";
  row.innerHTML = `
    <div class="avatar">✨</div>
    <div class="assistant-body">
      <div class="msg-meta">
        <span class="msg-author">Research Agent</span>
        <span>${currentTime()}</span>
      </div>
      <div class="typing-line">
        <span class="typing-text" id="typing-text">${AGENT_STATUSES[0]}</span>
        <span class="dots"><span></span><span></span><span></span></span>
      </div>
    </div>
  `;
  chatColumn.appendChild(row);
  scrollToBottom();

  let i = 0;
  const textEl = row.querySelector("#typing-text");
  const interval = setInterval(() => {
    i = (i + 1) % AGENT_STATUSES.length;
    textEl.textContent = AGENT_STATUSES[i];
  }, 1400);

  // Stop the interval whenever this row is removed from the DOM.
  const originalRemove = row.remove.bind(row);
  row.remove = () => {
    clearInterval(interval);
    originalRemove();
  };

  return row;
}

function appendAssistantMessage(markdownText, error) {
  const row = document.createElement("div");
  row.className = "msg assistant";

  const body = error
    ? `<div class="error-text">Something went wrong: ${escapeHtml(error.message || String(error))}</div>`
    : `<div class="assistant-text">${marked.parse(markdownText)}</div>`;

  row.innerHTML = `
    <div class="avatar">✨</div>
    <div class="assistant-body">
      <div class="msg-meta">
        <span class="msg-author">Research Agent</span>
        <span>${currentTime()}</span>
      </div>
      ${body}
    </div>
  `;
  chatColumn.appendChild(row);
  scrollToBottom();
}

// ---- Data sources ---------------------------------------------------------
async function loadMockData() {
  const res = await fetch("sample_response.json");
  if (!res.ok) {
    throw new Error("Could not load sample_response.json");
  }
  return res.json();
}

async function callResearchApi(query) {
  const res = await fetch(`${CONFIG.API_BASE_URL}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history: conversationHistory }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with status ${res.status}`);
  }

  return res.json();
}

// ---- Helpers ---------------------------------------------------------------
function currentTime() {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

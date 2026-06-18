/**
 * Healthcare CDSS — Chat UI Controller
 * WebSocket-based real-time chat with the LangGraph multi-agent system.
 */
(function () {
  "use strict";

  const chatArea = document.getElementById("chatArea");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const welcomeMsg = document.getElementById("welcomeMsg");
  const loginForm = document.getElementById("loginForm");
  const loginIdentifier = document.getElementById("loginIdentifier");
  const loginPassword = document.getElementById("loginPassword");
  const loginBtn = document.getElementById("loginBtn");
  const loginError = document.getElementById("loginError");
  const patientWorkspace = document.getElementById("patientWorkspace");
  const patientName = document.getElementById("patientName");
  const patientMeta = document.getElementById("patientMeta");
  const logoutBtn = document.getElementById("logoutBtn");
  const diseaseSelect = document.getElementById("diseaseSelect");
  const multiDiagnosis = document.getElementById("multiDiagnosis");
  const addDiseaseBtn = document.getElementById("addDiseaseBtn");
  const historyList = document.getElementById("historyList");

  let ws = null;
  let isProcessing = false;
  let transport = "rest";
  let sessionId =
    sessionStorage.getItem("chatSessionId") ||
    (crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`);
  let patientToken = sessionStorage.getItem("patientToken") || "";
  let currentPatient = readStoredPatient();
  sessionStorage.setItem("chatSessionId", sessionId);

  // ── WebSocket ──
  function connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      console.log("[WS] Connected");
      transport = "websocket";
      updateStatus("Connected", "connected");
    };

    ws.onmessage = function (event) {
      const data = JSON.parse(event.data);
      handleMessage(data);
    };

    ws.onerror = function (error) {
      console.error("[WS] Error:", error);
      transport = "rest";
      updateStatus("REST fallback", "connected");
    };

    ws.onclose = function () {
      console.log("[WS] Disconnected; using REST fallback.");
      transport = "rest";
      updateStatus("REST fallback", "connected");
    };
  }

  function updateStatus(text, state) {
    const indicator = document.getElementById("statusIndicator");
    const dot = indicator.querySelector(".status-dot");
    const textEl = indicator.querySelector(".status-text");
    textEl.textContent = text;

    const colors = {
      connected: "var(--success)",
      disconnected: "var(--warning)",
      error: "var(--danger)",
    };
    dot.style.background = colors[state] || "var(--success)";
  }

  // ── Messages ──
  function handleMessage(data) {
    switch (data.type) {
      case "processing":
        showTyping();
        break;
      case "response":
        removeTyping();
        if (data.guardrail_warnings && data.guardrail_warnings.length) {
          addMessage("assistant", data.guardrail_warnings.join("\n"));
        }
        addMessage("assistant", data.response);
        isProcessing = false;
        enableInput();
        break;
      case "error":
        removeTyping();
        addMessage("assistant", `⚠️ ${data.message}`);
        isProcessing = false;
        enableInput();
        break;
    }
  }

  function addMessage(role, text) {
    // Remove welcome message on first interaction
    if (welcomeMsg) welcomeMsg.remove();

    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "assistant" ? "🏥" : "👤";

    const content = document.createElement("div");
    content.className = "message-content";
    content.innerHTML = renderMarkdown(text);

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    chatArea.appendChild(msgDiv);
    scrollToBottom();
  }

  function showTyping() {
    removeTyping();
    const div = document.createElement("div");
    div.className = "message assistant";
    div.id = "typingIndicator";
    div.innerHTML = `
            <div class="message-avatar">🏥</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
    chatArea.appendChild(div);
    scrollToBottom();
  }

  function removeTyping() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
  }

  function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // ── Send ──
  async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isProcessing) return;

    addMessage("user", text);
    messageInput.value = "";
    autoResize();
    isProcessing = true;
    disableInput();

    if (transport === "websocket" && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ message: text, patient_token: patientToken }));
      return;
    }

    showTyping();
    try {
      const data = await apiRequest("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          patient_token: patientToken,
        }),
      });
      handleMessage({ type: "response", ...data });
    } catch (error) {
      handleMessage({ type: "error", message: error.message });
    }
  }

  function disableInput() {
    sendBtn.disabled = true;
    messageInput.disabled = false;
  }

  function enableInput() {
    sendBtn.disabled = false;
    messageInput.focus();
  }

  // Patient login and history
  function readStoredPatient() {
    try {
      return JSON.parse(sessionStorage.getItem("currentPatient") || "null");
    } catch (e) {
      return null;
    }
  }

  async function apiRequest(url, options) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed.");
    }
    return data;
  }

  async function loginPatient(event) {
    event.preventDefault();
    loginError.textContent = "";
    loginBtn.disabled = true;

    try {
      const data = await apiRequest("/api/patient/login", {
        method: "POST",
        body: JSON.stringify({
          identifier: loginIdentifier.value.trim(),
          password: loginPassword.value,
        }),
      });

      patientToken = data.token;
      currentPatient = data.patient;
      sessionStorage.setItem("patientToken", patientToken);
      sessionStorage.setItem("currentPatient", JSON.stringify(currentPatient));
      loginPassword.value = "";
      showPatientWorkspace();
      renderHistory(data.history || []);
      await loadDiseases();
    } catch (error) {
      loginError.textContent = error.message;
    } finally {
      loginBtn.disabled = false;
    }
  }

  async function logoutPatient() {
    if (patientToken) {
      try {
        await apiRequest("/api/patient/logout", {
          method: "POST",
          body: JSON.stringify({ token: patientToken }),
        });
      } catch (e) {
        console.warn("[Patient] Logout request failed:", e);
      }
    }

    patientToken = "";
    currentPatient = null;
    sessionStorage.removeItem("patientToken");
    sessionStorage.removeItem("currentPatient");
    loginForm.classList.remove("hidden");
    patientWorkspace.classList.add("hidden");
    historyList.innerHTML = "";
    diseaseSelect.innerHTML = '<option value="">Select disease</option>';
    loginIdentifier.focus();
  }

  function showPatientWorkspace() {
    if (!currentPatient) return;
    loginForm.classList.add("hidden");
    patientWorkspace.classList.remove("hidden");
    patientName.textContent = currentPatient.patient_name || "Patient";
    patientMeta.textContent =
      `ID ${currentPatient.patient_id}` +
      (currentPatient.email ? ` | ${currentPatient.email}` : "");
  }

  async function loadHistory() {
    if (!patientToken) return;
    try {
      const data = await apiRequest(
        `/api/patient/history?token=${encodeURIComponent(patientToken)}`,
      );
      renderHistory(data.history || []);
    } catch (error) {
      await logoutPatient();
      loginError.textContent = error.message;
    }
  }

  async function loadDiseases() {
    if (!patientToken) return;
    const data = await apiRequest(
      `/api/diseases?token=${encodeURIComponent(patientToken)}`,
    );

    diseaseSelect.innerHTML = '<option value="">Select disease</option>';
    (data.diseases || []).forEach(function (disease) {
      const option = document.createElement("option");
      option.value = disease.disease_id;
      option.textContent = `${disease.disease_name} (${disease.severity_level})`;
      diseaseSelect.appendChild(option);
    });
  }

  function renderHistory(history) {
    historyList.innerHTML = "";
    if (!history.length) {
      const empty = document.createElement("p");
      empty.className = "empty-history";
      empty.textContent = "No disease history recorded.";
      historyList.appendChild(empty);
      return;
    }

    history.forEach(function (row) {
      const item = document.createElement("article");
      item.className = "history-item";

      const body = document.createElement("div");
      body.className = "history-body";

      const title = document.createElement("h3");
      title.textContent = row.disease_name;

      const meta = document.createElement("p");
      meta.className = "history-meta";
      meta.textContent = `${row.severity_group} / ${row.severity_level} | ${row.triage_recommendation}`;

      const symptoms = document.createElement("p");
      symptoms.className = "history-symptoms";
      symptoms.textContent = row.common_symptoms || "";

      body.appendChild(title);
      body.appendChild(meta);
      body.appendChild(symptoms);

      const remove = document.createElement("button");
      remove.className = "icon-action";
      remove.type = "button";
      remove.title = "Remove condition";
      remove.setAttribute("aria-label", `Remove ${row.disease_name}`);
      remove.dataset.historyId = row.patient_disease_id;
      remove.textContent = "x";

      item.appendChild(body);
      item.appendChild(remove);
      historyList.appendChild(item);
    });
  }

  async function addDisease() {
    const diseaseId = diseaseSelect.value;
    if (!diseaseId || !patientToken) return;

    addDiseaseBtn.disabled = true;
    try {
      const data = await apiRequest("/api/patient/history", {
        method: "POST",
        body: JSON.stringify({
          token: patientToken,
          disease_id: diseaseId,
          multi_diagnosis: multiDiagnosis.checked,
        }),
      });
      diseaseSelect.value = "";
      renderHistory(data.history || []);
    } catch (error) {
      loginError.textContent = error.message;
    } finally {
      addDiseaseBtn.disabled = false;
    }
  }

  async function removeDisease(historyId) {
    if (!historyId || !patientToken) return;

    const data = await apiRequest("/api/patient/history/delete", {
      method: "POST",
      body: JSON.stringify({
        token: patientToken,
        patient_disease_id: historyId,
      }),
    });
    renderHistory(data.history || []);
  }

  // ── Markdown renderer (supports tables, headings, bold, lists, blockquotes) ──
  function renderMarkdown(text) {
    let html = text;
    // Escape HTML
    html = html
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Split into lines for table detection
    var lines = html.split("\n");
    var result = [];
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      // Detect markdown table (line contains | and next line is separator)
      if (
        line.indexOf("|") !== -1 &&
        i + 1 < lines.length &&
        /^\|?[\s:-]+\|[\s:\-|]+$/.test(lines[i + 1].trim())
      ) {
        // Parse table
        var tableLines = [];
        // Header
        var headerCells = line
          .split("|")
          .map(function (c) {
            return c.trim();
          })
          .filter(function (c) {
            return c.length > 0;
          });
        tableLines.push('<table class="md-table"><thead><tr>');
        headerCells.forEach(function (cell) {
          tableLines.push("<th>" + cell + "</th>");
        });
        tableLines.push("</tr></thead><tbody>");

        // Skip separator line
        i += 2;

        // Body rows
        while (
          i < lines.length &&
          lines[i].indexOf("|") !== -1 &&
          lines[i].trim().length > 0
        ) {
          var bodyCells = lines[i]
            .split("|")
            .map(function (c) {
              return c.trim();
            })
            .filter(function (c) {
              return c.length > 0;
            });
          tableLines.push("<tr>");
          bodyCells.forEach(function (cell) {
            // Render bold inside cells
            cell = cell.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
            tableLines.push("<td>" + cell + "</td>");
          });
          tableLines.push("</tr>");
          i++;
        }
        tableLines.push("</tbody></table>");
        result.push(tableLines.join(""));
        continue;
      }

      // Headings (### → h3, ## → h3)
      line = line.replace(/^### (.+)$/, "<h3>$1</h3>");
      line = line.replace(/^## (.+)$/, "<h3>$1</h3>");

      // Horizontal rule (---)
      if (/^---+$/.test(line.trim())) {
        result.push("<hr>");
        i++;
        continue;
      }

      // Blockquote
      line = line.replace(/^&gt; (.+)$/, "<blockquote>$1</blockquote>");

      // Bold
      line = line.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

      // Italic (single underscores or single asterisks)
      line = line.replace(/(?<!\w)_([^_]+)_(?!\w)/g, "<em>$1</em>");
      line = line.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");

      // List items
      line = line.replace(/^- (.+)$/, "<li>$1</li>");

      result.push(line);
      i++;
    }

    html = result.join("\n");

    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>[\s\S]*?<\/li>)(?=\n[^<]|$)/g, function (match) {
      return "<ul>" + match + "</ul>";
    });
    html = html.replace(/(<li>.+<\/li>\n?)+/g, "<ul>$&</ul>");

    // Line breaks (but not around block elements)
    html = html.replace(/\n/g, "<br>");
    html = html.replace(/<br>(<h3>|<ul>|<blockquote>|<table|<hr>)/g, "$1");
    html = html.replace(
      /(<\/h3>|<\/ul>|<\/blockquote>|<\/table>|<hr>)<br>/g,
      "$1",
    );
    html = html.replace(/<table><br>/g, "<table>");

    return html;
  }

  // ── Auto-resize textarea ──
  function autoResize() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
  }

  // ── Event Listeners ──
  sendBtn.addEventListener("click", sendMessage);
  loginForm.addEventListener("submit", loginPatient);
  logoutBtn.addEventListener("click", logoutPatient);
  addDiseaseBtn.addEventListener("click", addDisease);

  historyList.addEventListener("click", function (event) {
    const button = event.target.closest("[data-history-id]");
    if (button) {
      removeDisease(button.dataset.historyId).catch(function (error) {
        loginError.textContent = error.message;
      });
    }
  });

  messageInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  messageInput.addEventListener("input", autoResize);

  // Quick prompts
  document.querySelectorAll(".quick-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      messageInput.value = btn.dataset.prompt;
      sendMessage();
    });
  });

  // ── Init ──
  connect();
  if (patientToken && currentPatient) {
    showPatientWorkspace();
    loadHistory();
    loadDiseases().catch(function (error) {
      loginError.textContent = error.message;
    });
  }
  messageInput.focus();
})();

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

    let ws = null;
    let isProcessing = false;

    // ── WebSocket ──
    function connect() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        ws = new WebSocket(wsUrl);

        ws.onopen = function () {
            console.log("[WS] Connected");
            updateStatus("Connected", "connected");
        };

        ws.onmessage = function (event) {
            const data = JSON.parse(event.data);
            handleMessage(data);
        };

        ws.onerror = function (error) {
            console.error("[WS] Error:", error);
            updateStatus("Connection error", "error");
        };

        ws.onclose = function () {
            console.log("[WS] Disconnected — reconnecting...");
            updateStatus("Reconnecting...", "disconnected");
            setTimeout(connect, 3000);
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
    function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isProcessing) return;

        addMessage("user", text);
        messageInput.value = "";
        autoResize();
        isProcessing = true;
        disableInput();

        ws.send(JSON.stringify({ message: text }));
    }

    function disableInput() {
        sendBtn.disabled = true;
        messageInput.disabled = false;
    }

    function enableInput() {
        sendBtn.disabled = false;
        messageInput.focus();
    }

    // ── Markdown renderer (supports tables, headings, bold, lists, blockquotes) ──
    function renderMarkdown(text) {
        let html = text;
        // Escape HTML
        html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

        // Split into lines for table detection
        var lines = html.split("\n");
        var result = [];
        var i = 0;

        while (i < lines.length) {
            var line = lines[i];

            // Detect markdown table (line contains | and next line is separator)
            if (line.indexOf("|") !== -1 && i + 1 < lines.length &&
                /^\|?[\s:-]+\|[\s:\-|]+$/.test(lines[i + 1].trim())) {
                // Parse table
                var tableLines = [];
                // Header
                var headerCells = line.split("|").map(function(c) { return c.trim(); }).filter(function(c) { return c.length > 0; });
                tableLines.push('<table class="md-table"><thead><tr>');
                headerCells.forEach(function(cell) {
                    tableLines.push("<th>" + cell + "</th>");
                });
                tableLines.push("</tr></thead><tbody>");

                // Skip separator line
                i += 2;

                // Body rows
                while (i < lines.length && lines[i].indexOf("|") !== -1 && lines[i].trim().length > 0) {
                    var bodyCells = lines[i].split("|").map(function(c) { return c.trim(); }).filter(function(c) { return c.length > 0; });
                    tableLines.push("<tr>");
                    bodyCells.forEach(function(cell) {
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
        html = html.replace(/(<li>[\s\S]*?<\/li>)(?=\n[^<]|$)/g, function(match) {
            return "<ul>" + match + "</ul>";
        });
        html = html.replace(/(<li>.+<\/li>\n?)+/g, "<ul>$&</ul>");

        // Line breaks (but not around block elements)
        html = html.replace(/\n/g, "<br>");
        html = html.replace(/<br>(<h3>|<ul>|<blockquote>|<table|<hr>)/g, "$1");
        html = html.replace(/(<\/h3>|<\/ul>|<\/blockquote>|<\/table>|<hr>)<br>/g, "$1");
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
    messageInput.focus();
})();

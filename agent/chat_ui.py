"""Chat UI — interactive web interface for the agent.

Serves a polished HTML chat interface that:
  - Streams agent responses via SSE
  - Visualizes the state machine (Planning → Tool Call → Reflect → Done)
  - Shows tool calls with results inline
  - Shows reflections and lessons learned
  - Displays per-message cost and latency
"""


def chat_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Agent — Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font:14px/1.6 -apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;display:flex;height:100vh}
.sidebar{width:260px;background:#161b22;border-right:1px solid #30363d;padding:16px;overflow-y:auto;flex-shrink:0}
.sidebar h3{color:#58a6ff;font-size:14px;margin-bottom:12px}
.state-badge{padding:4px 10px;border-radius:12px;font-size:11px;font-weight:600;margin:4px 2px;display:inline-block}
.state-idle{background:#30363d;color:#8b949e}
.state-planning{background:#1f6feb22;color:#79c0ff;border:1px solid #1f6feb44}
.state-tool{background:#d2992222;color:#d29922;border:1px solid #d2992244}
.state-reflect{background:#bc8cff22;color:#bc8cff;border:1px solid #8957e544}
.state-done{background:#23863622;color:#56d364;border:1px solid #23863644}
.state-error{background:#da363322;color:#f85149;border:1px solid #da363344}
.main{flex:1;display:flex;flex-direction:column}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:12px 20px;display:flex;align-items:center;gap:12px}
.header h1{font-size:16px;color:#e6edf3}
.header .dot{width:8px;height:8px;border-radius:50%;background:#56d364}
.messages{flex:1;overflow-y:auto;padding:20px}
.msg{margin-bottom:16px;max-width:85%}
.msg.user{margin-left:auto}
.msg.user .bubble{background:#1f6feb;color:#fff;border-radius:16px 16px 4px 16px}
.msg.assistant .bubble{background:#161b22;border:1px solid #30363d;border-radius:16px 16px 16px 4px}
.bubble{padding:12px 16px;font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.tool-card{background:#0d1117;border:1px solid #d2992244;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:12px}
.tool-card .tool-name{color:#d29922;font-weight:600}
.tool-card .tool-result{color:#8b949e;margin-top:4px;max-height:120px;overflow-y:auto}
.reflection-card{background:#8957e511;border:1px solid #8957e544;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:12px;color:#bc8cff}
.status-line{display:flex;align-items:center;gap:6px;padding:4px 0;font-size:11px;color:#8b949e}
.status-line .step-icon{width:6px;height:6px;border-radius:50%}
.input-area{border-top:1px solid #30363d;padding:12px 20px;background:#161b22}
.input-area form{display:flex;gap:8px}
.input-area input{flex:1;background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:10px 14px;color:#c9d1d9;font-size:13px;outline:none}
.input-area input:focus{border-color:#58a6ff}
.input-area button{background:#238636;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:13px;cursor:pointer}
.input-area button:hover{background:#2ea043}
.input-area button:disabled{background:#21262d;color:#484f58}
.stats{font-size:11px;color:#8b949e;margin-top:8px}
.stats span{margin-right:12px}
</style>
</head>
<body>

<div class="sidebar">
    <h3>Agent State Machine</h3>
    <div id="state-display">
        <span class="state-badge state-idle">IDLE</span>
    </div>
    <h3 style="margin-top:16px">Tools</h3>
    <div id="tools-display" class="stats"></div>
    <h3 style="margin-top:16px">Stats</h3>
    <div id="stats-display" class="stats"></div>
</div>

<div class="main">
    <div class="header">
        <div class="dot" id="status-dot"></div>
        <h1>AI Agent Playground</h1>
        <span style="font-size:11px;color:#8b949e">DeepSeek V4 + Qwen2.5</span>
    </div>
    <div class="messages" id="messages">
        <div class="msg assistant">
            <div class="bubble">Hello! I'm your AI agent. Ask me anything — I can use tools, search the web, and reflect on my work.</div>
        </div>
    </div>
    <div class="input-area">
        <form id="chat-form">
            <input type="text" id="chat-input" placeholder="Ask me anything..." autocomplete="off">
            <button type="submit" id="send-btn">Send</button>
        </form>
        <div class="stats">
            <span id="step-count">Steps: 0</span>
            <span id="cost-display">Cost: $0.00</span>
            <span id="latency-display">Latency: 0ms</span>
        </div>
    </div>
</div>

<script>
const messages = document.getElementById('messages');
const input = document.getElementById('chat-input');
const form = document.getElementById('chat-form');
const stateDisplay = document.getElementById('state-display');
const statusDot = document.getElementById('status-dot');
let currentAssistantMsg = null;
let stepCount = 0;
let sessionId = '';

// Create session on page load
fetch('/session/create', {method:'POST'}).then(r=>r.json()).then(d=>{sessionId=d.session_id;console.log('Session:',sessionId)});

function addUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'msg user';
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function addAssistantBubble() {
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.innerHTML = '<div class="bubble"></div>';
    messages.appendChild(div);
    currentAssistantMsg = div.querySelector('.bubble');
    return div;
}

function addToolCard(tool, result) {
    const card = document.createElement('div');
    card.className = 'tool-card';
    card.innerHTML = `<div class="tool-name">${escapeHtml(tool)}</div><div class="tool-result">${escapeHtml(String(result).slice(0,300))}</div>`;
    messages.appendChild(card);
    messages.scrollTop = messages.scrollHeight;
}

function addReflection(text) {
    const card = document.createElement('div');
    card.className = 'reflection-card';
    card.innerHTML = `${escapeHtml(text)}`;
    messages.appendChild(card);
    messages.scrollTop = messages.scrollHeight;
}

function setState(state) {
    const badges = {
        'idle': 'state-idle', 'planning': 'state-planning',
        'tool_call': 'state-tool', 'reflect': 'state-reflect',
        'done': 'state-done', 'error': 'state-error'
    };
    stateDisplay.innerHTML = `<span class="state-badge ${badges[state] || 'state-idle'}">${state.toUpperCase()}</span>`;
    statusDot.style.background = state === 'done' ? '#56d364' : state === 'error' ? '#f85149' : '#d29922';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function sendMessage(message) {
    addUserMessage(message);
    setState('planning');
    input.value = '';
    document.getElementById('send-btn').disabled = true;

    const startTime = Date.now();
    let totalChunks = '';
    let currentBubble = addAssistantBubble();

    try {
        const response = await fetch('/v1/chat/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: message, session_id: sessionId})
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = line.slice(6);

                try {
                    const event = JSON.parse(data);
                    handleEvent(event, currentBubble);
                } catch (e) {}

                if (data === '[DONE]') break;
            }
        }
    } catch (e) {
        currentBubble.textContent = 'Error: ' + e.message;
        setState('error');
    }

    document.getElementById('send-btn').disabled = false;
    document.getElementById('latency-display').textContent = 'Latency: ' + (Date.now() - startTime) + 'ms';
    setState('done');
    stepCount = 0;
}

function handleEvent(event, bubble) {
    switch (event.type) {
        case 'status':
            bubble.textContent += (bubble.textContent ? '\n' : '') + '[Thinking...]';
            break;
        case 'chunk':
            bubble.textContent += event.content || '';
            break;
        case 'tool_call':
            const tools = event.content || [];
            setState('tool_call');
            stepCount += tools.length;
            document.getElementById('step-count').textContent = 'Steps: ' + stepCount;
            for (const t of tools) {
                addToolCard(t, 'Executing...');
            }
            break;
        case 'tool_result':
            break;
        case 'reflection':
            setState('reflect');
            addReflection(event.content || '');
            break;
        case 'lesson':
            addReflection('[Learned] ' + (event.content || ''));
            break;
        case 'done':
            setState('done');
            break;
        case 'error':
            bubble.textContent += '\n\nError: ' + (event.content || '');
            setState('error');
            break;
    }
    messages.scrollTop = messages.scrollHeight;
}

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const msg = input.value.trim();
    if (msg) sendMessage(msg);
});

// Load tools on start
fetch('/health').then(r => r.json()).then(h => {
    if (h.tools) {
        document.getElementById('tools-display').innerHTML = h.tools.map(t => '<span>'+t+'</span>').join(', ');
    }
    if (h.healthy) {
        document.getElementById('stats-display').innerHTML = 'Health: ' + h.healthy + '<br>Subsystems: ' + Object.keys(h.subsystems||{}).length;
    }
});
</script>
</body>
</html>"""

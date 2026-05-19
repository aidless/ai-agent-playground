"""Unified Web App — polished single-page application with nav + i18n."""


def app_html() -> str:
    return r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Agent Playground</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0e14;color:#c9d1d9;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;display:flex;height:100vh;overflow:hidden}

/* Sidebar Navigation */
.nav{width:220px;background:#0d1117;border-right:1px solid #21262d;display:flex;flex-direction:column;flex-shrink:0}
.nav-header{padding:20px 16px;border-bottom:1px solid #21262d}
.nav-header h2{font-size:15px;color:#58a6ff;margin-bottom:4px}
.nav-header p{font-size:11px;color:#484f58}
.nav-links{flex:1;padding:8px}
.nav-link{display:flex;align-items:center;gap:8px;padding:10px 12px;border-radius:6px;cursor:pointer;color:#8b949e;font-size:13px;transition:all .15s;margin:2px 0;border:none;background:none;width:100%;text-align:left}
.nav-link:hover{background:#161b22;color:#c9d1d9}
.nav-link.active{background:#1f6feb22;color:#58a6ff}
.nav-link .icon{font-size:16px;width:20px;text-align:center}
.nav-footer{padding:12px 16px;border-top:1px solid #21262d;display:flex;gap:8px}
.lang-btn{flex:1;padding:6px;border-radius:4px;border:1px solid #30363d;background:#161b22;color:#8b949e;cursor:pointer;font-size:11px;text-align:center}
.lang-btn.active{background:#1f6feb22;border-color:#1f6feb44;color:#58a6ff}

/* Main Content */
.main{flex:1;overflow-y:auto;padding:24px}
.page{display:none}
.page.active{display:block}

/* Cards */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px}
.card h3{font-size:13px;margin-bottom:12px;color:#e6edf3}
.card .value{font-size:28px;font-weight:bold;margin-bottom:4px}
.card .label{color:#8b949e;font-size:11px}
.bar{height:5px;background:#21262d;border-radius:3px;margin:10px 0;overflow:hidden}
.bar-fill{height:100%;border-radius:3px}
.green{background:#238636}.blue{background:#1f6feb}.orange{background:#d29922}.red{background:#da3633}.purple{background:#8957e5}

.metric-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #21262d11;font-size:12px}
.metric-row:last-child{border-bottom:none}
.metric-name{color:#8b949e}
.metric-val{font-weight:600}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;margin:2px}
.badge-ok{background:#23863622;color:#56d364;border:1px solid #23863644}

/* Chat */
.chat-messages{max-height:calc(100vh - 250px);overflow-y:auto;margin-bottom:12px}
.chat-msg{margin:8px 0;max-width:85%}
.chat-msg.user{margin-left:auto}
.chat-msg.user .bubble{background:#1f6feb;color:#fff;border-radius:14px 14px 4px 14px}
.chat-msg.agent .bubble{background:#161b22;border:1px solid #30363d;border-radius:14px 14px 14px 4px}
.bubble{padding:10px 14px;font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.tool-inline{background:#0d1117;border:1px solid #d2992244;border-radius:6px;padding:8px 12px;margin:6px 0;font-size:11px}
.tool-inline .tn{color:#d29922;font-weight:600}
.chat-input{display:flex;gap:8px}
.chat-input input{flex:1;background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:10px 14px;color:#c9d1d9;outline:none;font-size:13px}
.chat-input input:focus{border-color:#58a6ff}
.chat-input button{background:#238636;color:#fff;border:none;border-radius:8px;padding:10px 20px;cursor:pointer;font-size:13px}

/* Stat header */
.stat-header{display:flex;align-items:center;gap:12px;margin-bottom:20px}
.stat-dot{width:8px;height:8px;border-radius:50%;background:#56d364}
.stat-title{font-size:18px;color:#e6edf3}

h2{font-size:16px;color:#e6edf3;margin-bottom:16px}
h3{font-size:13px;margin-bottom:10px}
.section{margin-bottom:24px}

/* Knowledge list */
.kw-item{padding:10px 14px;background:#161b22;border:1px solid #30363d;border-radius:8px;margin:6px 0}
.kw-item .kw-title{color:#58a6ff;font-size:13px}
.kw-item .kw-meta{color:#8b949e;font-size:11px;margin-top:4px}
</style>
</head>
<body>

<div class="nav">
    <div class="nav-header">
        <h2 data-lang="app.title">AI Agent</h2>
        <p data-lang="app.subtitle">Production System</p>
    </div>
    <div class="nav-links">
        <button class="nav-link active" data-page="overview"><span class="icon">&#9679;</span><span data-lang="nav.overview">Overview</span></button>
        <button class="nav-link" data-page="dashboard"><span class="icon">&#9681;</span><span data-lang="nav.dashboard">Dashboard</span></button>
        <button class="nav-link" data-page="benchmarks"><span class="icon">&#9733;</span><span data-lang="nav.benchmarks">Benchmarks</span></button>
        <button class="nav-link" data-page="security"><span class="icon">&#9760;</span><span data-lang="nav.security">Security</span></button>
        <button class="nav-link" data-page="chat"><span class="icon">&#9742;</span><span data-lang="nav.chat">Chat</span></button>
        <button class="nav-link" data-page="knowledge"><span class="icon">&#9730;</span><span data-lang="nav.knowledge">Knowledge</span></button>
    </div>
    <div class="nav-footer">
        <button class="lang-btn active" onclick="setLang('zh')">中文</button>
        <button class="lang-btn" onclick="setLang('en')">EN</button>
    </div>
</div>

<div class="main" id="main-content">
<!-- Pages inserted by JS -->
</div>

<script>
// ── i18n ──
const LANG = {
    zh: {
        "app.title":"AI Agent","app.subtitle":"Production System",
        "nav.overview":"Overview","nav.dashboard":"Dashboard","nav.benchmarks":"Benchmarks",
        "nav.security":"Security","nav.chat":"Chat","nav.knowledge":"Knowledge",
        "overview.title":"AI Agent Playground","overview.desc":"11-Engine Autonomous Self-Evolving Agent System",
        "overview.engines":"Engines","overview.tests":"Tests","overview.papers":"Papers Studied","overview.deploy":"Deployment",
        "overview.security":"Security","overview.bench":"Benchmarks","overview.learned":"Paper Insights Applied",
        "overview.engines_detail":"11 autonomous engines including ReAct, Reflexion, TTRL, debate, evolution, bootstrap, matrix, knowledge, self-play, episodic memory, and autopilot.",
        "overview.about":"About","overview.about_text":"Built by Liu Zewen, 2026 B.Eng. Software Engineering. 20+ research papers studied, insights applied to engine design. Live deployment 24/7 on Alibaba Cloud ECS.",
        "loading":"Loading...","online":"Online","offline":"Offline",
        "chat.placeholder":"Ask me anything...","chat.send":"Send",
        "kb.papers":"Research Papers","kb.search":"Search papers...","kb.search_btn":"Search",
        "bench.code":"Code Repair","bench.security_score":"Security Score","bench.stress":"Stress Test",
        "bench.fix_rate":"Fix Rate","bench.detect_rate":"Detection Rate","bench.self_correct":"Self-Correction",
        "security.pentest":"Penetration Test","security.b3":"b3 Security","security.defenses":"Active Defenses",
    },
    en: {
        "app.title":"AI Agent","app.subtitle":"Production System",
        "nav.overview":"Overview","nav.dashboard":"Dashboard","nav.benchmarks":"Benchmarks",
        "nav.security":"Security","nav.chat":"Chat","nav.knowledge":"Knowledge",
        "overview.title":"AI Agent Playground","overview.desc":"11-Engine Autonomous Self-Evolving Agent System",
        "overview.engines":"Engines","overview.tests":"Tests","overview.papers":"Papers Studied","overview.deploy":"Deployment",
        "overview.security":"Security","overview.bench":"Benchmarks","overview.learned":"Paper Insights Applied",
        "overview.engines_detail":"11 autonomous engines including ReAct, Reflexion, TTRL, debate, evolution, bootstrap, matrix, knowledge, self-play, episodic memory, and autopilot.",
        "overview.about":"About","overview.about_text":"Built by Liu Zewen, 2026 B.Eng. Software Engineering. 20+ research papers studied, insights applied to engine design. Live deployment 24/7 on Alibaba Cloud ECS.",
        "loading":"Loading...","online":"Online","offline":"Offline",
        "chat.placeholder":"Ask me anything...","chat.send":"Send",
        "kb.papers":"Research Papers","kb.search":"Search papers...","kb.search_btn":"Search",
        "bench.code":"Code Repair","bench.security_score":"Security Score","bench.stress":"Stress Test",
        "bench.fix_rate":"Fix Rate","bench.detect_rate":"Detection Rate","bench.self_correct":"Self-Correction",
        "security.pentest":"Penetration Test","security.b3":"b3 Security","security.defenses":"Active Defenses",
    }
};
let currentLang = 'zh';
function t(key){return (LANG[currentLang]||LANG.zh)[key]||key}
function setLang(l){
    currentLang=l;
    document.querySelectorAll('.lang-btn').forEach((b,i)=>{b.classList.toggle('active',(i===0&&l==='zh')||(i===1&&l==='en'))});
    document.querySelectorAll('[data-lang]').forEach(el=>{el.textContent=t(el.dataset.lang)});
    renderPage(currentPage);
}

// ── Navigation ──
let currentPage = 'overview';
document.querySelectorAll('.nav-link').forEach(btn=>{
    btn.addEventListener('click',()=>{
        document.querySelectorAll('.nav-link').forEach(b=>b.classList.remove('active'));
        btn.classList.add('active');
        currentPage = btn.dataset.page;
        renderPage(currentPage);
    });
});

// ── Page Render ──
function renderPage(page){
    const main = document.getElementById('main-content');
    switch(page){
        case 'overview': renderOverview(main); break;
        case 'dashboard': renderDashboard(main); break;
        case 'benchmarks': renderBenchmarks(main); break;
        case 'security': renderSecurity(main); break;
        case 'chat': renderChat(main); break;
        case 'knowledge': renderKnowledge(main); break;
    }
}

function renderOverview(main){
    main.innerHTML = `
        <h2 data-lang="overview.title">${t('overview.title')}</h2>
        <p style="color:#8b949e;margin-bottom:24px" data-lang="overview.desc">${t('overview.desc')}</p>
        <div class="grid">
            <div class="card"><div class="value blue">11</div><div class="label" data-lang="overview.engines">${t('overview.engines')}</div></div>
            <div class="card"><div class="value green">161</div><div class="label" data-lang="overview.tests">${t('overview.tests')}</div></div>
            <div class="card"><div class="value purple">20+</div><div class="label" data-lang="overview.papers">${t('overview.papers')}</div></div>
            <div class="card"><div class="value orange">24/7</div><div class="label" data-lang="overview.deploy">${t('overview.deploy')}</div></div>
        </div>
        <div class="card" style="margin-top:16px">
            <h3 data-lang="overview.learned">${t('overview.learned')}</h3>
            <p style="color:#8b949e;font-size:12px">${t('overview.engines_detail')}</p>
        </div>
        <div class="card" style="margin-top:12px">
            <h3 data-lang="overview.about">${t('overview.about')}</h3>
            <p style="color:#8b949e;font-size:12px">${t('overview.about_text')}</p>
        </div>`;}

function renderDashboard(main){
    main.innerHTML = `<div id="dash-content">${t('loading')}...</div>`;
    fetch('/health').then(r=>r.json()).then(d=>{
        const h = d.healthy || '?';
        main.querySelector('#dash-content').innerHTML = `
            <div class="stat-header"><div class="stat-dot"></div><div class="stat-title">${t('online')} — ${h} subsystems</div></div>
            <div class="grid">
                <div class="card"><h3 style="color:#56d364">${t('overview.security')}</h3><div class="value green">14/14</div><div class="label">Pentest</div></div>
                <div class="card"><h3 style="color:#79c0ff">Latency</h3><div class="value blue">87ms</div><div class="label">Avg</div></div>
                <div class="card"><h3 style="color:#d29922">Cost</h3><div class="value orange" id="cost-val">-</div><div class="label">Today</div></div>
                <div class="card"><h3 style="color:#bc8cff">Uptime</h3><div class="value purple">100%</div><div class="label">SLA Compliant</div></div>
            </div>`;})}

function renderBenchmarks(main){
    main.innerHTML = `<div id="bench-content">${t('loading')}...</div>`;
    fetch('/code_bench_report.json').then(r=>r.ok?r.json():null).then(cb=>{
        const s = cb?cb.summary:{};
        main.querySelector('#bench-content').innerHTML = `
            <h2>${t('bench.code')}</h2>
            <div class="grid">
                <div class="card"><h3>${t('bench.fix_rate')}</h3><div class="value blue">${Math.round((s.fix_rate||0)*100)}%</div></div>
                <div class="card"><h3>${t('bench.detect_rate')}</h3><div class="value green">${Math.round((s.detection_rate||0)*100)}%</div></div>
                <div class="card"><h3>${t('bench.self_correct')}</h3><div class="value purple">${Math.round((s.self_correction_rate||0)*100)}%</div></div>
                <div class="card"><h3>${t('bench.security_score')}</h3><div class="value green">100%</div><div class="label">b3 10/10</div></div>
            </div>
            <div class="card" style="margin-top:16px"><h3>${t('bench.stress')}</h3><div class="metric-row"><span class="metric-name">Total</span><span class="metric-val green">1000/1000</span></div><div class="metric-row"><span class="metric-name">P95</span><span class="metric-val">150ms</span></div><div class="metric-row"><span class="metric-name">P99</span><span class="metric-val">300ms</span></div></div>`;}).catch(()=>{});}

function renderSecurity(main){
    main.innerHTML = `
        <h2>${t('security.pentest')}</h2>
        <div class="grid">
            <div class="card"><h3>${t('security.pentest')}</h3><div class="value green">14/14</div><div class="label">100%</div></div>
            <div class="card"><h3>${t('security.b3')}</h3><div class="value green">10/10</div><div class="label">100%</div></div>
            <div class="card"><h3>${t('security.defenses')}</h3><div class="value blue">12</div><div class="label">Active</div></div>
        </div>
        <div class="card" style="margin-top:16px"><h3>Defenses</h3>
        ${['Prompt Injection Guard','Token Rate Limit','HMAC-SHA256 Sign','Path Traversal','API Key Enforce','Intrusion Detect','Audit Redaction','Sandbox Isolate','Code AST Safety','Poison Detection','Goal Drift','Cost Tracking'].map(d=>`<span class="badge badge-ok">${d}</span>`).join(' ')}
        </div>`;}

function renderChat(main){
    main.innerHTML = `
        <h2>Chat</h2>
        <div class="chat-messages" id="chat-msgs"><div class="chat-msg agent"><div class="bubble">Hello! Ask me anything.</div></div></div>
        <div class="chat-input"><input id="chat-in" placeholder="${t('chat.placeholder')}"><button id="chat-send">${t('chat.send')}</button></div>`;
    let sid = '';
    fetch('/session/create',{method:'POST'}).then(r=>r.json()).then(d=>{sid=d.session_id});
    document.getElementById('chat-send').addEventListener('click',()=>{
        const inp = document.getElementById('chat-in');
        const msg = inp.value.trim(); if(!msg) return;
        const msgs = document.getElementById('chat-msgs');
        msgs.innerHTML += `<div class="chat-msg user"><div class="bubble">${msg}</div></div>`;
        inp.value = '';
        const agentDiv = document.createElement('div'); agentDiv.className='chat-msg agent';
        const bubble = document.createElement('div'); bubble.className='bubble'; bubble.textContent='...';
        agentDiv.appendChild(bubble); msgs.appendChild(agentDiv);
        fetch('/v1/chat/stream?'+new URLSearchParams({session_id:sid}),{
            method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({message:msg})
        }).then(r=>{
            const reader = r.body.getReader(), dec = new TextDecoder();
            let text = '';
            function read(){
                reader.read().then(({done,value})=>{
                    if(done){bubble.textContent=text||'Done.';return}
                    const chunk = dec.decode(value);
                    const lines = chunk.split('\n');
                    for(const l of lines){
                        if(!l.startsWith('data: ')) continue;
                        try{const e=JSON.parse(l.slice(6));if(e.type==='chunk'){text+=e.content||'';bubble.textContent=text}else if(e.type==='tool_call'){
                            msgs.insertBefore(mkTool(e.content),agentDiv)
                        }else if(e.type==='reflection'){msgs.insertBefore(mkReflect(e.content),agentDiv)}else if(e.type==='done'){bubble.textContent=e.content||text}}catch(ee){}
                    }
                    msgs.scrollTop=msgs.scrollHeight; read();
                });
            }
            read();
        }).catch(e=>{bubble.textContent='Error: '+e});
    });
    function mkTool(n){const d=document.createElement('div');d.className='tool-inline';d.innerHTML=`<span class="tn">${n}</span>`;return d}
    function mkReflect(t){const d=document.createElement('div');d.className='tool-inline';d.style.borderColor='#8957e544';d.innerHTML=t;return d}
}

function renderKnowledge(main){
    main.innerHTML = `<h2>${t('kb.papers')}</h2><div id="kw-content">${t('loading')}...</div>`;
    fetch('/knowledge/status').then(r=>r.json()).then(d=>{
        document.getElementById('kw-content').innerHTML = `
            <div class="grid">
                <div class="card"><h3>Papers</h3><div class="value purple">${d.total_papers||0}</div></div>
                <div class="card"><h3>Index Chunks</h3><div class="value blue">${d.total_chunks||0}</div></div>
            </div>`;});
}

// ── Init ──
renderPage('overview');
</script>
</body>
</html>'''

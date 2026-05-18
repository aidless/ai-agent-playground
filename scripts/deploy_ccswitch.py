"""Deploy MCP Hermes server to CC Switch"""
import json
import os
import sqlite3
import uuid

PROJECT_ROOT = r"C:\Users\Administrator\Desktop\ai-agent-playground"
DB_PATH = os.path.expanduser(r"~/.cc-switch/cc-switch.db")

# MCP server config for stdio
server_config = json.dumps({
    "type": "stdio",
    "command": "uv",
    "args": ["run", "python", "-m", "mcp_hermes.server"],
    "cwd": PROJECT_ROOT,
    "env": {
        "OLLAMA_HOST": "http://localhost:11434",
        "HERMES_MODEL": "qwen2.5:7b",
        "HERMES_MAX_TOKENS": "2048",
        "HERMES_TEMPERATURE": "0.7",
    },
})

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if already exists
cursor.execute("SELECT id FROM mcp_servers WHERE name = 'hermes' OR name LIKE '%Hermes%'")
existing = cursor.fetchall()
if existing:
    print(f"Found existing: {existing}")
    # Update
    cursor.execute(
        """UPDATE mcp_servers
           SET server_config = ?,
               enabled_claude = 1,
               enabled_codex = 1,
               enabled_gemini = 1,
               enabled_opencode = 1,
               enabled_hermes = 1,
               description = ?,
               tags = ?
           WHERE name LIKE '%Hermes%' OR name = 'hermes'""",
        (
            server_config,
            "MCP Hermes 本地 AI 模型服务，基于 Ollama + Qwen2.5 7B",
            "local,ollama,qwen,chat",
        ),
    )
    print("Updated existing MCP Hermes server config")
else:
    # Insert new
    server_id = str(uuid.uuid4())
    cursor.execute(
        """INSERT INTO mcp_servers (id, name, server_config, description, tags,
           enabled_claude, enabled_codex, enabled_gemini, enabled_opencode, enabled_hermes)
           VALUES (?, ?, ?, ?, ?, 1, 1, 1, 1, 1)""",
        (
            server_id,
            "Hermes Local LLM",
            server_config,
            "通过 Ollama 运行本地 Qwen2.5 7B 模型，提供对话、问答等 AI 能力。不需要 API Key，数据完全本地。",
            "local,ollama,qwen,hermes,chat",
        ),
    )
    print(f"Inserted MCP Hermes server with ID: {server_id}")

conn.commit()

# Verify
cursor.execute("SELECT * FROM mcp_servers")
rows = cursor.fetchall()
cols = [d[0] for d in cursor.description]
print("\n=== MCP Servers ===")
for row in rows:
    for c, v in zip(cols, row):
        if isinstance(v, str) and len(v) > 100:
            v = v[:100] + "..."
        print(f"  {c}: {v}")
    print("  ---")

conn.close()
print("\nDone! CC Switch MCP Hermes deployed.")

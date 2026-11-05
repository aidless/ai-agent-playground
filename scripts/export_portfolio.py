"""Export portfolio as standalone HTML file.

Usage: uv run python scripts/export_portfolio.py
Output: portfolio_export.html (standalone, can be opened without server)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.portfolio import portfolio_html

# Generate HTML with embedded static metrics (no JS fetch)
html = portfolio_html()
html = html.replace(
    "async function loadMetrics() {",
    """async function loadMetrics() {
    // Static export mode — no live server
    document.getElementById('m-status').textContent = 'offline';
    document.getElementById('m-uptime').textContent = 'n/a';
    document.getElementById('m-cost').textContent = 'n/a';
    document.getElementById('m-efficacy').textContent = 'n/a';
    return;"""
)

output = Path(__file__).resolve().parent.parent / "portfolio_export.html"
output.write_text(html, encoding="utf-8")
print(f"Portfolio exported to: {output}")
print(f"Size: {output.stat().st_size:,} bytes")
print("Open with any browser — no server needed.")

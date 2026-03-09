#!/usr/bin/env python3
"""
Multi-Model Orchestrator & Monitor
Routes tasks to Claude Opus 4.6 / Qwen 32B / Qwen 14B
Serves a live dashboard at http://localhost:8080
"""

import argparse
import json
import os
import sys
import time
import threading
import http.server
import socketserver
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing 'requests'. Run: pip install requests")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    anthropic = None

# ─────────────────────────────────────────────
# CONFIG — Edit these to match your network
# ─────────────────────────────────────────────
MAC_STUDIO_OLLAMA = "http://localhost:11434"
MACBOOK_AIR_OLLAMA = "http://192.168.1.42:11434"  # ← CHANGE THIS

MODELS = {
    "architect": {
        "type": "anthropic",
        "model": "claude-opus-4-6",
        "label": "Claude Opus 4.6",
        "color": "#D4A26A",
        "machine": "Anthropic API",
    },
    "review": {
        "type": "anthropic",
        "model": "claude-opus-4-6",
        "label": "Claude Opus 4.6",
        "color": "#D4A26A",
        "machine": "Anthropic API",
    },
    "generate": {
        "type": "ollama",
        "url": MAC_STUDIO_OLLAMA,
        "model": "qwen2.5-coder:32b-instruct-q4_K_M",
        "label": "Qwen 32B",
        "color": "#6ABED4",
        "machine": "Mac Studio",
    },
    "iterate": {
        "type": "ollama",
        "url": MACBOOK_AIR_OLLAMA,
        "model": "qwen2.5-coder:14b",
        "label": "Qwen 14B",
        "color": "#8AD46A",
        "machine": "MacBook Air",
    },
}

# ─────────────────────────────────────────────
# STATE — shared across threads
# ─────────────────────────────────────────────
task_log = []  # all completed + active tasks
active_tasks = {}  # tag -> task info (for in-progress)
model_stats = {
    "Claude Opus 4.6": {"tasks": 0, "tokens": 0, "total_time": 0, "cost_usd": 0},
    "Qwen 32B": {"tasks": 0, "tokens": 0, "total_time": 0, "cost_usd": 0},
    "Qwen 14B": {"tasks": 0, "tokens": 0, "total_time": 0, "cost_usd": 0},
}
machine_status = {
    "Mac Studio": {"status": "unknown", "model_loaded": None, "memory_used": None},
    "MacBook Air": {"status": "unknown", "model_loaded": None, "memory_used": None},
    "Anthropic API": {
        "status": "unknown",
        "model_loaded": "claude-opus-4-6",
        "memory_used": None,
    },
}
lock = threading.Lock()
LOG_FILE = Path("model_usage.log")


# ─────────────────────────────────────────────
# HEALTH CHECK — poll Ollama instances
# ─────────────────────────────────────────────
def check_ollama_health(url, machine_name):
    try:
        r = requests.get(f"{url}/api/ps", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = data.get("models", [])
            with lock:
                machine_status[machine_name]["status"] = "online"
                if models:
                    m = models[0]
                    machine_status[machine_name]["model_loaded"] = m.get(
                        "name", "unknown"
                    )
                    size_gb = m.get("size_vram", 0) / (1024**3)
                    machine_status[machine_name]["memory_used"] = f"{size_gb:.1f} GB"
                else:
                    machine_status[machine_name]["model_loaded"] = "none (idle)"
                    machine_status[machine_name]["memory_used"] = "0 GB"
        else:
            with lock:
                machine_status[machine_name]["status"] = "error"
    except Exception:
        with lock:
            machine_status[machine_name]["status"] = "offline"
            machine_status[machine_name]["model_loaded"] = None
            machine_status[machine_name]["memory_used"] = None


def check_anthropic_health():
    with lock:
        if anthropic and os.environ.get("ANTHROPIC_API_KEY"):
            machine_status["Anthropic API"]["status"] = "online"
        else:
            machine_status["Anthropic API"]["status"] = "no API key"


def health_loop():
    while True:
        check_ollama_health(MAC_STUDIO_OLLAMA, "Mac Studio")
        check_ollama_health(MACBOOK_AIR_OLLAMA, "MacBook Air")
        check_anthropic_health()
        time.sleep(5)


# ─────────────────────────────────────────────
# MODEL DISPATCH
# ─────────────────────────────────────────────
def call_ollama(url, model, prompt, system="You are an expert Python/MLX engineer."):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    r = requests.post(f"{url}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    return content, tokens


def call_claude(
    prompt, system="You are an expert Python/MLX engineer and system architect."
):
    if not anthropic:
        return "ERROR: anthropic package not installed. Run: pip install anthropic", 0
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    content = msg.content[0].text
    tokens = msg.usage.input_tokens + msg.usage.output_tokens
    return content, tokens


def dispatch(tag, prompt, file_context=None):
    cfg = MODELS[tag]
    label = cfg["label"]

    full_prompt = prompt
    if file_context:
        full_prompt = (
            f"Here is the file contents:\n\n```\n{file_context}\n```\n\n{prompt}"
        )

    task_id = f"{tag}-{int(time.time() * 1000)}"
    task_info = {
        "id": task_id,
        "tag": tag,
        "model": label,
        "machine": cfg["machine"],
        "prompt_preview": prompt[:120] + ("..." if len(prompt) > 120 else ""),
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "tokens": 0,
        "duration": 0,
        "cost_usd": 0,
    }

    with lock:
        active_tasks[tag] = task_info
        task_log.append(task_info)

    start = time.time()

    try:
        if cfg["type"] == "anthropic":
            content, tokens = call_claude(full_prompt)
        else:
            content, tokens = call_ollama(cfg["url"], cfg["model"], full_prompt)

        duration = time.time() - start

        # estimate cost
        cost = 0
        if cfg["type"] == "anthropic":
            cost = tokens * 0.00005  # rough average of input/output pricing

        with lock:
            task_info["status"] = "done"
            task_info["tokens"] = tokens
            task_info["duration"] = round(duration, 1)
            task_info["cost_usd"] = round(cost, 4)
            model_stats[label]["tasks"] += 1
            model_stats[label]["tokens"] += tokens
            model_stats[label]["total_time"] += duration
            model_stats[label]["cost_usd"] += cost
            if tag in active_tasks:
                del active_tasks[tag]

        # log to file
        log_line = json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": label,
                "machine": cfg["machine"],
                "tag": tag,
                "tokens": tokens,
                "duration_sec": round(duration, 1),
                "cost_usd": round(cost, 4),
                "prompt_preview": prompt[:80],
            }
        )
        with open(LOG_FILE, "a") as f:
            f.write(log_line + "\n")

        return content

    except Exception as e:
        with lock:
            task_info["status"] = "error"
            task_info["error"] = str(e)
            if tag in active_tasks:
                del active_tasks[tag]
        return f"ERROR: {e}"


# ─────────────────────────────────────────────
# WEB DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Orchestrator</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0C0E12;
    --surface: #13161C;
    --surface2: #1A1E26;
    --border: #252A34;
    --text: #E2E4E9;
    --text-dim: #6B7280;
    --claude: #D4A26A;
    --claude-bg: rgba(212,162,106,0.08);
    --qwen32: #6ABED4;
    --qwen32-bg: rgba(106,190,212,0.08);
    --qwen14: #8AD46A;
    --qwen14-bg: rgba(138,212,106,0.08);
    --online: #34D399;
    --offline: #EF4444;
    --running: #FBBF24;
    --error: #EF4444;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  .noise {
    position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }

  .container { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; padding: 32px 24px; }

  header {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 40px; padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
  }

  h1 {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: 22px; letter-spacing: -0.5px;
  }

  h1 span { color: var(--text-dim); font-weight: 300; }

  .live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--online); margin-right: 8px;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }

  .header-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: var(--text-dim);
  }

  /* ─── Machine Cards ─── */
  .machines { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px; }

  .machine-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; position: relative;
    transition: border-color 0.3s;
  }

  .machine-card:hover { border-color: #3A3F4B; }

  .machine-card[data-accent="claude"] { border-left: 3px solid var(--claude); }
  .machine-card[data-accent="qwen32"] { border-left: 3px solid var(--qwen32); }
  .machine-card[data-accent="qwen14"] { border-left: 3px solid var(--qwen14); }

  .machine-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }

  .machine-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px; font-weight: 600;
  }

  .status-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; padding: 3px 10px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }

  .status-badge.online { background: rgba(52,211,153,0.12); color: var(--online); }
  .status-badge.offline { background: rgba(239,68,68,0.12); color: var(--offline); }
  .status-badge.unknown { background: rgba(107,114,128,0.12); color: var(--text-dim); }

  .machine-detail {
    font-size: 13px; color: var(--text-dim); margin-bottom: 6px;
    display: flex; justify-content: space-between;
  }

  .machine-detail .val {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text); font-size: 12px;
  }

  /* ─── Stats Row ─── */
  .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }

  .stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px;
  }

  .stat-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }

  .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 700;
  }

  .stat-sub { font-size: 12px; color: var(--text-dim); margin-top: 4px; }

  /* ─── Task Feed ─── */
  .section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px; font-weight: 600; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }

  .task-feed { display: flex; flex-direction: column; gap: 8px; }

  .task-row {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px;
    display: grid; grid-template-columns: 100px 120px 1fr 90px 90px 80px;
    align-items: center; gap: 12px;
    font-size: 13px;
    animation: slideIn 0.3s ease-out;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .task-row.running { border-color: var(--running); background: rgba(251,191,36,0.03); }

  .tag-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; padding: 3px 10px; border-radius: 6px;
    text-transform: uppercase; letter-spacing: 0.5px;
    text-align: center;
  }

  .tag-badge.architect { background: var(--claude-bg); color: var(--claude); }
  .tag-badge.review { background: var(--claude-bg); color: var(--claude); }
  .tag-badge.generate { background: var(--qwen32-bg); color: var(--qwen32); }
  .tag-badge.iterate { background: var(--qwen14-bg); color: var(--qwen14); }

  .model-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 500;
  }

  .model-label.claude { color: var(--claude); }
  .model-label.qwen32 { color: var(--qwen32); }
  .model-label.qwen14 { color: var(--qwen14); }

  .task-prompt { color: var(--text-dim); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .task-tokens { font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: right; }
  .task-duration { font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: right; }

  .task-status { text-align: center; }
  .task-status .done { color: var(--online); }
  .task-status .running-text { color: var(--running); }
  .task-status .error-text { color: var(--error); }

  .empty-state {
    text-align: center; padding: 48px;
    color: var(--text-dim); font-size: 14px;
    background: var(--surface); border: 1px dashed var(--border);
    border-radius: 12px;
  }

  .empty-state code {
    font-family: 'JetBrains Mono', monospace;
    background: var(--surface2); padding: 2px 8px; border-radius: 4px;
    font-size: 12px;
  }

  /* ─── Cost Bar ─── */
  .cost-bar {
    margin-top: 32px; padding: 16px 20px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px;
    display: flex; align-items: center; gap: 24px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
  }

  .cost-segment { display: flex; align-items: center; gap: 8px; }
  .cost-dot { width: 10px; height: 10px; border-radius: 50%; }
</style>
</head>
<body>
<div class="noise"></div>
<div class="container">

  <header>
    <h1><span class="live-dot"></span>Model Orchestrator <span>/ 3 models &middot; 2 machines</span></h1>
    <div class="header-right" id="clock"></div>
  </header>

  <!-- Machine Cards -->
  <div class="machines" id="machines"></div>

  <!-- Stats -->
  <div class="stats-row" id="stats"></div>

  <!-- Task Feed -->
  <div class="section-title">Task Feed</div>
  <div class="task-feed" id="tasks"></div>

  <!-- Cost Bar -->
  <div class="cost-bar" id="costbar"></div>

</div>

<script>
const POLL_INTERVAL = 2000;

function modelClass(label) {
  if (label.includes('Claude')) return 'claude';
  if (label.includes('32B')) return 'qwen32';
  return 'qwen14';
}

function fmtDuration(sec) {
  if (!sec) return '—';
  if (sec < 60) return sec.toFixed(1) + 's';
  return (sec / 60).toFixed(1) + 'm';
}

function renderMachines(machines) {
  const el = document.getElementById('machines');
  const order = ['Anthropic API', 'Mac Studio', 'MacBook Air'];
  const accents = {'Anthropic API': 'claude', 'Mac Studio': 'qwen32', 'MacBook Air': 'qwen14'};
  el.innerHTML = order.map(name => {
    const m = machines[name] || {};
    const st = m.status || 'unknown';
    return `<div class="machine-card" data-accent="${accents[name]}">
      <div class="machine-header">
        <div class="machine-name">${name}</div>
        <div class="status-badge ${st === 'online' ? 'online' : st === 'offline' ? 'offline' : 'unknown'}">${st}</div>
      </div>
      <div class="machine-detail"><span>Model</span><span class="val">${m.model_loaded || '—'}</span></div>
      <div class="machine-detail"><span>VRAM</span><span class="val">${m.memory_used || '—'}</span></div>
    </div>`;
  }).join('');
}

function renderStats(stats) {
  const el = document.getElementById('stats');
  let totalTasks = 0, totalTokens = 0, totalTime = 0, totalCost = 0;
  Object.values(stats).forEach(s => {
    totalTasks += s.tasks;
    totalTokens += s.tokens;
    totalTime += s.total_time;
    totalCost += s.cost_usd;
  });
  el.innerHTML = `
    <div class="stat-card">
      <div class="stat-label">Total Tasks</div>
      <div class="stat-value">${totalTasks}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Tokens</div>
      <div class="stat-value">${totalTokens > 1000 ? (totalTokens/1000).toFixed(1) + 'k' : totalTokens}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Time</div>
      <div class="stat-value">${fmtDuration(totalTime)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">API Cost</div>
      <div class="stat-value">$${totalCost.toFixed(2)}</div>
      <div class="stat-sub">Claude Opus only</div>
    </div>`;
}

function renderTasks(tasks) {
  const el = document.getElementById('tasks');
  if (!tasks.length) {
    el.innerHTML = `<div class="empty-state">No tasks yet. Run a command:<br><br><code>python orchestrator.py --tag generate "Write data_loader.py..."</code></div>`;
    return;
  }
  const sorted = [...tasks].reverse().slice(0, 50);
  el.innerHTML = sorted.map(t => {
    const mc = modelClass(t.model);
    const isRunning = t.status === 'running';
    const statusHtml = t.status === 'done'
      ? '<span class="done">&#10003; done</span>'
      : t.status === 'running'
      ? '<span class="running-text">&#9679; running</span>'
      : '<span class="error-text">&#10007; error</span>';
    return `<div class="task-row ${isRunning ? 'running' : ''}">
      <div><div class="tag-badge ${t.tag}">${t.tag}</div></div>
      <div class="model-label ${mc}">${t.model}</div>
      <div class="task-prompt" title="${(t.prompt_preview||'').replace(/"/g,'&quot;')}">${t.prompt_preview || '—'}</div>
      <div class="task-tokens">${t.tokens ? t.tokens.toLocaleString() + ' tok' : '—'}</div>
      <div class="task-duration">${fmtDuration(t.duration)}</div>
      <div class="task-status">${statusHtml}</div>
    </div>`;
  }).join('');
}

function renderCostBar(stats) {
  const el = document.getElementById('costbar');
  const items = [
    { label: 'Claude Opus 4.6', color: '#D4A26A', tasks: stats['Claude Opus 4.6']?.tasks || 0, tokens: stats['Claude Opus 4.6']?.tokens || 0 },
    { label: 'Qwen 32B', color: '#6ABED4', tasks: stats['Qwen 32B']?.tasks || 0, tokens: stats['Qwen 32B']?.tokens || 0 },
    { label: 'Qwen 14B', color: '#8AD46A', tasks: stats['Qwen 14B']?.tasks || 0, tokens: stats['Qwen 14B']?.tokens || 0 },
  ];
  el.innerHTML = items.map(i =>
    `<div class="cost-segment">
      <div class="cost-dot" style="background:${i.color}"></div>
      <span style="color:${i.color}">${i.label}</span>
      <span style="color:var(--text-dim)">${i.tasks} tasks &middot; ${i.tokens > 1000 ? (i.tokens/1000).toFixed(1)+'k' : i.tokens} tokens</span>
    </div>`
  ).join('');
}

function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}

async function poll() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    renderMachines(data.machines);
    renderStats(data.model_stats);
    renderTasks(data.task_log);
    renderCostBar(data.model_stats);
  } catch (e) {}
  updateClock();
}

poll();
setInterval(poll, POLL_INTERVAL);
setInterval(updateClock, 1000);
</script>
</body>
</html>"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def do_GET(self):
        if self.path == "/api/status":
            with lock:
                payload = {
                    "machines": machine_status,
                    "model_stats": model_stats,
                    "task_log": task_log[-100:],
                    "active_tasks": list(active_tasks.values()),
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())


def start_dashboard(port=8080):
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        httpd.serve_forever()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py --tag architect "Design the merged project..."
  python orchestrator.py --tag generate --save src/model.py "Write model.py..."
  python orchestrator.py --tag iterate --file train.py "Add docstrings..."
  python orchestrator.py --tag review --file src/model.py "Review for bugs..."
  python orchestrator.py --dashboard              # just start the dashboard
        """,
    )
    parser.add_argument("prompt", nargs="?", help="Task prompt")
    parser.add_argument(
        "--tag",
        choices=["architect", "generate", "iterate", "review"],
        help="Route to model: architect/review→Claude, generate→32B, iterate→14B",
    )
    parser.add_argument("--file", help="Include file contents as context")
    parser.add_argument("--save", help="Save response to this file path")
    parser.add_argument(
        "--dashboard", action="store_true", help="Only start the dashboard (no task)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Dashboard port (default 8080)"
    )
    parser.add_argument("--air-ip", help="MacBook Air IP (overrides default)")

    args = parser.parse_args()

    if args.air_ip:
        global MACBOOK_AIR_OLLAMA
        MACBOOK_AIR_OLLAMA = f"http://{args.air_ip}:11434"
        MODELS["iterate"]["url"] = MACBOOK_AIR_OLLAMA

    # Always start dashboard + health in background
    threading.Thread(target=health_loop, daemon=True).start()
    threading.Thread(target=start_dashboard, args=(args.port,), daemon=True).start()

    print(f"\n  Dashboard → http://localhost:{args.port}")
    print(f"  Mac Studio Ollama → {MAC_STUDIO_OLLAMA}")
    print(f"  MacBook Air Ollama → {MACBOOK_AIR_OLLAMA}\n")

    if args.dashboard:
        print("  Dashboard-only mode. Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            return

    if not args.prompt or not args.tag:
        parser.print_help()
        print("\n  Dashboard is running. Open another terminal to send tasks.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    # Load file context if provided
    file_context = None
    if args.file:
        p = Path(args.file)
        if p.exists():
            file_context = p.read_text()
            print(f"  Loaded {p} ({len(file_context)} chars)")
        else:
            print(f"  Warning: {p} not found, proceeding without file context")

    cfg = MODELS[args.tag]
    print(f"  Sending to {cfg['label']} on {cfg['machine']}...")
    print(f"  Tag: {args.tag}")
    print(f"  Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}\n")

    result = dispatch(args.tag, args.prompt, file_context)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(result)
        print(f"\n  Saved to {args.save}")
    else:
        print(result)

    # Keep dashboard alive briefly so it updates
    time.sleep(2)


if __name__ == "__main__":
    main()

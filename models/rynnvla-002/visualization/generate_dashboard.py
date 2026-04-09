#!/usr/bin/env python3
"""
Generate a self-contained HTML training dashboard from a run directory.

Usage:
    python visualization/generate_dashboard.py \\
        --run_dir /path/to/training/run \\
        --output /path/to/training_output/run_name/dashboard.html

    If --output is omitted, writes dashboard.html next to the run directory.
"""

import argparse
import json
import os
import re
from pathlib import Path

import yaml


# ── Log parsing ───────────────────────────────────────────────────────────────

def _get_val(line: str, key: str) -> float | None:
    """Extract the first numeric value after 'key:' in a log line."""
    m = re.search(rf'\b{re.escape(key)}:\s+([\d.]+)', line)
    return float(m.group(1)) if m else None


def parse_log(run_dir: Path):
    """
    Parse rank-0.log (or common.log) from a training run directory.

    Returns:
        steps      – dict of parallel column arrays, one entry per logged step
        events     – list of timeline event dicts (epoch ends, checkpoints)
        raw_lines  – non-step log lines for the Raw Logs tab
    """
    log_path = run_dir / "rank-0.log"
    if not log_path.exists():
        log_path = run_dir / "common.log"
    if not log_path.exists():
        raise FileNotFoundError(f"No rank-0.log or common.log found in {run_dir}")

    steps = {k: [] for k in [
        "global_step", "epoch", "lr", "grad_norm",
        "closs", "loss_ct", "z_loss",
        "dataload_s", "update_s", "samples_sec", "max_mem_mb", "ts",
    ]}
    events = []
    raw_lines = []

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()

            # ── Per-step training line ────────────────────────────────────────
            if "misc.py:146" in line and ">> Epoch:" in line:
                try:
                    ts = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line).group(1)
                    epoch = int(re.search(r"Epoch: \[(\d+)\]", line).group(1))
                    sm = re.search(r"\[(\d+)/(\d+)\]", line)
                    step, total = int(sm.group(1)), int(sm.group(2))
                    t = _get_val(line, "time")
                    d = _get_val(line, "data")
                    steps["global_step"].append(epoch * total + step)
                    steps["epoch"].append(epoch)
                    steps["lr"].append(_get_val(line, "lr"))
                    steps["grad_norm"].append(_get_val(line, "grad_norm"))
                    steps["closs"].append(_get_val(line, "closs"))
                    steps["loss_ct"].append(_get_val(line, "loss_ct"))
                    steps["z_loss"].append(_get_val(line, "z_loss"))
                    steps["dataload_s"].append(d)
                    steps["update_s"].append(round(t - d, 4) if (t is not None and d is not None) else None)
                    steps["samples_sec"].append(_get_val(line, "samples/sec"))
                    steps["max_mem_mb"].append(_get_val(line, "max mem"))
                    steps["ts"].append(ts)
                except Exception:
                    pass
                # Skip step lines from raw log — they're redundant with the charts
                continue

            raw_lines.append(line)

            # ── Epoch end ─────────────────────────────────────────────────────
            if "misc.py:151" in line and "Total time" in line:
                try:
                    ts = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line).group(1)
                    m = re.search(r"Epoch: \[(\d+)\] Total time: (.+)", line)
                    events.append({
                        "type": "epoch_end",
                        "ts": ts,
                        "epoch": int(m.group(1)),
                        "label": f"Epoch {m.group(1)} complete — {m.group(2).strip()}",
                    })
                except Exception:
                    pass

            # ── Checkpoint saved / deleted ────────────────────────────────────
            elif "ckpt.py" in line and (">> Saved" in line or ">> Deleted" in line):
                try:
                    ts = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line).group(1)
                    m = re.search(r">> (Saved|Deleted) (.+)", line)
                    name = os.path.basename(m.group(2).strip())
                    events.append({
                        "type": f"ckpt_{m.group(1).lower()}",
                        "ts": ts,
                        "label": f"{m.group(1)}: {name}",
                    })
                except Exception:
                    pass

    return steps, events, raw_lines


# ── HTML template ─────────────────────────────────────────────────────────────

# Markers in the template that get replaced with real data/values:
#   __RUN_NAME__, __TRAINING_START__, __TRAINING_END__, __TOTAL_STEPS__
#   __DATA_JSON__, __RAW_LINES_JSON__

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Dashboard — __RUN_NAME__</title>
<!-- Plotly.js via CDN — requires internet to render charts -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background:
      linear-gradient(180deg, #f5f5f7 0%, #eef1f5 100%);
    color: #111827;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    margin: 18px 18px 0;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: #ffffff;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    border-radius: 24px;
  }
  header h1 { font-size: 13px; font-weight: 600; color: #6b7280; letter-spacing: 0.08em; text-transform: uppercase; }
  .run-name { font-size: 28px; font-weight: 700; color: #0f172a; letter-spacing: -0.03em; }
  .run-meta { margin-left: auto; display: flex; gap: 20px; flex-wrap: wrap; }
  .run-meta span {
    font-size: 12px;
    color: #6b7280;
    background: #f8fafc;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 999px;
    padding: 8px 12px;
  }
  .run-meta strong { color: #0f172a; }

  /* ── Tabs ── */
  nav {
    margin: 14px 18px 0;
    padding: 8px;
    display: flex;
    gap: 6px;
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: #f0f3f7;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
    border-radius: 20px;
  }
  nav button {
    background: transparent;
    border: none;
    border-radius: 14px;
    color: #6b7280;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 16px;
    transition: color 0.18s, background 0.18s, box-shadow 0.18s;
  }
  nav button:hover { color: #0f172a; background: rgba(255, 255, 255, 0.72); }
  nav button.active {
    color: #0f172a;
    background: #ffffff;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
  }

  /* ── Tab content ── */
  .tab { display: none; padding: 18px; }
  .tab.active { display: block; }

  /* ── Chart grid ── */
  .chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }
  .chart-card {
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.06);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    border-radius: 26px;
  }
  .chart-card.full { grid-column: 1 / -1; }
  .chart-card.primary {
    background:
      linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  }
  .chart-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    padding: 18px 20px 0;
  }
  .chart-title-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  .chart-title {
    font-size: 18px;
    font-weight: 650;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }
  .chart-note {
    color: #475569;
    font-size: 13px;
    line-height: 1.45;
    padding: 8px 20px 0;
    max-width: 72ch;
  }
  .chart-summary {
    display: inline-flex;
    align-items: baseline;
    gap: 10px;
    margin: 14px 20px 0;
    padding: 10px 14px;
    background: #f8fafc;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 18px;
  }
  .summary-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .summary-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.04em;
  }
  .summary-delta {
    font-size: 13px;
    font-weight: 600;
    color: #2563eb;
  }
  .info-wrap { position: relative; flex-shrink: 0; }
  .info-btn {
    width: 24px;
    height: 24px;
    border: none;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.16);
    border: 1px solid rgba(148, 163, 184, 0.12);
    color: #334155;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
  }
  .info-tip {
    position: absolute;
    top: 32px;
    right: 0;
    width: min(320px, calc(100vw - 48px));
    padding: 12px 14px;
    background: rgba(15, 23, 42, 0.92);
    color: #f8fafc;
    border-radius: 16px;
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.28);
    font-size: 12px;
    line-height: 1.5;
    opacity: 0;
    pointer-events: none;
    transform: translateY(-4px);
    transition: opacity 0.16s ease, transform 0.16s ease;
    z-index: 9999;
    white-space: normal;
  }
  .info-tip strong { color: #ffffff; }
  .info-wrap:hover .info-tip,
  .info-wrap:focus-within .info-tip {
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
  }
  .chart-plot      { width: 100%; height: 236px; }
  .chart-plot.tall { width: 100%; height: 320px; }

  /* ── Timeline ── */
  #timeline-chart {
    width: 100%;
    height: calc(100vh - 220px);
    min-height: 420px;
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.06);
    border-radius: 26px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
  }

  /* ── Raw logs ── */
  .log-toolbar { display: flex; gap: 10px; margin-bottom: 12px; align-items: center; }
  #log-search {
    background: #ffffff;
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 14px;
    color: #0f172a;
    font-size: 13px;
    padding: 10px 14px;
    width: 320px;
  }
  #log-search:focus { border-color: rgba(59, 130, 246, 0.5); outline: none; }
  #log-count { font-size: 12px; color: #64748b; }
  #log-viewer {
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.06);
    border-radius: 26px;
    padding: 14px 16px;
    height: calc(100vh - 230px);
    overflow-y: auto;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
  }
  pre#log-content {
    font-family: 'Cascadia Code', 'Fira Code', ui-monospace, monospace;
    font-size: 12px;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .ll-warn  { color: #b45309; }
  .ll-error { color: #dc2626; }
  .ll-epoch { color: #2563eb; font-weight: 600; }
  .ll-ckpt  { color: #059669; }
  .ll-info  { color: #475569; }
  mark { background: rgba(96, 165, 250, 0.22); color: #0f172a; border-radius: 4px; }

  @media (max-width: 900px) {
    .chart-grid { grid-template-columns: 1fr; }
    .run-meta { margin-left: 0; }
    .chart-card.full { grid-column: auto; }
    .summary-value { font-size: 24px; }
  }

  @media (max-width: 640px) {
    .chart-header {
      align-items: center;
    }
    .chart-title {
      font-size: 17px;
    }
    .info-tip {
      position: fixed;
      top: auto;
      left: 24px;
      right: 24px;
      bottom: 24px;
      width: auto;
      max-width: none;
      transform: translateY(8px);
    }
    .info-wrap:hover .info-tip,
    .info-wrap:focus-within .info-tip {
      transform: translateY(0);
    }
  }
</style>
</head>
<body>

<header>
  <h1>Training Dashboard</h1>
  <span class="run-name">__RUN_NAME__</span>
  <div class="run-meta">
    <span>Start <strong>__TRAINING_START__</strong></span>
    <span>End <strong>__TRAINING_END__</strong></span>
    <span>Total steps <strong>__TOTAL_STEPS__</strong></span>
  </div>
</header>

<nav>
  <button class="active" onclick="showTab('training', this)">Training Metrics</button>
  <button onclick="showTab('timeline', this)">Timeline</button>
  <button onclick="showTab('logs', this)">Raw Logs</button>
</nav>

<div id="training" class="tab active">
  <div class="chart-grid">

    <div class="chart-card full primary">
      <div class="chart-header">
        <div>
          <div class="chart-title-wrap">
            <div class="chart-title">Loss</div>
            <div class="info-wrap">
              <button class="info-btn" type="button" aria-label="Loss info">i</button>
              <div class="info-tip">
                <strong>closs</strong> — the main training loss. Measures how accurately the model predicts actions. Lower is better; a steady downward trend means training is working.<br><br>
                <strong>loss_ct</strong> — contrastive loss. Pushes the model to learn consistent representations across similar situations. Should also decrease over time alongside closs.<br><br>
                <strong>z_loss</strong> — auxiliary regularization loss (plotted on the right-hand axis — different scale). Keeps internal activations from growing unbounded. Usually small; large spikes can signal instability.<br><br>
                If any of these stall or spike, check Gradient Norm and Learning Rate next.
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="chart-summary">
        <span class="summary-label">Loss</span>
        <span class="summary-value" id="loss-latest">--</span>
        <span class="summary-delta" id="loss-delta"></span>
      </div>
      <div class="chart-plot tall" id="chart-loss"></div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <div>
          <div class="chart-title-wrap">
            <div class="chart-title">Gradient Norm</div>
            <div class="info-wrap">
              <button class="info-btn" type="button" aria-label="Gradient norm info">i</button>
              <div class="info-tip"><strong>What it is:</strong> the size of your parameter updates. <strong>What to look for:</strong> sudden spikes suggest exploding gradients; values collapsing toward zero suggest vanishing gradients. Either can quietly break training.</div>
            </div>
          </div>
        </div>
      </div>
      <div class="chart-plot" id="chart-gradnorm"></div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <div>
          <div class="chart-title-wrap">
            <div class="chart-title">Learning Rate</div>
            <div class="info-wrap">
              <button class="info-btn" type="button" aria-label="Learning rate info">i</button>
              <div class="info-tip"><strong>What it is:</strong> the step size chosen by the optimizer. <strong>What to look for:</strong> use it to explain the other charts. A flat loss means very different things under a tiny LR versus an aggressive schedule.</div>
            </div>
          </div>
        </div>
      </div>
      <div class="chart-plot" id="chart-lr"></div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <div>
          <div class="chart-title-wrap">
            <div class="chart-title">Throughput</div>
            <div class="info-wrap">
              <button class="info-btn" type="button" aria-label="Throughput info">i</button>
              <div class="info-tip"><strong>What it is:</strong> samples processed per second. <strong>What to look for:</strong> higher and stable is better. If it is low, you are burning GPU time no matter how good the loss looks.</div>
            </div>
          </div>
        </div>
      </div>
      <div class="chart-plot" id="chart-samples"></div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <div>
          <div class="chart-title-wrap">
            <div class="chart-title">Update Time</div>
            <div class="info-wrap">
              <button class="info-btn" type="button" aria-label="Update time info">i</button>
              <div class="info-tip"><strong>What it is:</strong> time spent doing the actual training update each step. <strong>What to look for:</strong> compare it with dataload time to separate compute bottlenecks from input pipeline problems.</div>
            </div>
          </div>
        </div>
      </div>
      <div class="chart-plot" id="chart-update"></div>
    </div>

    <div class="chart-card full">
      <div class="chart-header">
        <div>
          <div class="chart-title-wrap">
            <div class="chart-title">Dataload Time</div>
            <div class="info-wrap">
              <button class="info-btn" type="button" aria-label="Dataload time info">i</button>
              <div class="info-tip"><strong>What it is:</strong> time spent fetching and preparing each batch. <strong>What to look for:</strong> it is usually boring, until it is not. Spikes or drift here mean the data pipeline is now the urgent problem.</div>
            </div>
          </div>
        </div>
      </div>
      <div class="chart-plot" id="chart-dataload"></div>
    </div>

  </div>
</div>

<div id="timeline" class="tab">
  <div id="timeline-chart"></div>
</div>

<div id="logs" class="tab">
  <div class="log-toolbar">
    <input id="log-search" type="text" placeholder="Filter logs…" oninput="filterLogs(this.value)">
    <span id="log-count"></span>
  </div>
  <div id="log-viewer"><pre id="log-content"></pre></div>
</div>

<script>
const DATA = __DATA_JSON__;
const RAW_LINES = __RAW_LINES_JSON__;

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(id, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  if (id === 'timeline' && !window._timelineReady) renderTimeline();
}

// ── Shared Plotly config ──────────────────────────────────────────────────────
const BASE_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(255,255,255,0)',
  font: { color: '#0f172a', family: "-apple-system, BlinkMacSystemFont, sans-serif", size: 11 },
  xaxis: {
    gridcolor: 'rgba(148, 163, 184, 0.18)', zerolinecolor: 'rgba(148, 163, 184, 0.22)',
    title: { text: 'Step', font: { size: 11 } },
    tickfont: { size: 10, color: '#64748b' },
  },
  yaxis: {
    gridcolor: 'rgba(148, 163, 184, 0.18)', zerolinecolor: 'rgba(148, 163, 184, 0.22)',
    tickfont: { size: 10, color: '#64748b' },
  },
  margin: { t: 10, b: 54, l: 58, r: 20 },
  hovermode: 'x unified',
  hoverlabel: { bgcolor: '#ffffff', bordercolor: 'rgba(148, 163, 184, 0.32)', font: { size: 11, color: '#0f172a' } },
};

const CFG = { responsive: true, displayModeBar: false };

const s  = DATA.steps;
const xs = s.global_step;

// Vertical dashed lines at epoch boundaries
function epochShapes() {
  const shapes = [];
  let prev = s.epoch[0];
  for (let i = 1; i < s.epoch.length; i++) {
    if (s.epoch[i] !== prev) {
      shapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: xs[i], x1: xs[i], y0: 0, y1: 1,
        line: { color: 'rgba(148, 163, 184, 0.26)', width: 1, dash: 'dot' },
      });
      prev = s.epoch[i];
    }
  }
  return shapes;
}
const EPOCH_SHAPES = epochShapes();

function mkLayout(overrides) {
  return Object.assign({}, BASE_LAYOUT, { shapes: EPOCH_SHAPES }, overrides || {});
}

function line(y, color, name) {
  return { x: xs, y, name: name || '', type: 'scatter', mode: 'lines',
           line: { color, width: 2.2, shape: 'spline', smoothing: 0.55 },
           hovertemplate: '%{y:.4f}<extra>' + (name||'') + '</extra>' };
}

function finiteValues(values) {
  return (values || []).filter(v => Number.isFinite(v));
}

function latestValue(values) {
  const vals = finiteValues(values);
  return vals.length ? vals[vals.length - 1] : null;
}

function firstValue(values) {
  const vals = finiteValues(values);
  return vals.length ? vals[0] : null;
}

function updateLossSummary() {
  const latest = latestValue(s.closs);
  const start = firstValue(s.closs);
  document.getElementById('loss-latest').textContent =
    latest == null ? '--' : latest.toFixed(4);

  const deltaNode = document.getElementById('loss-delta');
  if (latest == null || start == null) {
    deltaNode.textContent = '';
    return;
  }

  const delta = latest - start;
  const improved = delta < 0;
  const arrow = improved ? '↓' : '↑';
  deltaNode.textContent = arrow + ' ' + Math.abs(delta).toFixed(4) + ' since start (' + (improved ? 'improving' : 'worsening') + ')';
  deltaNode.style.color = improved ? '#059669' : '#dc2626';
}

// ── Training charts ───────────────────────────────────────────────────────────
function renderTraining() {
  // Loss — closs on left y, loss_ct on left y, z_loss on right y (very different scale)
  Plotly.newPlot('chart-loss', [
    line(s.closs,    '#2563eb', 'closs'),
    line(s.loss_ct,  '#0f766e', 'loss_ct'),
    Object.assign(line(s.z_loss, '#f59e0b', 'z_loss'), { yaxis: 'y2' }),
  ], mkLayout({
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: 1.14, bgcolor: 'transparent', font: { size: 11, color: '#475569' } },
    yaxis2: {
      overlaying: 'y', side: 'right',
      gridcolor: 'rgba(148, 163, 184, 0.18)', tickfont: { size: 10, color: '#64748b' }, zerolinecolor: 'rgba(148, 163, 184, 0.22)',
      title: { text: 'z_loss', font: { size: 10, color: '#b45309' } },
    },
    margin: { t: 44, b: 54, l: 58, r: 60 },
  }), CFG);

  Plotly.newPlot('chart-gradnorm',
    [line(s.grad_norm,   '#ef4444', 'grad_norm')],
    mkLayout(), CFG);

  Plotly.newPlot('chart-lr',
    [line(s.lr,          '#7c3aed', 'lr')],
    mkLayout({ yaxis: { ...BASE_LAYOUT.yaxis, tickformat: '.2e' } }), CFG);

  Plotly.newPlot('chart-samples',
    [line(s.samples_sec, '#0f766e', 'samples/sec')],
    mkLayout(), CFG);

  Plotly.newPlot('chart-update',
    [line(s.update_s,    '#2563eb', 'update_s')],
    mkLayout(), CFG);

  Plotly.newPlot('chart-dataload',
    [line(s.dataload_s,  '#f59e0b', 'dataload_s')],
    mkLayout(), CFG);

  updateLossSummary();
}

// ── Timeline ──────────────────────────────────────────────────────────────────
function renderTimeline() {
  window._timelineReady = true;
  const events = DATA.events;

  // Progress line: training step vs wall time
  const progressTrace = {
    x: s.ts, y: xs,
    type: 'scatter', mode: 'lines',
    name: 'step progress',
    line: { color: '#58a6ff', width: 2 },
    hovertemplate: 'step %{y}<br>%{x}<extra></extra>',
  };

  // Event markers
  const typeStyle = {
    epoch_end:    { color: '#059669', symbol: 'star',       size: 14 },
    ckpt_deleted: { color: '#94a3b8', symbol: 'x',          size: 8  },
    ckpt_saved:   { color: '#f59e0b', symbol: 'triangle-up',size: 10 },
  };

  const groups = {};
  (events || []).forEach(e => {
    if (!groups[e.type]) groups[e.type] = { x: [], y: [], text: [] };
    groups[e.type].x.push(e.ts);
    // Place event markers near the step count at that time (estimate from nearest step ts)
    const idx = s.ts.findLastIndex(t => t <= e.ts) || 0;
    groups[e.type].y.push(xs[Math.max(0, idx)]);
    groups[e.type].text.push(e.label);
  });

  const eventTraces = Object.entries(groups).map(([type, g]) => {
    const st = typeStyle[type] || { color: '#8b949e', symbol: 'circle', size: 8 };
    return {
      x: g.x, y: g.y, text: g.text,
      type: 'scatter', mode: 'markers',
      name: type.replace(/_/g, ' '),
      marker: { color: st.color, symbol: st.symbol, size: st.size,
                line: { color: '#ffffff', width: 1 } },
      hovertemplate: '%{text}<br>%{x}<extra></extra>',
    };
  });

  Plotly.newPlot('timeline-chart',
    [progressTrace, ...eventTraces],
    Object.assign({}, BASE_LAYOUT, {
      showlegend: true,
      legend: { bgcolor: 'transparent', font: { size: 12 } },
      xaxis: { ...BASE_LAYOUT.xaxis, type: 'date',
               title: { text: 'Wall Time', font: { size: 12 } } },
      yaxis: { ...BASE_LAYOUT.yaxis,
               title: { text: 'Global Step', font: { size: 12 } } },
      margin: { t: 20, b: 56, l: 70, r: 20 },
      height: Math.max(400, window.innerHeight - 180),
    }), CFG);
}

// ── Raw logs ──────────────────────────────────────────────────────────────────
function classifyLine(l) {
  if (l.includes('WARNING'))              return 'll-warn';
  if (l.includes('ERROR'))               return 'll-error';
  if (l.includes('Total time'))          return 'll-epoch';
  if (l.includes('Saved') || l.includes('Deleted')) return 'll-ckpt';
  return 'll-info';
}

function renderLogs(lines, hl) {
  const re = hl ? new RegExp(hl.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'), 'gi') : null;
  const chunks = lines.map(l => {
    const cls = classifyLine(l);
    const esc = l.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const body = re ? esc.replace(re, m => '<mark>' + m + '</mark>') : esc;
    return '<span class="' + cls + '">' + body + '\\n</span>';
  });
  document.getElementById('log-content').innerHTML = chunks.join('');
  document.getElementById('log-count').textContent =
    lines.length + ' / ' + RAW_LINES.length + ' lines';
}

function filterLogs(q) {
  const lines = q.trim()
    ? RAW_LINES.filter(l => l.toLowerCase().includes(q.toLowerCase()))
    : RAW_LINES;
  renderLogs(lines, q.trim() || null);
}

// ── Init ──────────────────────────────────────────────────────────────────────
renderTraining();
renderLogs(RAW_LINES, null);
</script>
</body>
</html>
"""


# ── Dashboard generation ──────────────────────────────────────────────────────

def generate_dashboard(run_dir: str | Path, output_path: str | Path) -> None:
    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path)
    run_name = run_dir.name

    print(f"Parsing {run_dir} ...")
    steps, events, raw_lines = parse_log(run_dir)
    n_steps = len(steps["global_step"])
    print(f"  {n_steps} step entries, {len(events)} timeline events, {len(raw_lines)} raw log lines")

    training_start = steps["ts"][0]  if steps["ts"]          else "unknown"
    training_end   = steps["ts"][-1] if steps["ts"]          else "unknown"
    total_steps    = steps["global_step"][-1] if steps["global_step"] else 0

    # Cap raw lines to keep HTML size reasonable
    if len(raw_lines) > 3000:
        head, tail = raw_lines[:400], raw_lines[-2000:]
        omitted = len(raw_lines) - len(head) - len(tail)
        raw_lines = head + [f"--- {omitted} lines omitted ---"] + tail

    data = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "training_start": training_start,
        "training_end": training_end,
        "steps": steps,
        "events": events,
    }

    html = HTML_TEMPLATE
    html = html.replace("__RUN_NAME__",       run_name)
    html = html.replace("__TRAINING_START__", training_start)
    html = html.replace("__TRAINING_END__",   training_end)
    html = html.replace("__TOTAL_STEPS__",    str(total_steps))
    html = html.replace("__DATA_JSON__",      json.dumps(data))
    html = html.replace("__RAW_LINES_JSON__", json.dumps(raw_lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size // 1024
    print(f"Dashboard saved → {output_path}  ({size_kb} KB)")


def load_config() -> dict:
    """Read config.yaml from the model root (two levels up from this script)."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main():
    cfg = load_config()

    # Derive defaults from config.yaml
    work_dir        = cfg.get("work_dir", "my_data/training_pipeline")
    task_label      = cfg.get("task_label", "")
    robot           = cfg.get("robot", "")
    training_output = cfg.get("training_output", "training_output")

    run_name_default = f"{task_label}_{robot}" if (task_label and robot) else None
    default_run_dir  = f"{work_dir}/fine_tuning/{run_name_default}" if run_name_default else None
    default_output   = f"{training_output}/{run_name_default}/dashboard.html" if run_name_default else None

    parser = argparse.ArgumentParser(description="Generate a training dashboard HTML")
    parser.add_argument(
        "--run_dir",
        default=default_run_dir,
        help=f"Path to training run directory (default from config: {default_run_dir})",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output .html path (default from config: {default_output})",
    )
    args = parser.parse_args()

    if not args.run_dir:
        parser.error("--run_dir is required (or set task_label + robot in config.yaml)")

    generate_dashboard(args.run_dir, args.output)


if __name__ == "__main__":
    main()

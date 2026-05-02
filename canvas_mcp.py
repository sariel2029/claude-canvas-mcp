"""
canvas_mcp.py
-------------
A minimal MCP server that gives Claude a canvas to draw on.

Architecture:
  - FastMCP exposes two tools to Claude (canvas_draw, canvas_clear).
  - A FastAPI HTTP server (in a background thread) serves an HTML page
    that polls for strokes and renders them with SVG dasharray animation,
    so each stroke visibly "falls" onto the page.
  - Open http://127.0.0.1:1573 in a browser to watch.

Install:
  pip install fastmcp fastapi uvicorn

Run as MCP stdio server:
  python canvas_mcp.py
"""

from __future__ import annotations

import io
import re
import threading
import time
import uuid
from collections import deque
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.types import Image as MCPImage
from PIL import Image as PILImage
from PIL import ImageDraw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PORT = 1573
MAX_STROKES = 2000
MAX_PATH_LENGTH = 8000
MAX_POINTS = 1000
PATH_RE = re.compile(r"^[MmZzLlHhVvCcSsQqTtAa0-9eE.,+\-\s]+$")
TOKEN_RE = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_strokes: deque[dict[str, Any]] = deque(maxlen=MAX_STROKES)
_lock = threading.Lock()


def _all_strokes() -> list[dict[str, Any]]:
    with _lock:
        return list(_strokes)


def _clear_strokes() -> None:
    with _lock:
        _strokes.clear()


def _push_stroke_and_count(s: dict[str, Any]) -> int:
    with _lock:
        _strokes.append(s)
        return len(_strokes)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _validate_path(path: str) -> str:
    path = path.strip()
    if not path:
        raise ValueError("path must not be empty")
    if len(path) > MAX_PATH_LENGTH:
        raise ValueError(f"path is too long; max {MAX_PATH_LENGTH} characters")
    if not PATH_RE.fullmatch(path):
        raise ValueError("path contains unsupported SVG path characters")
    return path


def _validate_points(points: list[list[float]]) -> list[list[float]]:
    if not points:
        raise ValueError("points must not be empty")
    if len(points) > MAX_POINTS:
        raise ValueError(f"too many points; max {MAX_POINTS}")

    out: list[list[float]] = []
    for i, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"point {i} must contain exactly [x, y]")
        x = _clamp(float(point[0]), 0.0, 1000.0)
        y = _clamp(float(point[1]), 0.0, 1000.0)
        out.append([x, y])
    return out


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _quad_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    mt = 1.0 - t
    x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
    y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
    return (x, y)


def _cubic_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    mt = 1.0 - t
    x = (
        mt * mt * mt * p0[0]
        + 3 * mt * mt * t * p1[0]
        + 3 * mt * t * t * p2[0]
        + t * t * t * p3[0]
    )
    y = (
        mt * mt * mt * p0[1]
        + 3 * mt * mt * t * p1[1]
        + 3 * mt * t * t * p2[1]
        + t * t * t * p3[1]
    )
    return (x, y)


def _sample_pen_points(points: list[list[float]]) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) <= 2:
        return pts

    sampled = [pts[0]]
    for i in range(1, len(pts) - 1):
        start = sampled[-1]
        control = pts[i]
        end = _midpoint(pts[i], pts[i + 1])
        for step in range(1, 9):
            sampled.append(_quad_point(start, control, end, step / 8.0))
    sampled.append(pts[-1])
    return sampled


def _sample_svg_path(path: str) -> list[list[tuple[float, float]]]:
    tokens = TOKEN_RE.findall(path)
    subpaths: list[list[tuple[float, float]]] = []
    current_path: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    command = ""
    i = 0

    def add_point(pt: tuple[float, float]) -> None:
        nonlocal current_path
        if not current_path:
            current_path = [pt]
        elif current_path[-1] != pt:
            current_path.append(pt)

    while i < len(tokens):
        token = tokens[i]
        if re.fullmatch(r"[A-Za-z]", token):
            command = token
            i += 1
            if command in "Zz":
                add_point(start)
                if current_path:
                    subpaths.append(current_path)
                    current_path = []
                current = start
            continue

        if not command:
            raise ValueError("path is missing an initial command")

        absolute = command.isupper()
        op = command.upper()

        def read_float() -> float:
            nonlocal i
            value = float(tokens[i])
            i += 1
            return value

        if op == "M":
            x = read_float()
            y = read_float()
            current = (x, y) if absolute else (current[0] + x, current[1] + y)
            start = current
            if current_path:
                subpaths.append(current_path)
            current_path = [current]
            command = "L" if absolute else "l"
        elif op == "L":
            x = read_float()
            y = read_float()
            target = (x, y) if absolute else (current[0] + x, current[1] + y)
            add_point(target)
            current = target
        elif op == "H":
            x = read_float()
            target = (x, current[1]) if absolute else (current[0] + x, current[1])
            add_point(target)
            current = target
        elif op == "V":
            y = read_float()
            target = (current[0], y) if absolute else (current[0], current[1] + y)
            add_point(target)
            current = target
        elif op == "Q":
            x1 = read_float()
            y1 = read_float()
            x = read_float()
            y = read_float()
            control = (x1, y1) if absolute else (current[0] + x1, current[1] + y1)
            target = (x, y) if absolute else (current[0] + x, current[1] + y)
            for step in range(1, 17):
                add_point(_quad_point(current, control, target, step / 16.0))
            current = target
        elif op == "C":
            x1 = read_float()
            y1 = read_float()
            x2 = read_float()
            y2 = read_float()
            x = read_float()
            y = read_float()
            c1 = (x1, y1) if absolute else (current[0] + x1, current[1] + y1)
            c2 = (x2, y2) if absolute else (current[0] + x2, current[1] + y2)
            target = (x, y) if absolute else (current[0] + x, current[1] + y)
            for step in range(1, 25):
                add_point(_cubic_point(current, c1, c2, target, step / 24.0))
            current = target
        else:
            raise ValueError(f"unsupported SVG path command for snapshot: {command}")

    if current_path:
        subpaths.append(current_path)
    return subpaths


def _render_snapshot_png(size: int = 1000) -> bytes:
    image = PILImage.new("RGBA", (size, size), "#fffefa")
    draw = ImageDraw.Draw(image, "RGBA")

    for stroke in _all_strokes():
        width = max(1, int(round(float(stroke.get("width", 3)))))
        opacity = int(round(_clamp(float(stroke.get("opacity", 1.0)), 0.0, 1.0) * 255))
        color = stroke.get("stroke", "#1a1a1a")
        if isinstance(color, str) and color.startswith("#") and len(color) == 7:
            rgba = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) + (opacity,)
        else:
            rgba = (26, 26, 26, opacity)

        try:
            if stroke.get("type") == "pen":
                pts = _sample_pen_points(stroke.get("points", []))
                if len(pts) == 1:
                    x, y = pts[0]
                    draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=rgba)
                elif len(pts) > 1:
                    draw.line(pts, fill=rgba, width=width, joint="curve")
            else:
                for subpath in _sample_svg_path(stroke.get("path", "")):
                    if len(subpath) == 1:
                        x, y = subpath[0]
                        draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=rgba)
                    elif len(subpath) > 1:
                        draw.line(subpath, fill=rgba, width=width, joint="curve")
        except Exception:
            continue

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML canvas page
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Claude's Canvas</title>
<style>
  :root {
    --bg: #faf8f3;
    --paper: #fffefa;
    --line: #e0dcd1;
    --meta: #999;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 24px;
  }
  .wrap { display: flex; flex-direction: column; align-items: center; gap: 14px; }
  canvas.canvas {
    background: var(--paper);
    border: 1px solid var(--line);
    box-shadow: 0 2px 14px rgba(0,0,0,0.04);
    width: 90vmin;
    height: 90vmin;
    border-radius: 4px;
    cursor: crosshair;
  }
  .meta {
    color: var(--meta);
    font-size: 12px;
    letter-spacing: 0.04em;
    user-select: none;
  }
  .meta button {
    background: none; border: none; color: var(--meta);
    font: inherit; cursor: pointer; padding: 0 4px;
    border-bottom: 1px dashed var(--meta);
  }
  .meta button:hover { color: #555; border-color: #555; }
</style>
</head>
<body>
  <div class="wrap">
    <canvas class="canvas" id="canvas" width="1000" height="1000"></canvas>
    <div class="meta">
      <span id="count">0</span> strokes &middot; port 1573 &middot;
      <button onclick="doClear()">clear</button>
    </div>
  </div>

<script>
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const counter = document.getElementById('count');
const seen    = new Set();
let firstLoad = true;

ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function applyStyle(s) {
  ctx.strokeStyle = s.stroke || '#1a1a1a';
  ctx.lineWidth = s.width != null ? s.width : 3;
  ctx.globalAlpha = s.opacity != null ? s.opacity : 1;
}

function drawPathStroke(s) {
  try {
    applyStyle(s);
    const path = new Path2D(s.path);
    ctx.stroke(path);
  } catch (e) {
    // Bad browser-side path; server validation should usually catch this.
  } finally {
    ctx.globalAlpha = 1;
  }
}

function drawPenSegments(s, upto) {
  const points = s.points || [];
  if (points.length === 0) return;
  const count = Math.min(points.length, upto);
  applyStyle(s);
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);

  if (count === 1) {
    ctx.lineTo(points[0][0] + 0.01, points[0][1] + 0.01);
  } else if (count === 2) {
    ctx.lineTo(points[1][0], points[1][1]);
  } else {
    for (let i = 1; i < count - 1; i++) {
      const cx = points[i][0];
      const cy = points[i][1];
      const mx = (points[i][0] + points[i + 1][0]) / 2;
      const my = (points[i][1] + points[i + 1][1]) / 2;
      ctx.quadraticCurveTo(cx, cy, mx, my);
    }
    const last = points[count - 1];
    ctx.lineTo(last[0], last[1]);
  }

  ctx.stroke();
  ctx.globalAlpha = 1;
}

function animatePenStroke(s) {
  const points = s.points || [];
  if (points.length < 2) {
    drawPenSegments(s, points.length);
    return;
  }
  const duration = s.duration != null ? s.duration : 400;
  const started = performance.now();

  function tick(now) {
    const progress = duration <= 0 ? 1 : Math.min(1, (now - started) / duration);
    const upto = Math.max(2, Math.ceil(points.length * progress));
    clearCanvas();
    for (const old of rendered) {
      if (old.type === 'pen') drawPenSegments(old, old.points.length);
      else drawPathStroke(old);
    }
    drawPenSegments(s, upto);
    if (progress < 1) requestAnimationFrame(tick);
    else rendered.push(s);
  }

  requestAnimationFrame(tick);
}

const rendered = [];

function addStroke(s, animate) {
  if (seen.has(s.id)) return;
  seen.add(s.id);

  if (s.type === 'pen') {
    if (animate) animatePenStroke(s);
    else {
      drawPenSegments(s, s.points.length);
      rendered.push(s);
    }
  } else {
    drawPathStroke(s);
    rendered.push(s);
  }
}

async function poll() {
  try {
    const r = await fetch('/strokes', { cache: 'no-store' });
    const list = await r.json();
    if (list.length === 0 && seen.size > 0) {
      clearCanvas();
      seen.clear();
      rendered.length = 0;
      firstLoad = true;
    }
    for (const s of list) addStroke(s, !firstLoad);
    firstLoad = false;
    counter.textContent = seen.size;
  } catch (e) {
    // server might be restarting; just retry
  }
  setTimeout(poll, 200);
}

async function doClear() {
  await fetch('/clear', { method: 'POST' });
  // Clear locally too — the next poll will resync.
  clearCanvas();
  seen.clear();
  rendered.length = 0;
  counter.textContent = 0;
  firstLoad = true;
}

poll();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP side (FastAPI)
# ---------------------------------------------------------------------------

app = FastAPI(title="Claude Canvas")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/strokes")
def strokes_endpoint() -> list[dict[str, Any]]:
    return _all_strokes()


@app.post("/clear")
def clear_endpoint() -> dict[str, Any]:
    _clear_strokes()
    return {"ok": True}


def _run_http() -> None:
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


# ---------------------------------------------------------------------------
# MCP side (FastMCP)
# ---------------------------------------------------------------------------

mcp = FastMCP("canvas")


@mcp.tool
def canvas_draw(
    path: str,
    stroke: str = "#1a1a1a",
    width: float = 3.0,
    opacity: float = 1.0,
    duration: float = 400,
) -> dict[str, Any]:
    """Draw one stroke on the canvas.

    The canvas is a 1000 x 1000 normalized coordinate space (origin top-left).
    Strokes appear with a "falling" dashoffset animation so each one has
    visible duration — set `duration` to control how long that takes (ms).

    Args:
        path:     SVG path string. Examples:
                    'M 100 100 L 200 100'                 (straight line)
                    'M 100 200 Q 150 100 200 200'         (quadratic curve)
                    'M 100 100 C 150 50 250 50 300 100'   (cubic curve)
        stroke:   CSS color for the line. Default '#1a1a1a' (near-black).
        width:    Stroke width in canvas units (1000-wide canvas).
        opacity:  0.0 (invisible) to 1.0 (solid). Default 1.0.
        duration: How long the stroke takes to "fall" in ms. Default 400.
                  Use ~150 for quick scribbles, ~600+ for deliberate strokes.

    Returns:
        {ok, id, total}  — id of the stroke and total stroke count.
    """
    path = _validate_path(path)
    width = _clamp(float(width), 0.1, 80.0)
    opacity = _clamp(float(opacity), 0.0, 1.0)
    duration = _clamp(float(duration), 0.0, 5000.0)

    stroke_obj = {
        "id": f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}",
        "type": "path",
        "path": path,
        "stroke": stroke,
        "width": width,
        "opacity": opacity,
        "duration": duration,
    }
    total = _push_stroke_and_count(stroke_obj)
    return {"ok": True, "id": stroke_obj["id"], "total": total}


@mcp.tool
def pen_stroke(
    points: list[list[float]],
    stroke: str = "#1a1a1a",
    width: float = 5.0,
    opacity: float = 1.0,
    duration: float = 400,
) -> dict[str, Any]:
    """Draw one pen stroke as raw movement samples.

    The canvas is a 1000 x 1000 normalized coordinate space (origin top-left).
    Each point is an [x, y] position. The renderer connects points directly on
    a Canvas 2D surface without smoothing, so small movement choices remain
    visible instead of being turned into ideal SVG curves.

    Args:
        points:   List of [x, y] positions for one down-to-up pen stroke.
        stroke:   CSS color for the line. Default '#1a1a1a' (near-black).
        width:    Stroke width in canvas units (1000-wide canvas).
        opacity:  0.0 (invisible) to 1.0 (solid). Default 1.0.
        duration: How long the stroke takes to appear in ms. Default 400.

    Returns:
        {ok, id, total}  — id of the stroke and total stroke count.
    """
    points = _validate_points(points)
    width = _clamp(float(width), 0.1, 80.0)
    opacity = _clamp(float(opacity), 0.0, 1.0)
    duration = _clamp(float(duration), 0.0, 5000.0)

    stroke_obj = {
        "id": f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}",
        "type": "pen",
        "points": points,
        "stroke": stroke,
        "width": width,
        "opacity": opacity,
        "duration": duration,
    }
    total = _push_stroke_and_count(stroke_obj)
    return {"ok": True, "id": stroke_obj["id"], "total": total}


@mcp.tool
def canvas_clear() -> dict[str, Any]:
    """Clear the entire canvas."""
    _clear_strokes()
    return {"ok": True}


@mcp.tool(output_schema=None)
def canvas_snapshot() -> ToolResult:
    """Return the current canvas as a PNG image for visual feedback."""
    png = _render_snapshot_png()
    return ToolResult(
        content=[MCPImage(data=png, format="png").to_image_content()],
        structured_content=None,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    threading.Thread(target=_run_http, daemon=True).start()
    # Tiny delay so HTTP is up before we start blocking on stdio.
    time.sleep(0.2)
    mcp.run()

# Claude Canvas MCP

A small local MCP server that gives Claude a drawable canvas.

It exposes four tools:

- `canvas_draw(path)` for SVG-path strokes
- `pen_stroke(points)` for raw point-based pen movement
- `canvas_clear()` to wipe the canvas
- `canvas_snapshot()` to return the current canvas as a PNG image so the model can look back at what it has drawn

The server also starts a local HTTP page so you can watch the drawing in a browser.

## Requirements

- Python 3.11+
- `fastmcp`
- `fastapi`
- `uvicorn`
- `pillow`

## Install

```bash
/opt/homebrew/bin/python3.11 -m pip install --user -r requirements.txt
```

## Run

```bash
/opt/homebrew/bin/python3.11 canvas_mcp.py
```

Then open:

```txt
http://127.0.0.1:1573
```

## Claude Desktop config

Add this server under `mcpServers`:

```json
{
  "canvas": {
    "command": "/opt/homebrew/bin/python3.11",
    "args": [
      "/absolute/path/to/canvas_mcp.py"
    ]
  }
}
```

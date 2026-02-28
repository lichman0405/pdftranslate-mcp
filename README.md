# PDF2ZH — FeatherFlow MCP Server for PDF Translation

A slim [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) tool server that translates scientific PDF documents while **preserving formulas, charts, table of contents, and layout**. Designed to be launched and managed by [FeatherFlow](https://github.com/lichman0405/featherflow).

Based on [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate), stripped down to a single OpenAI-compatible translation backend — reusing the same LLM that FeatherFlow is already connected to.

## Features

- **MCP stdio transport** — plug-and-play with FeatherFlow (or any MCP-compatible host)
- **Preserves formulas & layout** — powered by ONNX-based document layout analysis + pdfminer/pymupdf
- **Dual output** — generates both *mono* (translated-only) and *dual* (bilingual side-by-side) PDFs
- **Shares FeatherFlow's LLM** — OpenAI-compatible endpoint via environment variables, no extra API key needed
- **Cross-platform** — works on Linux and Windows; all paths use `pathlib` for portability

## MCP Tools

| Tool | Description |
|------|-------------|
| `translate_pdf` | Translate a PDF file. Accepts `file`, `lang_in`, `lang_out`, optional `output_dir`. Returns **absolute paths** to mono & dual PDFs. |
| `list_supported_languages` | List all supported language codes (`en`, `zh`, `ja`, `ko`, `fr`, `de`, etc.) |

### File Path Resolution

The `file` parameter of `translate_pdf` supports both absolute and relative paths:

- **Absolute path** — used as-is (e.g. `/home/user/.featherflow/workspace/paper.pdf`)
- **Relative path** — resolved against the **workspace directory** (defaults to `~/.featherflow/workspace`, overridable via the `WORKSPACE_DIR` environment variable)

Output PDFs are written to the workspace directory by default. The returned paths are always absolute, making them directly usable by other MCP tools (e.g. feishu-mcp `upload_file` / `upload_file_and_share`).

## Requirements

> **⚠️ Python Environment Isolation — Important**
>
> This project depends on `babeldoc` / `onnxruntime`, which require **Python ≥3.10, <3.13**.
> FeatherFlow itself may run on a different Python version (e.g. 3.13+).
> You **must** create a separate Python environment for this project and point FeatherFlow's MCP config to this project's Python executable — not FeatherFlow's own Python.

- Python 3.10 – 3.12 (recommended: **3.12**)
- [uv](https://github.com/astral-sh/uv) (recommended), Conda, or virtualenv for environment isolation

## Installation

### 0. Install uv (one-time setup, recommended)

uv can automatically download and manage any Python version — no need to
install Python 3.12 manually.

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installation, then verify:

```bash
uv --version
```

### 1. Create a dedicated Python 3.12 environment

**Using uv** (recommended — auto-downloads Python 3.12 even if you only have 3.13+):

```bash
cd /path/to/pdftranslate-mcp
uv venv .venv --python 3.12
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

**Alternative: Conda**

```bash
conda create -p /path/to/pdftranslate-mcp/.venv python=3.12 -y
conda activate /path/to/pdftranslate-mcp/.venv
```

**Alternative: venv** (only if system Python is already 3.10–3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell
```

### 2. Install the package

```bash
pip install -e .
```

Or with uv (10-100x faster):

```bash
uv pip install -e .
```

This installs all dependencies: `pymupdf`, `pdfminer-six`, `babeldoc`, `onnxruntime`, `openai`, `mcp`, etc.

### 3. Verify

```bash
python -m pdf2zh.mcp_server --help
```

## FeatherFlow Configuration

Edit `~/.featherflow/config.json` (or the config file for your setup). Add `pdf2zh` under `tools.mcpServers`.

> **Key point:** The `command` must point to this project's own Python executable, **not** FeatherFlow's Python. This project requires Python <3.13, while FeatherFlow may run on a newer version.

### Example (Linux — production server)

```json
{
  "tools": {
    "mcpServers": {
      "pdf2zh": {
        "command": "/opt/PDFMathTranslate/.venv/bin/python",
        "args": ["-m", "pdf2zh.mcp_server"],
        "toolTimeout": 600,
        "env": {
          "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
          "OPENAI_API_KEY": "sk-or-v1-xxxxxxxx",
          "OPENAI_MODEL": "anthropic/claude-opus-4-5"
        }
      }
    }
  }
}
```

### Example (Windows — development)

```json
{
  "tools": {
    "mcpServers": {
      "pdf2zh": {
        "command": "C:/Users/<you>/code/PDFMathTranslate/.venv/python.exe",
        "args": ["-m", "pdf2zh.mcp_server"],
        "toolTimeout": 600,
        "env": {
          "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
          "OPENAI_API_KEY": "sk-or-v1-xxxxxxxx",
          "OPENAI_MODEL": "anthropic/claude-opus-4-5"
        }
      }
    }
  }
}
```

> **Tip:** On Windows, use forward slashes `/` in JSON paths — they work fine with Python's `pathlib`.

### Tool Timeout — Critical for PDF Translation

> **⚠️ You MUST set `toolTimeout` for the pdf2zh MCP server.**
>
> FeatherFlow's default MCP tool timeout is **30 seconds**. PDF translation is a heavy operation — a 6-page paper typically takes **1–5 minutes** depending on the LLM speed. Without increasing `toolTimeout`, the tool call will be cancelled mid-translation, and the agent will report a failure.
>
> **Recommended:** `"toolTimeout": 600` (10 minutes). For very long documents (50+ pages), consider `1200` (20 minutes).

```json
"pdf2zh": {
  "command": "...",
  "args": ["-m", "pdf2zh.mcp_server"],
  "toolTimeout": 600,
  ...
}
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_BASE_URL` | Yes | API base URL (e.g. `https://openrouter.ai/api/v1`, `https://api.openai.com/v1`) |
| `OPENAI_API_KEY` | Yes | API key for the provider |
| `OPENAI_MODEL` | Yes | Model identifier (e.g. `anthropic/claude-opus-4-5`, `gpt-4o`) |
| `OPENAI_TEMPERATURE` | No | Override LLM temperature. **Omit for reasoning models** (e.g. `kimi-k2.5`) which enforce their own temperature. Set to `0` for deterministic output with standard models. |
| `WORKSPACE_DIR` | No | Shared workspace directory. Defaults to `~/.featherflow/workspace` (same as FeatherFlow's built-in file tools and `paper_download`). Override this if your workspace is at a non-standard location. |

The `OPENAI_*` variables are the same credentials FeatherFlow uses — just pass them through via `env`. `WORKSPACE_DIR` usually does not need to be set; it automatically uses FeatherFlow's default workspace.

## Standalone Usage (without FeatherFlow)

### stdio mode (default)

```bash
python -m pdf2zh.mcp_server
```

### SSE mode (for web-based MCP clients)

```bash
python -m pdf2zh.mcp_server --sse --host 0.0.0.0 --port 3001
```

## Cross-MCP Workflow: pdf2zh + feishu-mcp

This project is designed to work alongside [feishu-mcp](https://github.com/lichman0405/feishu-mcp). A typical end-to-end flow:

```
User: "Translate this paper and share it in the Feishu group"
  ↓
FeatherFlow (LLM orchestration):
  1. paper_download → ~/.featherflow/workspace/paper.pdf
  2. pdf2zh.translate_pdf(file="paper.pdf", lang_in="en", lang_out="zh")
     → ~/.featherflow/workspace/paper-mono.pdf
     → ~/.featherflow/workspace/paper-dual.pdf
  3. feishu-mcp.upload_file_and_share(file_path="/home/user/.featherflow/workspace/paper-dual.pdf")
     → share_url
  4. feishu-mcp.send_message(chat_id, share_url)
```

**Why this works seamlessly:**

- pdf2zh writes output to `~/.featherflow/workspace` by default
- feishu-mcp `upload_file` / `upload_file_and_share` accepts absolute file paths
- pdf2zh returns absolute paths in its result — the LLM can extract and pass them directly to feishu-mcp
- Both MCP servers run as local processes on the same machine, sharing the same filesystem

## Project Structure

```
pdf2zh/
  __init__.py        # Package entry, exports translate_stream
  mcp_server.py      # MCP server (entry point, tools definition)
  translator.py      # BaseTranslator + OpenAITranslator
  converter.py       # PDF content conversion & layout processing
  high_level.py      # Core translation pipeline (translate_stream)
  config.py          # Configuration & constants
  cache.py           # Translation cache (SQLite via peewee)
  doclayout.py       # ONNX document layout model loading
  pdfinterp.py       # Extended PDF interpreter
pyproject.toml       # Dependencies & build config
```

## License

[AGPL-3.0](LICENSE)

## Credits

- Core PDF translation engine from [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate) by Byaidu
- MCP host integration for [FeatherFlow](https://github.com/lichman0405/featherflow)

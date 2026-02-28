"""
PDF2ZH MCP Server — a slim MCP tool server for PDF translation.

Designed to be launched by FeatherFlow (or any MCP-compatible host) via
the stdio transport.

Environment variables:
    OPENAI_BASE_URL  – e.g. https://openrouter.ai/api/v1
    OPENAI_API_KEY   – API key for the provider
    OPENAI_MODEL     – model identifier (e.g. anthropic/claude-opus-4-5)
    WORKSPACE_DIR    – shared workspace directory (defaults to
                       ~/.featherflow/workspace). Relative file paths
                       in tool calls are resolved against this directory,
                       and it is also the default output location.

FeatherFlow config example (in ~/.featherflow/config.json):
    {
      "tools": {
        "mcpServers": {
          "pdf2zh": {
            "command": "/path/to/PDFMathTranslate/.venv/bin/python",
            "args": ["-m", "pdf2zh.mcp_server"],
            "env": {
              "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
              "OPENAI_API_KEY": "sk-or-v1-xxx",
              "OPENAI_MODEL": "anthropic/claude-opus-4-5"
            }
          }
        }
      }
    }

Note: WORKSPACE_DIR defaults to ~/.featherflow/workspace — the same
directory FeatherFlow's built-in file tools and paper_download use.
This ensures that output PDFs are accessible to other MCP servers
(e.g. feishu-mcp upload_file / upload_file_and_share).
"""

from mcp.server.fastmcp import FastMCP, Context
from pdf2zh import translate_stream
from pdf2zh.doclayout import ModelInstance, OnnxModel
from pathlib import Path

import contextlib
import io
import logging
import os

logger = logging.getLogger(__name__)


def _get_workspace() -> Path:
    """Return the shared workspace directory.

    Priority: WORKSPACE_DIR env → ~/.featherflow/workspace (default).
    Creates the directory if it does not exist.
    """
    ws = os.environ.get("WORKSPACE_DIR")
    if ws:
        p = Path(ws).expanduser().resolve()
    else:
        p = Path.home() / ".featherflow" / "workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_file(file: str) -> Path:
    """Resolve a file path.

    - Absolute paths are used as-is.
    - Relative paths are resolved against the workspace directory.
    """
    p = Path(file).expanduser()
    if not p.is_absolute():
        p = _get_workspace() / p
    return p.resolve()


def _ensure_model_loaded():
    """Lazily load the doc-layout ONNX model on first call."""
    if ModelInstance.value is None:
        ModelInstance.value = OnnxModel.load_available()


def create_mcp_app() -> FastMCP:
    mcp = FastMCP("pdf2zh")

    @mcp.tool()
    async def translate_pdf(
        file: str,
        lang_in: str,
        lang_out: str,
        output_dir: str = "",
        ctx: Context = None,
    ) -> str:
        """Translate a PDF file while preserving formulas and layout.

        Args:
            file: Path to the input PDF file.  Absolute paths are used
                  as-is; relative paths are resolved against the shared
                  workspace directory (~/.featherflow/workspace by default,
                  overridable via WORKSPACE_DIR env var).
            lang_in: Source language code (e.g. "en", "auto" for auto-detect).
            lang_out: Target language code (e.g. "zh", "ja", "ko", "fr", "de").
            output_dir: Directory for output files. Defaults to the shared
                        workspace directory so that other MCP tools
                        (e.g. feishu-mcp upload_file) can access the outputs.

        Returns:
            A summary with absolute paths to the mono (translated-only) and
            dual (bilingual side-by-side) output PDF files.  These paths
            can be passed directly to feishu-mcp upload_file / upload_file_and_share.

        Environment variables that control the LLM used for translation:
            OPENAI_BASE_URL – API base URL
            OPENAI_API_KEY  – API key
            OPENAI_MODEL    – Model name
        """
        file_path = _resolve_file(file)
        if not file_path.exists():
            return f"Error: file not found: {file} (resolved to {file_path})"
        if not file_path.suffix.lower() == ".pdf":
            return f"Error: not a PDF file: {file}"

        _ensure_model_loaded()

        if ctx:
            await ctx.log(level="info", message=f"Starting translation of {file}")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Build envs from the environment that FeatherFlow passes in
        envs = {}
        for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
            val = os.environ.get(key)
            if val:
                envs[key] = val

        with contextlib.redirect_stdout(io.StringIO()):
            doc_mono_bytes, doc_dual_bytes = translate_stream(
                file_bytes,
                lang_in=lang_in,
                lang_out=lang_out,
                service="openai",
                model=ModelInstance.value,
                thread=4,
                envs=envs,
            )

        # Determine output path — default to workspace (shared with FeatherFlow
        # and other MCP servers like feishu-mcp)
        if output_dir:
            out_dir = _resolve_file(output_dir)
        else:
            out_dir = _get_workspace()
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = file_path.stem

        doc_mono_path = out_dir / f"{filename}-mono.pdf"
        doc_dual_path = out_dir / f"{filename}-dual.pdf"

        with open(doc_mono_path, "wb") as f:
            f.write(doc_mono_bytes)
        with open(doc_dual_path, "wb") as f:
            f.write(doc_dual_bytes)

        if ctx:
            await ctx.log(level="info", message="Translation complete")

        return (
            f"Translation complete.\n"
            f"  Mono PDF (translated only): {doc_mono_path.absolute()}\n"
            f"  Dual PDF (bilingual):       {doc_dual_path.absolute()}"
        )

    @mcp.tool()
    async def list_supported_languages(ctx: Context = None) -> str:
        """List language codes supported for translation.

        Returns a table of common language codes that can be used as
        lang_in or lang_out parameters.
        """
        languages = {
            "en": "English",
            "zh": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "ru": "Russian",
            "pt": "Portuguese",
            "ar": "Arabic",
            "th": "Thai",
            "hi": "Hindi",
            "uk": "Ukrainian",
            "auto": "Auto-detect (lang_in only)",
        }
        lines = ["Code | Language", "---- | --------"]
        for code, name in languages.items():
            lines.append(f"{code} | {name}")
        return "\n".join(lines)

    return mcp


# ── Entry point ──────────────────────────────────────────────────────
# Default: stdio transport (for FeatherFlow / MCP hosts)
# With --sse: SSE transport (for web-based clients)


def main():
    """CLI entry point for pdf2zh MCP server."""
    import argparse

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("mcp").setLevel(logging.ERROR)
    # suppress noisy loggers
    for name in ("httpx", "openai", "httpcore", "http11"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False

    parser = argparse.ArgumentParser(
        description="PDF2ZH MCP Server — translate scientific PDFs"
    )
    parser.add_argument(
        "--sse",
        default=False,
        action="store_true",
        help="Run with SSE transport instead of stdio",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (SSE mode only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3001,
        help="Port to bind (SSE mode only)",
    )
    args = parser.parse_args()

    mcp = create_mcp_app()

    if args.sse:
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Mount, Route

        mcp_server = mcp._mcp_server
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request) -> None:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )

        starlette_app = Starlette(
            debug=False,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    else:
        # stdio — the standard mode for FeatherFlow integration
        mcp.run()


if __name__ == "__main__":
    main()

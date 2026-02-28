#!/usr/bin/env python3
"""pdf2zh — Scientific PDF translator (MCP server).

This module forwards to the MCP server entry point.
Run ``python -m pdf2zh.mcp_server`` or simply ``pdf2zh`` after
installing the package.
"""

from pdf2zh.mcp_server import main

if __name__ == "__main__":
    main()

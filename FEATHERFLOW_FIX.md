# FeatherFlow MCP Progress — Correct Implementation

## Root Cause

The previous implementation used wrong MCP SDK APIs:
- `ClientSession(progress_notification_handler=...)` — this parameter does NOT exist
- `session.call_tool(progress_token=...)` — this parameter does NOT exist

The CORRECT API in mcp SDK is:
- `session.call_tool(progress_callback=...)` — takes an async callable
- No changes needed to `ClientSession.__init__()`

## Required Changes to `featherflow/agent/tools/mcp.py`

### Step 1: Remove all previous progress-related additions

Remove these items that were added in the previous commit:
- The `_progress_callbacks: dict` module-level variable
- The `_handle_progress_notification()` function
- The `ClientSession(progress_notification_handler=...)` constructor change
  (revert back to plain `ClientSession(read, write)`)
- The `progress_token` UUID generation in `execute()`

### Step 2: Rewrite the `execute()` method with correct API

Replace the entire `execute()` method with:

```python
async def execute(self, *, _on_progress=None, **kwargs: Any) -> str:
    from mcp import types

    # Build a progress_callback if the caller provided one.
    # The mcp SDK calls this as: await progress_callback(progress, total)
    # where progress and total are floats.
    progress_callback = None
    if _on_progress:
        async def progress_callback(progress: float, total: float | None) -> None:
            await _on_progress(progress, total or 0)

    try:
        result = await asyncio.wait_for(
            self._session.call_tool(
                self._original_name,
                arguments=kwargs,
                progress_callback=progress_callback,
            ),
            timeout=self._tool_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "MCP tool '{}' timed out after {}s", self._name, self._tool_timeout
        )
        return f"(MCP tool call timed out after {self._tool_timeout}s)"

    parts = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts) or "(no output)"
```

### What does NOT change

- `_on_progress` keyword argument — keep it
- `registry.py` `**extra` forwarding — keep it
- `loop.py` `_mcp_on_progress` callback — keep it

## Summary of net diff

After the change, `mcp.py` should:
1. `connect_mcp_servers`: use plain `ClientSession(read, write)` — no progress_notification_handler
2. `MCPToolWrapper.execute()`: use `call_tool(progress_callback=progress_callback)` — no progress_token, no UUID, no _progress_callbacks dict
3. No module-level `_progress_callbacks` dict
4. No `_handle_progress_notification` function

## Why this works

The mcp Python SDK (v1.26+) implements progress like this:
- Client passes `progress_callback` (an async function) to `call_tool()`
- SDK internally generates a progressToken and sends it to the server in `_meta`
- When server calls `ctx.report_progress(progress, total)`, SDK receives the
  `notifications/progress` message and calls the `progress_callback` function
- No manual token management needed on the client side

## No changes needed in other projects

- `pdftranslate-mcp`: `ctx.report_progress()` in `mcp_server.py` is already correct
- `feishu-mcp`: does not need progress reporting
- `zeopp-backend`: can add `ctx.report_progress()` calls the same way as pdftranslate-mcp

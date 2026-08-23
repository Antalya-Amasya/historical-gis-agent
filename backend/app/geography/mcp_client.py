import anyio
import json
import sys
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def parse_tool_result(result, tool: str):
    if result.isError:
        summary = "; ".join(getattr(item, "text", str(item)) for item in result.content)
        raise RuntimeError(f"MCP tool {tool} failed: {summary[:500]}")
    if result.structuredContent is not None:
        return result.structuredContent
    texts = [item.text for item in result.content if hasattr(item, "text")]
    if not texts:
        raise RuntimeError(f"MCP tool {tool} returned no structured content")
    text = "".join(texts)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    if not isinstance(value, (dict, list)):
        raise RuntimeError(f"MCP tool {tool} returned JSON scalar")
    return value


class GeographyMcpClient:
    def __init__(self, python_executable: str | None = None):
        self.python_executable = python_executable or sys.executable
    async def _call(self, tool, arguments):
        params = StdioServerParameters(command=self.python_executable,args=["-m","geography_mcp.mcp_server"])
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                return parse_tool_result(await session.call_tool(tool,arguments),tool)
    def call(self, tool, arguments):
        return anyio.run(self._call,tool,arguments)

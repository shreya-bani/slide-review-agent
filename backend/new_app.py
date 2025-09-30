from mcp.server import Server
from mcp.types import Tool, TextContent
import json

# Create MCP server instance
mcp_server = Server("slide-review-agent")

@app.post("/mcp/tools/list")
async def mcp_list_tools():
    """MCP endpoint: List available tools."""
    tools = [
        {
            "name": "analyze_document",
            "description": "Analyze a slide deck for style compliance",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "File ID from previous upload"
                    }
                },
                "required": ["file_id"]
            }
        },
        {
            "name": "get_analysis_result",
            "description": "Retrieve analysis results by file_id",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"}
                },
                "required": ["file_id"]
            }
        }
    ]
    return {"tools": tools}

@app.post("/mcp/tools/call")
async def mcp_call_tool(request: dict):
    """MCP endpoint: Call a tool."""
    tool_name = request.get("name")
    arguments = request.get("arguments", {})
    
    if tool_name == "analyze_document":
        file_id = arguments.get("file_id")
        
        # Find existing analysis
        output_dir = Path(settings.output_dir)
        analysis_file = output_dir / f"{file_id}_analysis.json"
        
        if not analysis_file.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({"error": "Analysis not found"})
                }]
            }
        
        with open(analysis_file, 'r') as f:
            result = json.load(f)
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }]
        }
    
    elif tool_name == "get_analysis_result":
        # Similar logic
        pass
    
    return {"content": [{"type": "text", "text": "Unknown tool"}]}
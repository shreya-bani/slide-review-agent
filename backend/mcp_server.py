"""
MCP Server for Slide Review Agent
Exposes document analysis capabilities via Model Context Protocol
"""
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from .processors.document_normalizer import DocumentNormalizer
from .analyzers.backup_simple_style_checker import check_document
from .config.settings import settings

# Initialize MCP server
mcp = Server("slide-review-agent")

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the MCP client."""
    return [
        Tool(
            name="analyze_document",
            description="Analyze a slide deck or PDF for style compliance with Amida Style Guide",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to PPTX or PDF file to analyze"
                    },
                    "include_ai": {
                        "type": "boolean",
                        "description": "Include AI-powered analysis (slower)",
                        "default": False
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="get_style_rules",
            description="Get list of style rules from Amida Style Guide",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["voice", "grammar", "formatting", "inclusivity", "all"],
                        "description": "Filter rules by category"
                    }
                }
            }
        ),
        Tool(
            name="list_analyses",
            description="List previous document analyses",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                }
            }
        )
    ]

@mcp.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls from MCP client."""
    
    if name == "analyze_document":
        return await analyze_document_tool(arguments)
    elif name == "get_style_rules":
        return await get_style_rules_tool(arguments)
    elif name == "list_analyses":
        return await list_analyses_tool(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def analyze_document_tool(args: dict) -> Sequence[TextContent]:
    """Analyze a document and return findings."""
    file_path = args.get("file_path")
    include_ai = args.get("include_ai", False)
    
    if not file_path or not Path(file_path).exists():
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"File not found: {file_path}"})
        )]
    
    try:
        # Normalize document
        normalizer = DocumentNormalizer()
        normalized_obj = await asyncio.to_thread(
            normalizer.normalize_document, 
            file_path
        )
        normalized = normalized_obj.to_dict()
        
        # Run style checks
        findings = await asyncio.to_thread(check_document, normalized)
        
        # TODO: Add AI analysis if requested
        if include_ai:
            # Integrate your AI checker here
            pass
        
        result = {
            "success": True,
            "file_path": file_path,
            "document_type": normalized["document_type"],
            "total_pages": normalized["summary"]["total_pages"],
            "total_findings": len(findings),
            "findings": findings[:50],  # Limit for readability
            "findings_by_severity": {
                "critical": len([f for f in findings if f["severity"] == "critical"]),
                "warning": len([f for f in findings if f["severity"] == "warning"]),
                "suggestion": len([f for f in findings if f["severity"] == "suggestion"])
            }
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

async def get_style_rules_tool(args: dict) -> Sequence[TextContent]:
    """Get style rules from configuration."""
    from .config.style_rules import amida_rules
    
    category = args.get("category", "all")
    
    if category == "all":
        rules = amida_rules.rules
    else:
        rules = amida_rules.get_rules_by_category(category)
    
    rules_data = [
        {
            "name": r.name,
            "category": r.category,
            "severity": r.severity,
            "description": r.description,
            "guide_section": r.guide_section,
            "examples": r.examples
        }
        for r in rules
    ]
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "total_rules": len(rules_data),
            "category": category,
            "rules": rules_data
        }, indent=2)
    )]

async def list_analyses_tool(args: dict) -> Sequence[TextContent]:
    """List previous analyses."""
    limit = args.get("limit", 10)
    
    output_dir = Path(settings.output_dir)
    analysis_files = sorted(
        output_dir.glob("*_analysis.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:limit]
    
    history = []
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                history.append({
                    "file_id": data.get("file_id"),
                    "filename": data.get("original_filename"),
                    "processed_at": data.get("processed_at"),
                    "total_pages": data.get("processing_summary", {}).get("total_pages")
                })
        except:
            continue
    
    return [TextContent(
        type="text",
        text=json.dumps({"analyses": history}, indent=2)
    )]

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
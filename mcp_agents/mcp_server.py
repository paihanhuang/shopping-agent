#!/usr/bin/env python3
"""
MCP Server for Shopping Agent - Stdio Transport

This server implements the Model Context Protocol (MCP) specification
and can be connected to Claude Desktop or other MCP-compatible clients.

Usage:
  python -m mcp_agents.mcp_server

Or add to Claude Desktop config:
  {
    "mcpServers": {
      "shopping-agent": {
        "command": "python",
        "args": ["-m", "mcp_agents.mcp_server"],
        "cwd": "/path/to/shopping-agent"
      }
    }
  }
"""

import asyncio
import json
import sys
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# MCP Protocol Types
# ============================================================================

@dataclass
class JsonRpcRequest:
    jsonrpc: str
    method: str
    id: Optional[int] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class JsonRpcResponse:
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class ServerInfo:
    name: str = "shopping-agent"
    version: str = "1.0.0"


# ============================================================================
# Shopping Tools Implementation
# ============================================================================

class ShoppingTools:
    """Implementation of shopping-related MCP tools."""
    
    @staticmethod
    def get_tools() -> List[Tool]:
        """Return all available tools."""
        return [
            Tool(
                name="search_product_prices",
                description="Search for product prices across multiple retailers. Returns prices, URLs, tax estimates, and shipping costs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_query": {
                            "type": "string",
                            "description": "The product to search for (e.g., 'PlayStation 5', 'AirPods Pro 3')"
                        }
                    },
                    "required": ["product_query"]
                }
            ),
            Tool(
                name="lookup_cashback_rates",
                description="Look up cashback rates from shopping portals (Rakuten, Capital One Shopping, ShopBack) for specific retailers.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "retailers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of retailer names to look up cashback for"
                        },
                        "category": {
                            "type": "string",
                            "description": "Product category (Electronics, Clothing, Home, Beauty, General)",
                            "default": "General"
                        }
                    },
                    "required": ["retailers"]
                }
            ),
            Tool(
                name="get_credit_card_recommendations",
                description="Get credit card recommendations for maximizing rewards at specific retailers.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "retailers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of retailer names"
                        }
                    },
                    "required": ["retailers"]
                }
            ),
            Tool(
                name="verify_product_url",
                description="Verify if a URL is a valid product page (not a homepage or search results page).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to verify"
                        },
                        "expected_product": {
                            "type": "string",
                            "description": "The product that should be on this page"
                        }
                    },
                    "required": ["url", "expected_product"]
                }
            ),
            Tool(
                name="complete_shopping_search",
                description="Run a complete shopping search with price comparison, cashback rates, and credit card recommendations. This is the main tool for finding the best deal.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_query": {
                            "type": "string",
                            "description": "The product to search for"
                        }
                    },
                    "required": ["product_query"]
                }
            ),
            Tool(
                name="start_price_tracking",
                description="Start tracking prices for a product over time. Monitors prices periodically and alerts on significant changes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_query": {
                            "type": "string",
                            "description": "The product to track"
                        },
                        "interval_minutes": {
                            "type": "integer",
                            "description": "How often to check prices (in minutes)",
                            "default": 60
                        },
                        "duration_hours": {
                            "type": "integer",
                            "description": "How long to track (in hours). Omit for indefinite tracking."
                        }
                    },
                    "required": ["product_query"]
                }
            ),
            Tool(
                name="get_tracking_statistics",
                description="Get price statistics for a tracking session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "integer",
                            "description": "The tracking session ID"
                        }
                    },
                    "required": ["session_id"]
                }
            )
        ]
    
    @staticmethod
    async def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        try:
            if name == "search_product_prices":
                return await ShoppingTools._search_products(arguments["product_query"])
            
            elif name == "lookup_cashback_rates":
                return await ShoppingTools._lookup_cashback(
                    arguments["retailers"],
                    arguments.get("category", "General")
                )
            
            elif name == "get_credit_card_recommendations":
                return ShoppingTools._get_card_recommendations(arguments["retailers"])
            
            elif name == "verify_product_url":
                return ShoppingTools._verify_url(
                    arguments["url"],
                    arguments["expected_product"]
                )
            
            elif name == "complete_shopping_search":
                return await ShoppingTools._complete_search(arguments["product_query"])
            
            elif name == "start_price_tracking":
                return ShoppingTools._start_tracking(
                    arguments["product_query"],
                    arguments.get("interval_minutes", 60),
                    arguments.get("duration_hours")
                )
            
            elif name == "get_tracking_statistics":
                return ShoppingTools._get_tracking_stats(arguments["session_id"])
            
            else:
                return f"Unknown tool: {name}"
        
        except Exception as e:
            return f"Error executing tool {name}: {str(e)}"
    
    @staticmethod
    async def _search_products(query: str) -> str:
        """Search for product prices."""
        from main import search_product_prices
        return search_product_prices(query)
    
    @staticmethod
    async def _lookup_cashback(retailers: List[str], category: str) -> str:
        """Look up cashback rates."""
        from main import lookup_cashback_rates
        return lookup_cashback_rates(category)
    
    @staticmethod
    def _get_card_recommendations(retailers: List[str]) -> str:
        """Get credit card recommendations."""
        recommendations = []
        
        card_rates = {
            "Amazon": ("Amazon Prime Visa", "5% back"),
            "Costco": ("Costco Anywhere Visa", "2% back (Visa only!)"),
            "Best Buy": ("BofA Customized Cash", "3% back (Online Shopping category)"),
            "Walmart": ("BofA Customized Cash", "3% back (Online Shopping category)"),
            "Target": ("Target RedCard", "5% back"),
            "B&H Photo": ("BofA Customized Cash", "3% back (Online Shopping category)"),
            "Newegg": ("BofA Customized Cash", "3% back (Online Shopping category)"),
        }
        
        default = ("Citi Double Cash", "2% back on everything")
        
        for retailer in retailers:
            card, rate = card_rates.get(retailer, default)
            recommendations.append(f"• {retailer}: {card} - {rate}")
        
        result = "💳 Credit Card Recommendations:\n\n"
        result += "\n".join(recommendations)
        result += "\n\n⚠️ Notes:\n"
        result += "• Costco only accepts Visa cards\n"
        result += "• BofA Customized Cash requires setting Online Shopping as your 3% category"
        
        return result
    
    @staticmethod
    def _verify_url(url: str, expected_product: str) -> str:
        """Verify a product URL."""
        issues = []
        
        # Check for homepage
        if url.count('/') <= 3:
            issues.append("URL appears to be a homepage, not a product page")
        
        # Check for search results
        search_indicators = ['/search', '/s?', 'searchpage', 'query=', 'q=']
        for indicator in search_indicators:
            if indicator in url.lower():
                issues.append(f"URL appears to be a search page (contains '{indicator}')")
                break
        
        # Check for product identifiers
        product_indicators = ['/dp/', '/ip/', '/product/', '/p/', 'sku=', 'pid=']
        has_product_id = any(ind in url.lower() for ind in product_indicators)
        
        if not has_product_id and not issues:
            issues.append("URL may not contain a product identifier")
        
        if issues:
            return "❌ URL Validation Issues:\n" + "\n".join(f"  • {i}" for i in issues)
        else:
            return f"✅ URL appears to be a valid product page for '{expected_product}'"
    
    @staticmethod
    async def _complete_search(query: str) -> str:
        """Run complete shopping search."""
        from main import search_product_prices
        return search_product_prices(query)
    
    @staticmethod
    def _start_tracking(query: str, interval: int, duration: Optional[int]) -> str:
        """Start price tracking."""
        from price_tracker import PriceTracker
        
        tracker = PriceTracker()
        session_id = tracker.start_tracking(query, interval, duration)
        
        return f"""✅ Price Tracking Started!

📋 Session ID: {session_id}
📦 Product: {query}
⏱️ Interval: Every {interval} minutes
⏳ Duration: {duration if duration else 'Indefinite'} hours

Use get_tracking_statistics with session_id={session_id} to check progress."""
    
    @staticmethod
    def _get_tracking_stats(session_id: int) -> str:
        """Get tracking statistics."""
        from price_tracker import PriceTracker
        import io
        import sys
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        tracker = PriceTracker()
        tracker.get_statistics(session_id)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        return output if output else f"No statistics available for session {session_id}"


# ============================================================================
# MCP Server Implementation
# ============================================================================

class MCPServer:
    """
    MCP Server implementation using stdio transport.
    Handles JSON-RPC messages according to MCP specification.
    """
    
    def __init__(self):
        self.tools = ShoppingTools()
    
    async def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Handle an incoming MCP request."""
        method = request.method
        params = request.params or {}
        
        try:
            if method == "initialize":
                return JsonRpcResponse(
                    id=request.id,
                    result={
                        "protocolVersion": "2024-11-05",
                        "serverInfo": asdict(ServerInfo()),
                        "capabilities": {
                            "tools": {}
                        }
                    }
                )
            
            elif method == "notifications/initialized":
                # Client acknowledged initialization
                return None  # No response needed for notifications
            
            elif method == "tools/list":
                tools = ShoppingTools.get_tools()
                return JsonRpcResponse(
                    id=request.id,
                    result={
                        "tools": [asdict(t) for t in tools]
                    }
                )
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                result = await ShoppingTools.execute_tool(tool_name, arguments)
                
                return JsonRpcResponse(
                    id=request.id,
                    result={
                        "content": [
                            {"type": "text", "text": result}
                        ]
                    }
                )
            
            else:
                return JsonRpcResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                )
        
        except Exception as e:
            return JsonRpcResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": str(e)
                }
            )
    
    async def run_stdio(self):
        """Run the server using stdio transport."""
        # Log to stderr so it doesn't interfere with stdio protocol
        sys.stderr.write("🛒 Shopping Agent MCP Server starting...\n")
        sys.stderr.write("   Waiting for MCP client connection via stdio\n")
        sys.stderr.flush()
        
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())
        
        while True:
            try:
                # Read line from stdin
                line = await reader.readline()
                if not line:
                    break
                
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                
                # Parse JSON-RPC request
                data = json.loads(line)
                request = JsonRpcRequest(
                    jsonrpc=data.get("jsonrpc", "2.0"),
                    method=data["method"],
                    id=data.get("id"),
                    params=data.get("params")
                )
                
                # Handle request
                response = await self.handle_request(request)
                
                # Send response (if not a notification)
                if response is not None:
                    response_json = json.dumps({
                        "jsonrpc": response.jsonrpc,
                        "id": response.id,
                        **({"result": response.result} if response.result is not None else {}),
                        **({"error": response.error} if response.error is not None else {})
                    })
                    writer.write((response_json + "\n").encode('utf-8'))
                    await writer.drain()
            
            except json.JSONDecodeError as e:
                sys.stderr.write(f"JSON decode error: {e}\n")
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")


async def main():
    """Main entry point."""
    server = MCPServer()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())


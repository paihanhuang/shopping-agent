"""
MCP Server implementations for Shopping Agents.

Each server exposes tools via Model Context Protocol for:
- Product Search
- Cashback Lookup
- Credit Card Rewards
- Result Verification
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# MCP Protocol Types
@dataclass
class Tool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ToolResult:
    """Result from tool execution."""
    content: str
    is_error: bool = False


@dataclass
class MCPMessage:
    """MCP protocol message."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


class MCPServer(ABC):
    """Base class for MCP servers."""
    
    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, Tool] = {}
        self._register_tools()
    
    @abstractmethod
    def _register_tools(self):
        """Register tools provided by this server."""
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool with given arguments."""
        pass
    
    def list_tools(self) -> List[Tool]:
        """List all available tools."""
        return list(self.tools.values())
    
    async def handle_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle incoming MCP message."""
        if message.method == "tools/list":
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema
                    }
                    for t in self.list_tools()
                ]
            }
        elif message.method == "tools/call":
            tool_name = message.params.get("name")
            arguments = message.params.get("arguments", {})
            result = await self.execute_tool(tool_name, arguments)
            return {
                "content": [{"type": "text", "text": result.content}],
                "isError": result.is_error
            }
        else:
            return {"error": f"Unknown method: {message.method}"}


class ProductSearchServer(MCPServer):
    """MCP Server for product price searches."""
    
    def __init__(self):
        super().__init__("product-search")
    
    def _register_tools(self):
        self.tools["search_product"] = Tool(
            name="search_product",
            description="Search for product prices across multiple retailers",
            input_schema={
                "type": "object",
                "properties": {
                    "product_query": {
                        "type": "string",
                        "description": "Product to search for (e.g., 'PlayStation 5')"
                    },
                    "max_retailers": {
                        "type": "integer",
                        "description": "Maximum number of retailers to search",
                        "default": 15
                    }
                },
                "required": ["product_query"]
            }
        )
        
        self.tools["search_retailer"] = Tool(
            name="search_retailer",
            description="Search for a product at a specific retailer",
            input_schema={
                "type": "object",
                "properties": {
                    "product_query": {"type": "string"},
                    "retailer": {"type": "string", "description": "Retailer name (e.g., 'Amazon', 'Best Buy')"}
                },
                "required": ["product_query", "retailer"]
            }
        )
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "search_product":
            return await self._search_product(
                arguments["product_query"],
                arguments.get("max_retailers", 15)
            )
        elif tool_name == "search_retailer":
            return await self._search_retailer(
                arguments["product_query"],
                arguments["retailer"]
            )
        else:
            return ToolResult(f"Unknown tool: {tool_name}", is_error=True)
    
    async def _search_product(self, query: str, max_retailers: int) -> ToolResult:
        """Search for product across retailers."""
        # Import here to avoid circular imports
        from langchain_openai import ChatOpenAI
        from langchain_tavily import TavilySearch
        from langchain.agents import create_agent
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        search = TavilySearch(max_results=10, search_depth="advanced")
        
        prompt = f"""Search for "{query}" prices at major retailers.
        For each retailer, find: name, product URL, price, tax estimate, shipping.
        Search at least {max_retailers} different retailers."""
        
        agent = create_agent(llm, [search], system_prompt="You are a product price search specialist.")
        result = agent.invoke({"messages": [("human", prompt)]})
        
        # Extract content from result
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, 'content') and msg.content and not getattr(msg, 'tool_calls', None):
                return ToolResult(msg.content)
        
        return ToolResult("No results found", is_error=True)
    
    async def _search_retailer(self, query: str, retailer: str) -> ToolResult:
        """Search for product at specific retailer."""
        from langchain_openai import ChatOpenAI
        from langchain_tavily import TavilySearch
        from langchain.agents import create_agent
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        search = TavilySearch(max_results=5)
        
        prompt = f'Search for "{query}" at {retailer}. Find the direct product page URL and current price.'
        
        agent = create_agent(llm, [search], system_prompt="You are a product search specialist.")
        result = agent.invoke({"messages": [("human", prompt)]})
        
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, 'content') and msg.content and not getattr(msg, 'tool_calls', None):
                return ToolResult(msg.content)
        
        return ToolResult("No results found", is_error=True)


class CashbackServer(MCPServer):
    """MCP Server for cashback rate lookups."""
    
    def __init__(self):
        super().__init__("cashback-lookup")
    
    def _register_tools(self):
        self.tools["lookup_cashback"] = Tool(
            name="lookup_cashback",
            description="Look up cashback rates from shopping portals for retailers",
            input_schema={
                "type": "object",
                "properties": {
                    "retailers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of retailer names"
                    },
                    "category": {
                        "type": "string",
                        "description": "Product category (Electronics, Clothing, etc.)"
                    }
                },
                "required": ["retailers"]
            }
        )
        
        self.tools["get_portal_rates"] = Tool(
            name="get_portal_rates",
            description="Get all current rates for a specific shopping portal",
            input_schema={
                "type": "object",
                "properties": {
                    "portal": {
                        "type": "string",
                        "enum": ["Rakuten", "Capital One Shopping", "ShopBack"],
                        "description": "Shopping portal name"
                    }
                },
                "required": ["portal"]
            }
        )
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "lookup_cashback":
            return await self._lookup_cashback(
                arguments["retailers"],
                arguments.get("category", "General")
            )
        elif tool_name == "get_portal_rates":
            return await self._get_portal_rates(arguments["portal"])
        else:
            return ToolResult(f"Unknown tool: {tool_name}", is_error=True)
    
    async def _lookup_cashback(self, retailers: List[str], category: str) -> ToolResult:
        """Look up cashback rates for retailers."""
        from langchain_openai import ChatOpenAI
        from langchain_tavily import TavilySearch
        from langchain.agents import create_agent
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        search = TavilySearch(max_results=10)
        
        retailers_str = ", ".join(retailers)
        prompt = f"""Find current cashback rates for {category} purchases at: {retailers_str}
        
        Check these portals:
        - Rakuten
        - Capital One Shopping
        - ShopBack
        
        Known exclusions: Costco (no cashback), Apple Store (very limited)
        
        Format: [Retailer]: Rakuten X%, Capital One X%, ShopBack X%"""
        
        agent = create_agent(llm, [search], system_prompt="You are a cashback research specialist.")
        result = agent.invoke({"messages": [("human", prompt)]})
        
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, 'content') and msg.content and not getattr(msg, 'tool_calls', None):
                return ToolResult(msg.content)
        
        return ToolResult("No cashback data found", is_error=True)
    
    async def _get_portal_rates(self, portal: str) -> ToolResult:
        """Get all rates for a specific portal."""
        from langchain_openai import ChatOpenAI
        from langchain_tavily import TavilySearch
        from langchain.agents import create_agent
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        search = TavilySearch(max_results=10)
        
        prompt = f"Find current {portal} cashback rates for major retailers. List retailer and percentage."
        
        agent = create_agent(llm, [search], system_prompt="You are a cashback research specialist.")
        result = agent.invoke({"messages": [("human", prompt)]})
        
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, 'content') and msg.content and not getattr(msg, 'tool_calls', None):
                return ToolResult(msg.content)
        
        return ToolResult("No data found", is_error=True)


class CreditCardServer(MCPServer):
    """MCP Server for credit card rewards lookups."""
    
    def __init__(self):
        super().__init__("credit-card-rewards")
    
    def _register_tools(self):
        self.tools["get_card_rewards"] = Tool(
            name="get_card_rewards",
            description="Get reward rates for a specific credit card",
            input_schema={
                "type": "object",
                "properties": {
                    "card_name": {
                        "type": "string",
                        "description": "Credit card name (e.g., 'Citi Double Cash')"
                    }
                },
                "required": ["card_name"]
            }
        )
        
        self.tools["recommend_card"] = Tool(
            name="recommend_card",
            description="Recommend best credit card for specific retailers",
            input_schema={
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
        )
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "get_card_rewards":
            return await self._get_card_rewards(arguments["card_name"])
        elif tool_name == "recommend_card":
            return await self._recommend_card(arguments["retailers"])
        else:
            return ToolResult(f"Unknown tool: {tool_name}", is_error=True)
    
    async def _get_card_rewards(self, card_name: str) -> ToolResult:
        """Get rewards for a specific card."""
        from langchain_openai import ChatOpenAI
        from langchain_tavily import TavilySearch
        from langchain.agents import create_agent
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        search = TavilySearch(max_results=5)
        
        prompt = f"Find current reward rates for {card_name} credit card. Include base rate and bonus categories."
        
        agent = create_agent(llm, [search], system_prompt="You are a credit card rewards specialist.")
        result = agent.invoke({"messages": [("human", prompt)]})
        
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, 'content') and msg.content and not getattr(msg, 'tool_calls', None):
                return ToolResult(msg.content)
        
        return ToolResult("No data found", is_error=True)
    
    async def _recommend_card(self, retailers: List[str]) -> ToolResult:
        """Recommend best cards for retailers."""
        # Use known data for common cases
        recommendations = []
        
        card_rates = {
            "Amazon": ("Amazon Prime Visa", "5%"),
            "Costco": ("Costco Anywhere Visa", "2% (Visa only)"),
            "Best Buy": ("BofA Customized Cash", "3% (Online Shopping)"),
            "Walmart": ("BofA Customized Cash", "3% (Online Shopping)"),
            "Target": ("Target RedCard", "5%"),
        }
        
        default_card = ("Citi Double Cash", "2%")
        
        for retailer in retailers:
            card, rate = card_rates.get(retailer, default_card)
            recommendations.append(f"{retailer}: {card} - {rate}")
        
        result = "Credit Card Recommendations:\n" + "\n".join(recommendations)
        result += "\n\nNote: Costco only accepts Visa cards."
        
        return ToolResult(result)


class VerificationServer(MCPServer):
    """MCP Server for result verification and cleanup."""
    
    def __init__(self):
        super().__init__("verification")
    
    def _register_tools(self):
        self.tools["verify_results"] = Tool(
            name="verify_results",
            description="Verify and clean shopping results",
            input_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "string",
                        "description": "Raw shopping results to verify"
                    },
                    "product_query": {
                        "type": "string",
                        "description": "Original product query"
                    }
                },
                "required": ["results", "product_query"]
            }
        )
        
        self.tools["validate_url"] = Tool(
            name="validate_url",
            description="Validate if a URL is a valid product page",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "retailer": {"type": "string"},
                    "expected_product": {"type": "string"}
                },
                "required": ["url", "expected_product"]
            }
        )
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == "verify_results":
            return await self._verify_results(
                arguments["results"],
                arguments["product_query"]
            )
        elif tool_name == "validate_url":
            return self._validate_url(
                arguments["url"],
                arguments.get("retailer", ""),
                arguments["expected_product"]
            )
        else:
            return ToolResult(f"Unknown tool: {tool_name}", is_error=True)
    
    async def _verify_results(self, results: str, product_query: str) -> ToolResult:
        """Verify and clean results."""
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = f"""Verify and clean these shopping results for "{product_query}":

{results}

Tasks:
1. Remove duplicate retailers
2. Remove non-retailers (news sites, deal aggregators)
3. Mark invalid URLs (homepages, search pages)
4. Flag wrong products (different model/version)

Return cleaned results only."""
        
        response = llm.invoke([
            {"role": "system", "content": "You are a data verification specialist."},
            {"role": "user", "content": prompt}
        ])
        
        return ToolResult(response.content)
    
    def _validate_url(self, url: str, retailer: str, expected_product: str) -> ToolResult:
        """Validate a product URL."""
        issues = []
        
        # Check for homepage
        if url.count('/') <= 3:
            issues.append("URL appears to be a homepage, not a product page")
        
        # Check for search results
        search_indicators = ['/search', '/s?', 'searchpage', 'query=', 'q=']
        for indicator in search_indicators:
            if indicator in url.lower():
                issues.append(f"URL appears to be a search results page (contains '{indicator}')")
                break
        
        # Check for product identifiers
        product_indicators = ['/dp/', '/ip/', '/product/', '/p/', 'sku=', 'pid=']
        has_product_id = any(ind in url.lower() for ind in product_indicators)
        
        if not has_product_id and not issues:
            issues.append("URL may not contain a product identifier")
        
        if issues:
            return ToolResult(f"URL validation issues:\n" + "\n".join(f"- {i}" for i in issues))
        else:
            return ToolResult("URL appears valid")


# Server registry
SERVERS = {
    "product-search": ProductSearchServer,
    "cashback": CashbackServer,
    "credit-card": CreditCardServer,
    "verification": VerificationServer,
}


def get_server(name: str) -> MCPServer:
    """Get a server instance by name."""
    if name not in SERVERS:
        raise ValueError(f"Unknown server: {name}. Available: {list(SERVERS.keys())}")
    return SERVERS[name]()


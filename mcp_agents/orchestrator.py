"""
MCP Orchestrator - Coordinates multiple MCP servers for shopping agent.

This orchestrator:
1. Manages multiple MCP servers
2. Routes tool calls to appropriate servers
3. Runs parallel searches across servers
4. Combines and enriches results
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import concurrent.futures

from .servers import (
    MCPServer,
    ProductSearchServer,
    CashbackServer,
    CreditCardServer,
    VerificationServer,
    ToolResult,
    MCPMessage
)


@dataclass
class AgentTask:
    """Task to be executed by an agent."""
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]


class MCPOrchestrator:
    """
    Orchestrates multiple MCP servers for collaborative shopping search.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    ORCHESTRATOR                          │
    │  Coordinates parallel execution & result aggregation     │
    └─────────────────────────────────────────────────────────┘
              │              │              │              │
              ▼              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Product    │ │   Cashback   │ │ Credit Card  │ │ Verification │
    │   Search     │ │   Lookup     │ │   Rewards    │ │    Server    │
    │   Server     │ │   Server     │ │   Server     │ │              │
    └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
    """
    
    def __init__(self):
        # Initialize all servers
        self.servers: Dict[str, MCPServer] = {
            "product": ProductSearchServer(),
            "cashback": CashbackServer(),
            "credit_card": CreditCardServer(),
            "verification": VerificationServer(),
        }
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool on a specific server."""
        if server_name not in self.servers:
            return ToolResult(f"Unknown server: {server_name}", is_error=True)
        
        server = self.servers[server_name]
        return await server.execute_tool(tool_name, arguments)
    
    def list_all_tools(self) -> Dict[str, List[Dict]]:
        """List all tools from all servers."""
        return {
            name: [
                {"name": t.name, "description": t.description}
                for t in server.list_tools()
            ]
            for name, server in self.servers.items()
        }
    
    async def search_product_complete(self, product_query: str) -> Dict[str, Any]:
        """
        Complete product search with all enrichments.
        Runs product search, cashback, and credit card lookups in parallel.
        """
        print(f"\n{'='*60}")
        print(f"🚀 MCP ORCHESTRATOR - Starting parallel agent execution")
        print(f"{'='*60}")
        print(f"Product: {product_query}\n")
        
        # Detect category
        category = self._detect_category(product_query)
        print(f"📂 Detected category: {category}")
        
        # Common retailers for lookups
        retailers = ["Amazon", "Best Buy", "Walmart", "Target", "Costco", "B&H Photo", "Newegg"]
        
        # Run parallel searches
        print(f"\n🔄 Running parallel MCP server calls...")
        print(f"   📦 Product Search Server")
        print(f"   💰 Cashback Server")
        print(f"   💳 Credit Card Server")
        
        # Create tasks for parallel execution
        tasks = [
            self._run_async(
                self.call_tool("product", "search_product", {"product_query": product_query})
            ),
            self._run_async(
                self.call_tool("cashback", "lookup_cashback", {"retailers": retailers, "category": category})
            ),
            self._run_async(
                self.call_tool("credit_card", "recommend_card", {"retailers": retailers})
            ),
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        product_results = results[0]
        cashback_results = results[1]
        credit_card_results = results[2]
        
        print(f"\n✅ All parallel searches complete!")
        
        # Verify results
        print(f"\n🔎 Running verification server...")
        verified_results = await self.call_tool(
            "verification",
            "verify_results",
            {"results": product_results.content, "product_query": product_query}
        )
        
        print(f"✅ Verification complete!")
        
        # Combine results
        print(f"\n✨ Combining all results...")
        final_results = await self._enrich_results(
            verified_results.content,
            cashback_results.content,
            credit_card_results.content,
            product_query
        )
        
        return {
            "product_query": product_query,
            "category": category,
            "product_results": verified_results.content,
            "cashback_data": cashback_results.content,
            "credit_card_data": credit_card_results.content,
            "final_results": final_results,
        }
    
    async def _run_async(self, coro):
        """Helper to run coroutine."""
        return await coro
    
    def _detect_category(self, query: str) -> str:
        """Detect product category."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['phone', 'laptop', 'computer', 'tv', 'headphone', 'airpod', 
                                           'playstation', 'xbox', 'nintendo', 'camera', 'tablet', 'watch']):
            return "Electronics"
        elif any(w in query_lower for w in ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'clothing']):
            return "Clothing"
        elif any(w in query_lower for w in ['sofa', 'chair', 'table', 'bed', 'furniture', 'mattress']):
            return "Home/Furniture"
        elif any(w in query_lower for w in ['makeup', 'skincare', 'beauty', 'cosmetic']):
            return "Beauty"
        else:
            return "General"
    
    async def _enrich_results(self, product_results: str, cashback_data: str, 
                              credit_card_data: str, product_query: str) -> str:
        """Combine all results into final enriched output."""
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = f"""Combine these shopping results for "{product_query}":

=== PRODUCT PRICES ===
{product_results}

=== CASHBACK RATES ===
{cashback_data}

=== CREDIT CARD RECOMMENDATIONS ===
{credit_card_data}

Create a final report with:
1. Each retailer with price, cashback, and recommended credit card
2. 🏆 BEST OVERALL DEAL (considering all factors)
3. 💳 Credit Card Strategy Summary
4. ⚠️ Important notes

Format as a numbered list for each retailer."""
        
        response = llm.invoke([
            {"role": "system", "content": "You are a shopping results aggregator."},
            {"role": "user", "content": prompt}
        ])
        
        return response.content
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)


async def run_mcp_search(product_query: str):
    """Run a complete MCP-based product search."""
    orchestrator = MCPOrchestrator()
    
    try:
        results = await orchestrator.search_product_complete(product_query)
        
        print(f"\n{'='*60}")
        print(f"📊 FINAL MCP SEARCH RESULTS")
        print(f"{'='*60}")
        print(results["final_results"])
        
        return results
    finally:
        orchestrator.close()


def main():
    """Main entry point for MCP-based shopping agent."""
    import sys
    
    print(f"{'='*60}")
    print("🛒 MCP-BASED SHOPPING AGENT")
    print("   Using Model Context Protocol for agent collaboration")
    print(f"{'='*60}\n")
    
    # Show available servers
    orchestrator = MCPOrchestrator()
    print("Available MCP Servers:")
    for name, tools in orchestrator.list_all_tools().items():
        print(f"\n  📡 {name}:")
        for tool in tools:
            print(f"      • {tool['name']}: {tool['description'][:50]}...")
    
    print(f"\n{'-'*60}")
    
    # Get user query
    default_query = "PlayStation 5"
    user_input = input(f"Enter product to search (default: {default_query}): ").strip()
    
    if not user_input:
        user_input = default_query
    
    print(f"\n🔍 Searching for: {user_input}")
    
    # Run search
    asyncio.run(run_mcp_search(user_input))
    
    orchestrator.close()


if __name__ == "__main__":
    main()


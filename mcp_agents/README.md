# MCP Shopping Agent

This module implements the Shopping Agent as an **MCP (Model Context Protocol) server** that can be connected to Claude Desktop or other MCP-compatible clients.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP CLIENT (Claude Desktop)                  │
│                                                                  │
│  "Find me the best price for PlayStation 5 with cashback"       │
└─────────────────────────────────────────────────────────────────┘
                              │ stdio
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP SERVER (shopping-agent)                 │
├─────────────────────────────────────────────────────────────────┤
│  Tools:                                                          │
│  • search_product_prices      - Search retailers for prices      │
│  • lookup_cashback_rates      - Get Rakuten/ShopBack rates      │
│  • get_credit_card_recommendations - Best card for each store   │
│  • verify_product_url         - Validate product URLs           │
│  • complete_shopping_search   - Full search with all data       │
│  • start_price_tracking       - Monitor prices over time        │
│  • get_tracking_statistics    - Get tracking session stats      │
└─────────────────────────────────────────────────────────────────┘
```

## Setup for Claude Desktop

### 1. Locate Claude Desktop Config

**macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```bash
~/.config/claude/claude_desktop_config.json
```

### 2. Add Shopping Agent Server

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "shopping-agent": {
      "command": "python",
      "args": ["-m", "mcp_agents.mcp_server"],
      "cwd": "/home/etem/shopping-agent",
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "TAVILY_API_KEY": "your-key-here"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

After saving the config, restart Claude Desktop to load the new MCP server.

## Available Tools

### `search_product_prices`
Search for product prices across multiple retailers.

**Input:**
```json
{
  "product_query": "PlayStation 5"
}
```

### `lookup_cashback_rates`
Look up cashback rates from shopping portals.

**Input:**
```json
{
  "retailers": ["Amazon", "Best Buy", "Walmart"],
  "category": "Electronics"
}
```

### `get_credit_card_recommendations`
Get best credit card for each retailer.

**Input:**
```json
{
  "retailers": ["Amazon", "Costco", "Best Buy"]
}
```

### `verify_product_url`
Validate if a URL is a proper product page.

**Input:**
```json
{
  "url": "https://amazon.com/dp/B0BCNKKZ91",
  "expected_product": "PlayStation 5"
}
```

### `complete_shopping_search`
Run full search with prices, cashback, and credit card recommendations.

**Input:**
```json
{
  "product_query": "AirPods Pro 3"
}
```

### `start_price_tracking`
Start monitoring prices over time.

**Input:**
```json
{
  "product_query": "PlayStation 5",
  "interval_minutes": 60,
  "duration_hours": 24
}
```

### `get_tracking_statistics`
Get statistics for a tracking session.

**Input:**
```json
{
  "session_id": 1
}
```

## Testing the Server

### Test Standalone

```bash
cd /home/etem/shopping-agent
source venv/bin/activate
python -m mcp_agents.mcp_server
```

Then send JSON-RPC messages via stdin:

```json
{"jsonrpc":"2.0","method":"initialize","id":1}
{"jsonrpc":"2.0","method":"tools/list","id":2}
{"jsonrpc":"2.0","method":"tools/call","id":3,"params":{"name":"get_credit_card_recommendations","arguments":{"retailers":["Amazon","Costco"]}}}
```

### Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python -m mcp_agents.mcp_server
```

## Example Conversation in Claude

Once connected, you can ask Claude:

> "Find me the best price for a PlayStation 5, including cashback and which credit card to use"

Claude will use the shopping agent tools to:
1. Search multiple retailers for prices
2. Look up cashback rates from Rakuten, Capital One Shopping, etc.
3. Recommend the best credit card for each retailer
4. Provide a final recommendation considering all factors


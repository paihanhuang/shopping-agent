# 🛒 Shopping Price Comparison Agent

An AI-powered shopping agent that searches the web for product prices across multiple websites and provides comprehensive price comparisons with cashback rates and credit card recommendations.

Built with **LangChain**, **Tavily Search API**, and **MCP (Model Context Protocol)**.

## Features

- 🔍 Searches across 15+ distinct retailers (Amazon, Best Buy, Walmart, Target, Costco, etc.)
- 💰 Compares prices including tax and shipping costs
- 💵 Finds cashback rates from Rakuten, Capital One Shopping, ShopBack
- 💳 Recommends best credit card for each retailer
- 📍 Calculates estimates for ZIP code 94022 (Los Altos, CA)
- 🤖 **MCP Support** - Connect to Claude Desktop or other MCP clients
- 📈 **Price Tracking** - Monitor prices over time

---

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp env.example .env
```

Edit `.env` with your API keys:
```
OPENAI_API_KEY=your_actual_openai_key
TAVILY_API_KEY=your_actual_tavily_key
```

**Get your API keys:**
- OpenAI API Key: https://platform.openai.com/api-keys
- Tavily API Key: https://tavily.com/

### 3. Run the Agent

```bash
source venv/bin/activate
python main.py
```

---

## Three Ways to Run

| Mode | Command | Use Case |
|------|---------|----------|
| **Standalone** | `python main.py` | Quick terminal searches |
| **MCP Server** | `python -m mcp_agents.mcp_server` | Connect to Claude Desktop |
| **Price Tracker** | `python price_tracker.py` | Monitor prices over time |

---

## 🔌 Running with MCP (Model Context Protocol)

MCP allows you to connect the shopping agent to Claude Desktop or test it via a web UI.

### Option A: Test with MCP Inspector (Web UI)

**Step 1:** Start MCP Inspector
```bash
cd /home/etem/shopping-agent
source venv/bin/activate
DANGEROUSLY_OMIT_AUTH=true npx -y @modelcontextprotocol/inspector ./venv/bin/python -m mcp_agents.mcp_server
```

**Step 2:** Open browser
```
http://localhost:6274
```

**Step 3:** In the web UI:
1. Click **"Tools"** in the sidebar
2. Select `complete_shopping_search`
3. Enter input:
   ```json
   {
     "product_query": "PlayStation 5"
   }
   ```
4. Click **"Run"**

### Option B: Connect to Claude Desktop

**Step 1:** Locate Claude Desktop config file

| OS | Path |
|----|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/claude/claude_desktop_config.json` |

**Step 2:** Add shopping agent configuration

```json
{
  "mcpServers": {
    "shopping-agent": {
      "command": "/home/etem/shopping-agent/venv/bin/python",
      "args": ["-m", "mcp_agents.mcp_server"],
      "cwd": "/home/etem/shopping-agent"
    }
  }
}
```

**Step 3:** Restart Claude Desktop

**Step 4:** In Claude Desktop, ask:
> "Find me the best price for PlayStation 5 with cashback and credit card recommendations"

Claude will automatically use the shopping agent tools!

### Option C: Test MCP Server Directly (Command Line)

```bash
cd /home/etem/shopping-agent
source venv/bin/activate
python -m mcp_agents.mcp_server
```

Then send JSON-RPC messages via stdin:
```json
{"jsonrpc":"2.0","method":"tools/list","id":1}
```

---

## 🛠️ Available MCP Tools

| Tool | Description | Input |
|------|-------------|-------|
| `complete_shopping_search` | Full search with prices, cashback, credit cards | `{"product_query": "..."}` |
| `search_product_prices` | Search retailers for prices only | `{"product_query": "..."}` |
| `lookup_cashback_rates` | Get cashback from Rakuten, Capital One, ShopBack | `{"retailers": [...], "category": "..."}` |
| `get_credit_card_recommendations` | Best card for each retailer | `{"retailers": [...]}` |
| `verify_product_url` | Validate product page URLs | `{"url": "...", "expected_product": "..."}` |
| `start_price_tracking` | Monitor prices over time | `{"product_query": "...", "interval_minutes": 60}` |
| `get_tracking_statistics` | Get stats for tracking session | `{"session_id": 1}` |

---

## Usage Examples

### Standalone Mode
```bash
python main.py
# Enter: "Find me the best price for AirPods Pro"
```

### MCP Mode (via Inspector)
```json
{
  "product_query": "Sony WH-1000XM5 headphones"
}
```

### Price Tracking
```bash
python price_tracker.py
# Choose option 1 to start tracking
# Enter product and duration
```

---

## Example Output

```
📊 FINAL PRICE COMPARISON RESULTS
============================================================

1. **Amazon**
   - URL: https://amazon.com/dp/...
   - Base Price: $349.99
   - Tax (9.25%): $32.37
   - Shipping: Free
   - 💰 Cashback:
     * Rakuten: 2% (~$6.99 back)
     * Capital One Shopping: 1%
   - 💳 Best Credit Card: Amazon Prime Visa - 5% back (~$17.50)
   - **TOTAL: $382.36**
   - **Potential Savings: ~$24.49**

2. **Best Buy**
   ...

🏆 BEST OVERALL DEAL: Amazon with ~$24.49 savings
💳 Credit Card Strategy:
   * Amazon: Use Amazon Prime Visa for 5%
   * Costco: Use Costco Anywhere Visa (only Visa accepted)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP CLIENT                               │
│            (Claude Desktop / MCP Inspector)                  │
└─────────────────────────────────────────────────────────────┘
                              │ JSON-RPC over stdio
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP SERVER (mcp_server.py)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│         (Parallel execution of search agents)                │
├──────────────────┬──────────────────┬───────────────────────┤
│ Product Search   │  Cashback Lookup │  Credit Card Lookup   │
└──────────────────┴──────────────────┴───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LangChain + Tavily + OpenAI GPT-4o-mini        │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
shopping-agent/
├── main.py                 # Standalone shopping agent
├── price_tracker.py        # Price monitoring agent
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not in git)
├── env.example             # API keys template
├── mcp_config.json         # MCP configuration
└── mcp_agents/
    ├── __init__.py
    ├── mcp_server.py       # MCP server (stdio transport)
    ├── orchestrator.py     # Coordinates parallel agents
    ├── servers.py          # Individual tool implementations
    └── README.md           # MCP-specific documentation
```

---

## Troubleshooting

### "Port is in use" error
```bash
pkill -f "mcp_agents.mcp_server"
pkill -f "@modelcontextprotocol/inspector"
```

### OAuth/Authentication error in MCP Inspector
Use incognito browser window or add `DANGEROUSLY_OMIT_AUTH=true`:
```bash
DANGEROUSLY_OMIT_AUTH=true npx @modelcontextprotocol/inspector ...
```

### No results found
- Check your API keys in `.env`
- Ensure you have API credits
- Try a simpler product query

---

## Notes

- Tax estimates are based on California's sales tax rate (9.25%) for ZIP 94022
- Cashback rates are looked up in real-time and may vary
- Costco only accepts Visa credit cards
- URLs marked with ⚠️ should be verified before purchase

---

## License

MIT

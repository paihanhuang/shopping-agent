# 🛒 Shopping Price Comparison Agent

An AI-powered shopping agent that searches the web for product prices across multiple websites and provides comprehensive price comparisons with recommendations.

Built with **LangChain** and **Tavily Search API**.

## Features

- 🔍 Searches across 15+ distinct websites/retailers
- 💰 Compares prices including tax and shipping costs
- 📍 Calculates estimates for ZIP code 94022 (Los Altos, CA)
- 💡 Provides recommendations on the best place to buy

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp env.example .env
```

Then edit `.env` with your actual API keys:

```
OPENAI_API_KEY=your_actual_openai_key
TAVILY_API_KEY=your_actual_tavily_key
```

**Get your API keys:**
- OpenAI API Key: https://platform.openai.com/api-keys
- Tavily API Key: https://tavily.com/

### 3. Run the Agent

```bash
python main.py
```

## Usage

When you run the agent, you'll be prompted to enter a search query. For example:

- "Find me the best price for PlayStation 5"
- "Compare prices for iPhone 15 Pro Max 256GB"
- "Best deals on Samsung 65 inch OLED TV"

The agent will:
1. Search multiple websites using Tavily Search API
2. Gather pricing information from 15+ retailers
3. Calculate total costs including estimated tax (9.25%) and shipping
4. Present a comparison table
5. Recommend the best option based on price and reliability

## Example Output

```
📊 PRICE COMPARISON RESULTS
================================================

| Website      | URL                  | Base Price | Tax    | Shipping | Total   |
|--------------|----------------------|------------|--------|----------|---------|
| Amazon       | amazon.com/...       | $449.99    | $41.62 | FREE     | $491.61 |
| Walmart      | walmart.com/...      | $449.99    | $41.62 | FREE     | $491.61 |
| Best Buy     | bestbuy.com/...      | $449.99    | $41.62 | FREE     | $491.61 |
| ...          | ...                  | ...        | ...    | ...      | ...     |

🏆 RECOMMENDATION:
Based on the search results, I recommend purchasing from [Website] because...
```

## Architecture

- **LLM**: ChatOpenAI (GPT-4o) for intelligent reasoning and comparison
- **Search Tool**: TavilySearchResults for comprehensive web searching
- **Agent Framework**: LangChain's tool-calling agent with AgentExecutor

## Notes

- The agent performs multiple search queries to ensure comprehensive coverage
- Tax estimates are based on California's sales tax rate for ZIP 94022
- Shipping costs are estimated when exact information is not available
- Results may vary based on current availability and pricing



"""
Shopping Agent - AI-powered price comparison tool using LangChain and Tavily Search API.

This agent searches the web for product prices across multiple websites and provides
a comprehensive comparison with recommendations.
"""

import os
import sys
import concurrent.futures
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables
load_dotenv()

# Validate API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found. Please set it in your .env file.")


# Simplified search prompt (cashback and credit cards handled separately)
SEARCH_PROMPT = """You are an expert shopping assistant. Search for product prices only.

SEARCH INSTRUCTIONS:
1. Search at least 20 different retailers:
   - "[product] site:amazon.com", "[product] site:bestbuy.com", etc.
   - "[product] price", "[product] best deals"

2. For each retailer found, report:
   - Retailer Name
   - Product URL (direct product page, not homepage)
   - Base Price
   - Tax (9.25% for ZIP 94022)
   - Shipping cost
   - Total Price (before cashback)

3. ONLY include actual retailers:
   ✅ Amazon, Best Buy, Walmart, Target, Costco, B&H, Adorama, Newegg, Apple, Samsung
   ❌ EXCLUDE: news sites, deal aggregators, shopping portals, price comparison sites

4. Format as numbered list. Include summary at end.

Destination: ZIP 94022 (Los Altos, California).
"""


# Cashback lookup prompt
CASHBACK_PROMPT = """You are a cashback research specialist. Find current cashback rates from shopping portals.

SEARCH INSTRUCTIONS:
1. Search for cashback rates for these retailers:
   Amazon, Best Buy, Walmart, Target, Costco, B&H Photo, Adorama, Newegg, Apple, Samsung

2. Search portals:
   - Rakuten: "Rakuten [retailer] cashback"
   - Capital One Shopping: "Capital One Shopping [retailer]"
   - ShopBack: "ShopBack [retailer] cashback"

3. CATEGORY-SPECIFIC rates:
   - Electronics: usually 1-2%
   - Clothing: usually 3-8%
   - Home/Furniture: usually 2-5%

4. KNOWN EXCLUSIONS:
   - Costco: No cashback on any portal
   - Apple Store: Very limited/no cashback

5. Output format:
   [Retailer]: Rakuten X%, Capital One Shopping X%, ShopBack X%
   If no cashback: "[Retailer]: No cashback"

Only report rates ACTUALLY FOUND. Do NOT guess.
"""


# Credit card rewards lookup prompt
CREDIT_CARD_PROMPT = """You are a credit card rewards specialist. Find the best credit card reward rates for shopping.

SEARCH INSTRUCTIONS:
1. Search for current reward rates for these credit cards:
   - "Citi Double Cash rewards rate"
   - "Chase Sapphire Reserve rewards categories"
   - "Capital One Venture X rewards"
   - "Bank of America Customized Cash rewards categories"
   - "Amazon Prime Visa rewards"
   - "Costco Anywhere Visa rewards"

2. For each card, find:
   - Base reward rate (on general purchases)
   - Bonus categories and rates
   - Any special retailer partnerships

3. KNOWN RATES (verify if still current):
   - Citi Double Cash: 2% on everything (1% buy + 1% pay)
   - Chase Sapphire Reserve: 3x travel/dining, 1x other
   - Capital One Venture X: 2x on everything
   - BofA Customized Cash: 3% on chosen category, 2% grocery/wholesale, 1% other
   - Amazon Prime Visa: 5% at Amazon/Whole Foods, 2% restaurants/gas/drugstores
   - Costco Anywhere Visa: 4% gas, 3% restaurants/travel, 2% Costco, 1% other

4. RETAILER-SPECIFIC recommendations:
   - Amazon: Amazon Prime Visa (5%) or BofA Customized Cash (3% online)
   - Costco: Costco Anywhere Visa (2%) - NOTE: Costco only accepts Visa
   - Best Buy: BofA Customized Cash (3% online) or Citi Double Cash (2%)
   - General online: BofA Customized Cash (3% if set to Online Shopping)

5. Output format:
   [Card Name]: Base rate X%, Bonus categories: [list]
   
   Retailer recommendations:
   [Retailer]: Best card [Card] at X%
"""


# Verification prompt
VERIFICATION_PROMPT = """You are a data verification assistant. Review and correct the shopping results.

VERIFICATION TASKS:
1. REMOVE DUPLICATES: Keep only one entry per retailer
2. REMOVE NON-RETAILERS: news sites, deal aggregators, shopping portals
3. VERIFY URLs: Must be product pages, not homepages or search results
4. VERIFY PRODUCT: URL must match exact product searched

Return CORRECTED list only.
"""


# Enrichment prompt
ENRICH_PROMPT = """You are a data enrichment assistant. Combine product prices with cashback and credit card data.

INSTRUCTIONS:
1. Take the verified product results
2. Add cashback rates from the cashback data for each retailer
3. Add best credit card recommendation from the credit card data for each retailer
4. Calculate potential savings

OUTPUT FORMAT for each retailer:
1. **[Retailer Name]**
   - URL: [product URL]
   - Base Price: $XX.XX
   - Tax (9.25%): $XX.XX
   - Shipping: Free / $XX.XX
   - 💰 Cashback:
     * Rakuten: X% (~$X.XX back)
     * Capital One Shopping: X%
     * ShopBack: X%
   - 💳 Best Credit Card: [Card Name] - X% back (~$X.XX)
   - **TOTAL: $XX.XX** (before cashback/rewards)
   - **Potential Savings: ~$X.XX** (best cashback + credit card)

End with:
- 🏆 BEST OVERALL DEAL (price + cashback + credit card combined)
- 💳 Credit Card Strategy:
  * [Retailer]: Use [Card] for X%
  * Note: Costco only accepts Visa
- ⚠️ Verify current rates before purchase
"""


class ProgressCallback(BaseCallbackHandler):
    """Callback handler to show progress during agent execution."""
    
    def __init__(self, prefix=""):
        self.search_count = 0
        self.prefix = prefix
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.search_count += 1
        if isinstance(input_str, dict):
            query = input_str.get('query', input_str.get('search_query', str(input_str)))
        else:
            query = str(input_str)
        query_display = query[:50] + "..." if len(query) > 50 else query
        print(f"{self.prefix}🔍 #{self.search_count}: \"{query_display}\"")
        sys.stdout.flush()
    
    def on_tool_end(self, output, **kwargs):
        sys.stdout.flush()


def create_agent_with_search(system_prompt: str, callbacks=None):
    """Create an agent with Tavily search capability."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=callbacks)
    tavily_search = TavilySearch(
        max_results=10,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )
    return create_agent(llm, [tavily_search], system_prompt=system_prompt)


def search_products(query: str) -> str:
    """Search for product prices (without cashback/credit card info)."""
    print("\n📦 [Product Search] Starting...\n")
    callback = ProgressCallback(prefix="  ")
    agent = create_agent_with_search(SEARCH_PROMPT, callbacks=[callback])
    
    enhanced_query = f"""
{query}

Search for prices from at least 15 different retailers.
Format as numbered list with: Retailer, URL, Price, Tax, Shipping, Total.
"""
    
    result = agent.invoke(
        {"messages": [("human", enhanced_query)]},
        config={"callbacks": [callback]}
    )
    
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue
            if msg.content and len(msg.content) > 100:
                print("  ✅ Product search complete!")
                return msg.content
    return ""


def lookup_cashback_rates(product_category: str) -> str:
    """Look up cashback rates from shopping portals (runs in parallel)."""
    print("\n💰 [Cashback Lookup] Starting...\n")
    callback = ProgressCallback(prefix="  ")
    agent = create_agent_with_search(CASHBACK_PROMPT, callbacks=[callback])
    
    cashback_query = f"""
Find current cashback rates for "{product_category}" category.

Search for rates at: Amazon, Best Buy, Walmart, Target, Costco, B&H Photo, Adorama, Newegg, Apple, Samsung

Portals to check: Rakuten, Capital One Shopping, ShopBack

Remember: Costco = No cashback, Apple = Very limited
"""
    
    result = agent.invoke(
        {"messages": [("human", cashback_query)]},
        config={"callbacks": [callback]}
    )
    
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue
            if msg.content and len(msg.content) > 50:
                print("  ✅ Cashback lookup complete!")
                return msg.content
    return ""


def lookup_credit_card_rewards(retailers: list) -> str:
    """
    Look up credit card reward rates (runs in parallel).
    
    Args:
        retailers: List of retailer names to get recommendations for
    
    Returns:
        Credit card reward rates and recommendations
    """
    print("\n💳 [Credit Card Rewards] Starting...\n")
    callback = ProgressCallback(prefix="  ")
    agent = create_agent_with_search(CREDIT_CARD_PROMPT, callbacks=[callback])
    
    retailers_str = ", ".join(retailers) if retailers else "Amazon, Best Buy, Walmart, Target, Costco, B&H Photo, Newegg"
    
    credit_card_query = f"""
Find the best credit card reward rates for shopping at these retailers:
{retailers_str}

Cards to research:
1. Citi Double Cash
2. Chase Sapphire Reserve
3. Capital One Venture X
4. Bank of America Customized Cash
5. Amazon Prime Visa (for Amazon)
6. Costco Anywhere Visa (for Costco - only Visa accepted)

For each retailer, recommend the best card to use and why.
Note any restrictions (e.g., Costco only accepts Visa).
"""
    
    result = agent.invoke(
        {"messages": [("human", credit_card_query)]},
        config={"callbacks": [callback]}
    )
    
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue
            if msg.content and len(msg.content) > 50:
                print("  ✅ Credit card lookup complete!")
                return msg.content
    return ""


def verify_results(raw_results: str, product_query: str) -> str:
    """Verify and correct the search results."""
    print("\n🔎 [Verification] Checking results...")
    sys.stdout.flush()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    verification_query = f"""
Original search: "{product_query}"

Verify these results:
{raw_results}

Remove duplicates, non-retailers, invalid URLs. Check product matches query.
"""
    
    response = llm.invoke([
        {"role": "system", "content": VERIFICATION_PROMPT},
        {"role": "user", "content": verification_query}
    ])
    
    print("  ✅ Verification complete!")
    return response.content


def enrich_with_rewards(verified_results: str, cashback_data: str, credit_card_data: str, product_query: str) -> str:
    """
    Enrich verified results with cashback AND credit card recommendations.
    """
    print("\n✨ [Enrichment] Combining all data...")
    sys.stdout.flush()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    enrich_query = f"""
Product searched: "{product_query}"

=== PRODUCT SEARCH RESULTS ===
{verified_results}

=== CASHBACK DATA ===
{cashback_data}

=== CREDIT CARD REWARDS DATA ===
{credit_card_data}

Please combine all this data:
1. For each retailer, add:
   - Cashback rates from each portal
   - Best credit card to use and its reward rate
   - Potential savings (cashback + credit card rewards)
2. Calculate which combination gives the best total value
3. End with BEST OVERALL DEAL and Credit Card Strategy

Remember: Costco only accepts Visa (recommend Costco Visa or Citi Double Cash)
"""
    
    response = llm.invoke([
        {"role": "system", "content": ENRICH_PROMPT},
        {"role": "user", "content": enrich_query}
    ])
    
    print("  ✅ Enrichment complete!")
    return response.content


def detect_product_category(query: str) -> str:
    """Detect the product category from the query."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['phone', 'laptop', 'computer', 'tv', 'headphone', 'airpod', 'playstation', 'xbox', 'nintendo', 'camera', 'tablet', 'ipad', 'watch', 'speaker', 'monitor']):
        return "Electronics"
    elif any(word in query_lower for word in ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'clothing', 'apparel', 'sneaker']):
        return "Clothing"
    elif any(word in query_lower for word in ['sofa', 'chair', 'table', 'bed', 'furniture', 'mattress', 'desk']):
        return "Home/Furniture"
    elif any(word in query_lower for word in ['makeup', 'skincare', 'beauty', 'cosmetic', 'perfume']):
        return "Beauty"
    else:
        return "General"


def search_product_prices(query: str) -> str:
    """
    Search for product prices with parallel lookups.
    
    Architecture:
    1. Product Search + Cashback Lookup + Credit Card Lookup (ALL PARALLEL)
    2. Verification
    3. Enrichment with cashback + credit card data
    """
    # Detect product category
    category = detect_product_category(query)
    print(f"📂 Detected category: {category}")
    
    # Common retailers for credit card lookup
    common_retailers = ["Amazon", "Best Buy", "Walmart", "Target", "Costco", "B&H Photo", "Newegg", "Apple"]
    
    # Run ALL THREE lookups in PARALLEL
    print("\n" + "=" * 55)
    print("🚀 Running 3 parallel searches...")
    print("   📦 Products | 💰 Cashback | 💳 Credit Cards")
    print("=" * 55)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all three tasks
        product_future = executor.submit(search_products, query)
        cashback_future = executor.submit(lookup_cashback_rates, category)
        credit_card_future = executor.submit(lookup_credit_card_rewards, common_retailers)
        
        # Wait for all to complete
        product_results = product_future.result()
        cashback_data = cashback_future.result()
        credit_card_data = credit_card_future.result()
    
    if not product_results:
        return "No results found. Please try again."
    
    # Step 2: Verify results
    print("\n" + "-" * 55)
    verified_results = verify_results(product_results, query)
    
    # Step 3: Enrich with cashback AND credit card data
    print("\n" + "-" * 55)
    final_results = enrich_with_rewards(verified_results, cashback_data, credit_card_data, query)
    
    return final_results


def main():
    """Main entry point for the shopping agent."""
    print("=" * 60)
    print("🛒 Shopping Price Comparison Agent")
    print("=" * 60)
    print("\nThis agent runs 3 parallel searches:")
    print("  📦 Product prices from 15+ retailers")
    print("  💰 Cashback rates (Rakuten, Capital One, ShopBack)")
    print("  💳 Credit card rewards (best card for each store)\n")
    
    default_query = "Find me the best price for PlayStation 5"
    user_input = input(f"Enter your search query (or press Enter for default: '{default_query}'): ").strip()
    
    if not user_input:
        user_input = default_query
    
    print(f"\n🔍 Searching for: {user_input}")
    print("⏳ Running parallel searches...\n")
    
    try:
        result = search_product_prices(user_input)
        print("\n" + "=" * 60)
        print("📊 FINAL PRICE COMPARISON RESULTS")
        print("=" * 60)
        print(result)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nPlease check:")
        print("1. Your API keys are correctly set in the .env file")
        print("2. You have sufficient API credits")
        print("3. Your internet connection is stable")


if __name__ == "__main__":
    main()

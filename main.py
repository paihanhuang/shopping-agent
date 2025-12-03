"""
Shopping Agent - AI-powered price comparison tool using LangChain and Tavily Search API.

This agent searches the web for product prices across multiple websites and provides
a comprehensive comparison with recommendations.
"""

import os
import sys
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


# System prompt for the shopping agent
SYSTEM_PROMPT = """You are an expert shopping assistant specialized in finding the best deals online.

Your task is to search for product prices across the web and provide comprehensive price comparisons, INCLUDING cashback offers.

IMPORTANT INSTRUCTIONS:
1. Perform multiple searches to gather information from at least 15 DISTINCT websites/retailers.
2. Search strategies to use:
   - Search for "[product] price" 
   - Search for "[product] buy online"
   - Search for "[product] best deals"
   - Search for "[product] site:amazon.com" (to find exact product page on Amazon)
   - Search for "[product] site:bestbuy.com" (to find exact product page on Best Buy)
   - Search for "[product] site:walmart.com" (repeat for each major retailer)
   - Search for "[product] [major retailer name]" (e.g., Amazon, Walmart, Best Buy, Target, etc.)
   - Search for "[product] discount" or "[product] sale"
   - Search for "Rakuten [retailer name] cashback rate [product category] 2025"
   - Search for "Capital One Shopping [retailer name] cashback"
   - Search for "ShopBack [retailer name] [product category] cashback"

3. CRITICAL - CASHBACK CATEGORY RULES:
   Cashback rates VARY BY PRODUCT CATEGORY within each retailer. You MUST:
   - Search for the SPECIFIC category rate, not just the general retailer rate
   - Common category differences:
     * Electronics/Tech: Usually LOWER rates (often 1-2%)
     * Clothing/Apparel: Usually HIGHER rates (often 3-8%)
     * Home/Furniture: Medium rates (often 2-5%)
     * Beauty/Health: Often higher rates (3-10%)
   - Example: Rakuten at Best Buy might be 1% for electronics but higher for other categories
   - If you only find a general rate, note it as "General rate - may vary by category"

4. CRITICAL - CASHBACK ACCURACY RULES:
   - ONLY report cashback rates that you ACTUALLY FOUND in search results
   - Match the rate to the PRODUCT CATEGORY being searched
   - If no cashback info was found for a retailer, write "No cashback"
   - KNOWN EXCLUSIONS (NO cashback available):
     * Costco - No cashback on any portal
     * T-Mobile - No Rakuten cashback
     * Apple Store direct - Very limited/no cashback
     * Gift cards, warranties - Usually excluded from cashback
   - DO NOT make up or guess cashback percentages
   - When uncertain, say "Verify on portal" instead of guessing

5. CRITICAL - PRODUCT URL REQUIREMENTS:
   For EVERY retailer, you MUST provide a DIRECT product page URL.
   - Search "[product] site:[retailer].com" to find the exact product page
   - The URL must contain product identifiers (SKU, product ID, or product name in path)
   
   ✅ VALID URLs (contain product info):
     * https://www.amazon.com/dp/B0FQFB8FMG
     * https://www.bestbuy.com/site/apple-airpods-pro-3/6535178.p
     * https://www.bhphotovideo.com/c/product/1234567-REG/apple_airpods.html
     * https://www.walmart.com/ip/Apple-AirPods-Pro-3/123456789
   
   ❌ INVALID URLs (do NOT use these):
     * https://www.bhphotovideo.com (homepage)
     * https://www.target.com (homepage)
     * https://www.newegg.com (homepage)
     * Any URL without product ID or specific product path
   
   If you CANNOT find the direct URL after searching, write:
   "🔍 Direct link not found - Search '[product name]' on [retailer].com"
   
6. For each website found, extract and report in a NUMBERED LIST format (NOT a table):
   - Website/Retailer Name (bold)
   - Product URL (must be direct product link - see rules above)
   - Base Price
   - Estimated Tax (9.25% for ZIP 94022 - Los Altos, CA)
   - Estimated Shipping Cost (note if free shipping is available)
   - Cashback/Rewards (category-specific rates):
     * Rakuten: X% for [category] or "No cashback"
     * Capital One Shopping: X% or "No cashback"  
     * ShopBack: X% or "No cashback"
   - 💳 Best Credit Card: Recommend the best card based on retailer category:
     * BofA Customized Cash: 3% if category matches (Online Shopping, Gas, Dining, Travel, Drug Stores, Home Improvement), 2% grocery/wholesale
     * Citi Double Cash: 2% flat cashback on everything
     * Capital One Venture X: 2x miles on everything (~2% value)
     * Chase Sapphire Reserve: 3x on travel/dining, 10x on hotels via portal, 1x other (~1.5x value with travel redemption)
   - **TOTAL PRICE** (before cashback/rewards)

7. NEVER use markdown tables. Always use numbered lists with bullet points for each detail.

8. ONLY include retailers where you found a VALID direct product URL. Skip retailers where you only have homepage links.

9. ONLY INCLUDE ACTUAL RETAILERS - EXCLUDE THESE:
   ❌ Shopping portals/cashback sites: Rakuten, ShopBack, Capital One Shopping, Honey, RetailMeNot
   ❌ Price comparison sites: Google Shopping, PriceGrabber, Shopzilla, NexTag
   ❌ News/review sites: CNET, TechRadar, Engadget, The Verge, Tom's Guide, PCMag, IGN
   ❌ Deal aggregators: Slickdeals, DealNews, Brad's Deals
   ❌ Social/forum sites: Reddit, Twitter, Facebook
   
   ✅ ONLY include actual retailers where you can BUY the product:
   Amazon, Best Buy, Walmart, Target, Costco, B&H Photo, Adorama, Newegg, 
   Apple Store, Samsung, official brand stores, etc.

10. BEFORE FINAL OUTPUT - URL VERIFICATION STEP:
   Review ALL URLs in your results and verify EACH one carefully:
   
   A) NO DUPLICATE RETAILERS:
      - Each retailer should appear ONLY ONCE
      - If you have multiple entries for same retailer, keep only the best deal
   
   B) Check URL structure - must be PRODUCT PAGE, not search results:
      ✅ Valid: https://www.amazon.com/dp/B0FQFB8FMG (product page)
      ❌ Invalid: https://www.amazon.com/s?k=airpods+pro+3 (search results page)
      ❌ Invalid: https://www.bestbuy.com/site/searchpage.jsp?... (search page)
      ❌ Invalid: Any URL containing "/search", "/s?", "searchpage", "query="
   
   C) CRITICAL - Check URL points to CORRECT PRODUCT:
      - Read the product name/model in the URL path
      - Does it match the EXACT product being searched?
      - Watch for WRONG products:
        * Wrong model (e.g., "AirPods Pro 2" instead of "AirPods Pro 3")
        * Wrong version (e.g., "PlayStation 4" instead of "PlayStation 5")
        * Wrong variant (e.g., "128GB" instead of "256GB")
      - If URL points to WRONG product → search again or mark as invalid
   
   D) For ANY invalid URL:
      * Search again: "[exact product name and model] site:[retailer].com"
      * Replace with correct URL
      * Or change to: "🔍 Search '[exact product]' on [retailer].com"
      * Remove retailer if correct product URL cannot be found

11. After listing all retailers, provide:
   - A summary of the best deals found
   - A clear recommendation on which website offers the best value
   - ⚠️ Note: Cashback rates are category-specific and change frequently - always verify on the portal
   - Tips on how to stack discounts when available

12. Always prioritize ACCURACY over completeness. It's better to say "unknown" than to guess.

Remember: The user's shipping destination is ZIP code 94022 (Los Altos, California).
"""


class ProgressCallback(BaseCallbackHandler):
    """Callback handler to show progress during agent execution."""
    
    def __init__(self):
        self.search_count = 0
        self.current_query = ""
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts running."""
        self.search_count += 1
        # Extract query from input
        if isinstance(input_str, dict):
            query = input_str.get('query', input_str.get('search_query', str(input_str)))
        else:
            query = str(input_str)
        self.current_query = query[:60] + "..." if len(query) > 60 else query
        print(f"🔍 Search #{self.search_count}: \"{self.current_query}\"")
        sys.stdout.flush()
    
    def on_tool_end(self, output, **kwargs):
        """Called when a tool finishes."""
        # Count results in the output
        output_str = str(output)
        url_count = output_str.count("http")
        # print(f"   ✓ Found ~{url_count} results")
        sys.stdout.flush()


def create_shopping_agent(callbacks=None):
    """Create and configure the shopping comparison agent."""
    
    # Initialize the LLM with system prompt
    # Using gpt-4o-mini for better rate limits and cost efficiency
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=callbacks,
    )
    
    # Initialize Tavily Search with optimized settings
    # Note: include_raw_content=False to avoid token limits
    tavily_search = TavilySearch(
        max_results=10,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )
    
    tools = [tavily_search]
    
    # Create the agent using LangChain's create_agent
    agent = create_agent(
        llm,
        tools,
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent


def search_product_prices(query: str) -> str:
    """
    Search for product prices across multiple websites.
    
    Args:
        query: The product search query (e.g., "Find me the best price for PlayStation 5")
    
    Returns:
        A comprehensive price comparison report.
    """
    # Create progress callback
    progress_callback = ProgressCallback()
    
    agent = create_shopping_agent(callbacks=[progress_callback])
    
    # Enhance the query with specific instructions
    enhanced_query = f"""
{query}

Please search extensively and find prices from at least 15 different websites/retailers.

IMPORTANT - Determine the PRODUCT CATEGORY first (e.g., Electronics, Clothing, Home, Beauty, etc.)
Then search for CATEGORY-SPECIFIC cashback rates from Rakuten, Capital One Shopping, and ShopBack.
Example: "Rakuten Best Buy electronics cashback rate 2025"

CRITICAL CASHBACK RULES:
- Cashback rates VARY BY CATEGORY (e.g., electronics often 1-2%, clothing 3-8%)
- ONLY report cashback you actually found in search results for THIS CATEGORY
- Say "No cashback" if no partnership exists (e.g., Costco, T-Mobile)
- Say "Verify on portal" if uncertain about the category rate
- DO NOT guess or use general rates when category rates differ

Format your response as a NUMBERED LIST (not a table):

1. **[Retailer Name]**
   - URL: [DIRECT product page link - NOT homepage]
     * If not found, write: "Search [product] on [retailer].com"
   - Base Price: $XX.XX
   - Tax (9.25%): $XX.XX
   - Shipping: Free / $XX.XX
   - 💰 Cashback (for [category]):
     * Rakuten: X% or "No cashback"
     * Capital One Shopping: X% or "No cashback"
     * ShopBack: X% or "No cashback"
   - 💳 Best Credit Card: [Card Name] - [rate/reason]
     * For online shopping: BofA Customized Cash (3% if set to Online Shopping)
     * For general purchases: Citi Double Cash (2%) or Venture X (2x)
     * For travel/dining retailers: Chase Sapphire Reserve (3x)
     * For wholesale clubs (Costco): Citi Double Cash (2%) - Costco only takes Visa
   - **TOTAL: $XX.XX**

2. **[Next Retailer]**
   ... and so on

⚠️ BEFORE OUTPUTTING - FINAL VERIFICATION:
1. ONLY ACTUAL RETAILERS - Remove any:
   - Shopping portals (Rakuten, ShopBack, Honey)
   - News/review sites (CNET, TechRadar, Engadget, IGN)
   - Price comparison sites (Google Shopping)
   - Deal aggregators (Slickdeals)
2. NO DUPLICATES - Each retailer appears only once
3. URLs must be PRODUCT PAGES, not search results:
   - ❌ Remove URLs with "/search", "/s?", "searchpage", "query="
   - ✅ Keep URLs with product ID like "/dp/", "/ip/", "/product/"
4. VERIFY CORRECT PRODUCT - URL matches exact product searched
5. Remove retailers with invalid URLs

After listing all retailers with VERIFIED URLs, provide:
- 🏆 BEST OVERALL DEAL (considering price + cashback + credit card rewards)
- Your recommendation for the best place to buy
- 💳 Credit Card Strategy Summary:
  * Which card to use for best rewards at each store type
  * Note any store-specific restrictions (e.g., Costco = Visa only)
- ⚠️ Note: Rates shown are for [category] - verify current rates before purchase
- Tips on stacking: Portal cashback + Credit card rewards + Store promotions
"""
    
    print("\n📡 Starting search process...\n")
    
    # Use invoke (not stream) and pass the callback for progress updates
    result = agent.invoke(
        {"messages": [("human", enhanced_query)]},
        config={"callbacks": [progress_callback]}
    )
    
    print("\n✅ Analysis complete!\n")
    
    # Extract the final response from the messages
    messages = result.get("messages", [])
    if messages:
        # Get the last message with content
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                # Skip tool messages
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    continue
                # Return the content if it looks like a final response
                content = msg.content
                if content and len(content) > 100:  # Final response should be substantial
                    return content
    
    return "No results found. Please try again."


def main():
    """Main entry point for the shopping agent."""
    print("=" * 60)
    print("🛒 Shopping Price Comparison Agent")
    print("=" * 60)
    print("\nThis agent will search the web and compare prices from")
    print("at least 15 different websites for your product.\n")
    
    # Get user input
    default_query = "Find me the best price for PlayStation 5"
    user_input = input(f"Enter your search query (or press Enter for default: '{default_query}'): ").strip()
    
    if not user_input:
        user_input = default_query
    
    print(f"\n🔍 Searching for: {user_input}")
    print("⏳ This may take a minute as we search multiple websites...\n")
    print("-" * 60)
    
    try:
        result = search_product_prices(user_input)
        print("\n" + "=" * 60)
        print("📊 PRICE COMPARISON RESULTS")
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

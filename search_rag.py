"""
Search RAG Module - Retrieval-Augmented Generation for retailer information.

This module uses a local knowledge base and vector embeddings to retrieve
retailer information for product searches.
"""

import os
import json
from pathlib import Path
from typing import Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class SearchRAG:
    """RAG system for retailer information and search configuration."""
    
    COLLECTION_NAME = "retailer_info"
    PERSIST_DIR = Path(__file__).parent / "search_data" / "chroma_db"
    KNOWLEDGE_BASE_PATH = Path(__file__).parent / "search_data" / "retailers_knowledge_base.json"
    
    def __init__(self):
        """Initialize the RAG system."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore: Optional[Chroma] = None
        self.knowledge_base: dict = {}
        
        # Load knowledge base
        self._load_knowledge_base()
        
        # Initialize or load vector store
        self._init_vectorstore()
    
    def _load_knowledge_base(self):
        """Load the JSON knowledge base."""
        if self.KNOWLEDGE_BASE_PATH.exists():
            with open(self.KNOWLEDGE_BASE_PATH, 'r') as f:
                self.knowledge_base = json.load(f)
            print(f"  📚 Loaded retailer knowledge base (updated: {self.knowledge_base.get('last_updated', 'unknown')})")
        else:
            print(f"  ⚠️ Retailer knowledge base not found at {self.KNOWLEDGE_BASE_PATH}")
            self.knowledge_base = {}
    
    def _create_documents(self) -> list[Document]:
        """Convert knowledge base to LangChain documents for embedding."""
        documents = []
        
        retailers = self.knowledge_base.get("retailers", {})
        
        for retailer_id, retailer_data in retailers.items():
            name = retailer_data.get("name", retailer_id)
            domain = retailer_data.get("domain", "")
            search_url = retailer_data.get("search_url_pattern", "")
            categories = retailer_data.get("categories", [])
            shipping = retailer_data.get("shipping", {})
            tax = retailer_data.get("tax", {})
            notes = retailer_data.get("notes", "")
            best_for = retailer_data.get("best_for", [])
            
            # Create comprehensive document for this retailer
            content = f"""
Retailer: {name}
Domain: {domain}
Search URL Pattern: {search_url}
Categories: {', '.join(categories)}
Best For: {', '.join(best_for)}
Shipping: Free threshold ${shipping.get('free_threshold', 'varies')}, Standard cost ${shipping.get('standard_cost', 'varies')}
Tax Rate (94022): {tax.get('rate_94022', 9.25)}%
Notes: {notes}
"""
            
            # Add membership info if applicable
            if retailer_data.get("membership_required"):
                content += f"\nMembership Required: Yes, ${retailer_data.get('membership_cost', 0)}/year"
            
            # Add payment restrictions
            if retailer_data.get("payment_restriction"):
                content += f"\nPayment Restriction: {retailer_data.get('payment_restriction')}"
            
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "retailer_id": retailer_id,
                    "name": name,
                    "domain": domain,
                    "search_url_pattern": search_url,
                    "categories": ",".join(categories),
                    "source": "retailers_knowledge_base"
                }
            )
            documents.append(doc)
        
        # Add category-retailer mapping documents
        for category, retailer_list in self.knowledge_base.get("category_retailers", {}).items():
            retailer_names = []
            for r_id in retailer_list:
                if r_id in retailers:
                    retailer_names.append(retailers[r_id].get("name", r_id))
            
            content = f"""
Product Category: {category}
Recommended Retailers: {', '.join(retailer_names)}
Number of Retailers: {len(retailer_names)}
Search Tip: {self.knowledge_base.get('search_tips', {}).get(category, 'Use specific product name')}
"""
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "category": category,
                    "retailer_ids": ",".join(retailer_list),
                    "source": "category_mapping"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _init_vectorstore(self):
        """Initialize or load the vector store."""
        if self.PERSIST_DIR.exists() and any(self.PERSIST_DIR.iterdir()):
            print(f"  📂 Loading existing retailer vector store")
            self.vectorstore = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.PERSIST_DIR)
            )
        else:
            print("  🔨 Building new retailer vector store...")
            self._build_vectorstore()
    
    def _build_vectorstore(self):
        """Build the vector store from knowledge base."""
        documents = self._create_documents()
        
        if not documents:
            print("  ⚠️ No documents to index")
            return
        
        print(f"  📝 Indexing {len(documents)} retailer documents...")
        
        self.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            persist_directory=str(self.PERSIST_DIR)
        )
        
        print(f"  ✅ Retailer vector store built")
    
    def rebuild_index(self):
        """Force rebuild of the vector store."""
        import shutil
        if self.PERSIST_DIR.exists():
            shutil.rmtree(self.PERSIST_DIR)
        self._load_knowledge_base()
        self._build_vectorstore()
    
    def get_retailers_for_category(self, category: str) -> list[dict]:
        """Get recommended retailers for a product category."""
        category_lower = category.lower()
        category_retailers = self.knowledge_base.get("category_retailers", {})
        retailers = self.knowledge_base.get("retailers", {})
        
        # Direct lookup first
        if category_lower in category_retailers:
            retailer_ids = category_retailers[category_lower]
        else:
            # Default to general electronics retailers
            retailer_ids = category_retailers.get("electronics", 
                ["amazon", "best_buy", "walmart", "target", "costco"])
        
        result = []
        for r_id in retailer_ids:
            if r_id in retailers:
                r_data = retailers[r_id]
                result.append({
                    "id": r_id,
                    "name": r_data.get("name", r_id),
                    "domain": r_data.get("domain", ""),
                    "search_url_pattern": r_data.get("search_url_pattern", ""),
                    "best_for": r_data.get("best_for", [])
                })
        
        return result
    
    def get_search_urls(self, product_query: str, category: str = "electronics") -> dict:
        """
        Get search URLs for all relevant retailers.
        
        Args:
            product_query: The product to search for
            category: Product category to determine relevant retailers
        
        Returns:
            Dict of {retailer_name: search_url}
        """
        retailers = self.get_retailers_for_category(category)
        encoded_query = product_query.replace(" ", "+")
        
        urls = {}
        for retailer in retailers:
            pattern = retailer.get("search_url_pattern", "")
            if pattern:
                url = pattern.replace("{query}", encoded_query)
                urls[retailer["name"]] = url
        
        return urls
    
    def get_retailer_info(self, retailer_name: str) -> dict:
        """Get detailed information about a specific retailer."""
        retailers = self.knowledge_base.get("retailers", {})
        
        # Try exact match first
        retailer_key = retailer_name.lower().replace(" ", "_").replace("&", "").replace("-", "")
        
        if retailer_key in retailers:
            return retailers[retailer_key]
        
        # Try partial match
        for key, data in retailers.items():
            if retailer_name.lower() in data.get("name", "").lower():
                return data
        
        return {}
    
    def generate_search_prompt(self, product_query: str, category: str = "electronics") -> str:
        """
        Generate a search prompt with retailer-specific information using RAG.
        
        Args:
            product_query: The product to search for
            category: Product category
        
        Returns:
            Generated search prompt string
        """
        # Get retailers and URLs
        retailers = self.get_retailers_for_category(category)
        urls = self.get_search_urls(product_query, category)
        
        # Build retailer info section
        retailer_info = []
        for retailer in retailers:
            name = retailer["name"]
            url = urls.get(name, "")
            best_for = ", ".join(retailer.get("best_for", []))
            retailer_info.append(f"- {name}: {url}")
        
        # Get search tip for category
        search_tip = self.knowledge_base.get("search_tips", {}).get(
            category.lower(), "Use specific product name and model number"
        )
        
        prompt = f"""Search for "{product_query}" prices at these retailers:

RETAILER URLS (use these exact URLs in your results):
{chr(10).join(retailer_info)}

SEARCH TIP: {search_tip}

For each retailer, report:
- Retailer Name
- URL: Use the exact URL from above
- Base Price
- Tax (9.25% for ZIP 94022)
- Shipping (Free unless specified)
- Total Price

Format as: [Retailer](URL)
"""
        return prompt
    
    def lookup_retailers(self, product_query: str, category: str = "electronics") -> str:
        """
        Look up retailer information using RAG.
        
        Args:
            product_query: Product to search for
            category: Product category
        
        Returns:
            Formatted retailer information string
        """
        if not self.vectorstore:
            return self._fallback_lookup(product_query, category)
        
        # Query vector store for relevant retailers
        query = f"retailers for {category} products like {product_query}"
        results = self.vectorstore.similarity_search(query, k=5)
        
        # Get URLs
        urls = self.get_search_urls(product_query, category)
        
        # Format output
        output_lines = [f"Retailers for {product_query} ({category}):"]
        output_lines.append("")
        
        for name, url in urls.items():
            retailer_data = self.get_retailer_info(name)
            notes = retailer_data.get("notes", "")
            best_for = ", ".join(retailer_data.get("best_for", []))
            output_lines.append(f"**{name}**")
            output_lines.append(f"  URL: {url}")
            output_lines.append(f"  Best for: {best_for}")
            if notes:
                output_lines.append(f"  Note: {notes}")
            output_lines.append("")
        
        return "\n".join(output_lines)
    
    def _fallback_lookup(self, product_query: str, category: str) -> str:
        """Fallback when vector store not available."""
        urls = self.get_search_urls(product_query, category)
        
        lines = [f"Search URLs for {product_query}:"]
        for name, url in urls.items():
            lines.append(f"- {name}: {url}")
        
        return "\n".join(lines)


# Singleton instance
_search_rag_instance: Optional[SearchRAG] = None


def get_search_rag() -> SearchRAG:
    """Get or create the singleton SearchRAG instance."""
    global _search_rag_instance
    if _search_rag_instance is None:
        _search_rag_instance = SearchRAG()
    return _search_rag_instance


def get_search_urls_rag(product_query: str, category: str = "electronics") -> dict:
    """Convenience function to get search URLs using RAG."""
    rag = get_search_rag()
    return rag.get_search_urls(product_query, category)


def generate_search_prompt_rag(product_query: str, category: str = "electronics") -> str:
    """Convenience function to generate search prompt using RAG."""
    rag = get_search_rag()
    return rag.generate_search_prompt(product_query, category)


if __name__ == "__main__":
    # Test the RAG system
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing Search RAG System\n" + "=" * 40)
    
    rag = SearchRAG()
    
    # Test getting retailers for category
    print("\n=== Electronics Retailers ===")
    retailers = rag.get_retailers_for_category("electronics")
    for r in retailers[:5]:
        print(f"  {r['name']}: {r['domain']}")
    
    # Test getting search URLs
    print("\n=== Search URLs for 'Sony WH-1000XM5' ===")
    urls = rag.get_search_urls("Sony WH-1000XM5", "electronics")
    for name, url in list(urls.items())[:5]:
        print(f"  {name}: {url[:60]}...")
    
    # Test generating search prompt
    print("\n=== Generated Search Prompt ===")
    prompt = rag.generate_search_prompt("AirPods Pro 2", "electronics")
    print(prompt[:500] + "...")


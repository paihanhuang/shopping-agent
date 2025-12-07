"""
Cashback RAG Module - Retrieval-Augmented Generation for cashback rates.

This module uses a local knowledge base and vector embeddings to retrieve
accurate cashback information without relying on web searches.
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


class CashbackRAG:
    """RAG system for cashback rate lookups."""
    
    COLLECTION_NAME = "cashback_rates"
    PERSIST_DIR = Path(__file__).parent / "cashback_data" / "chroma_db"
    KNOWLEDGE_BASE_PATH = Path(__file__).parent / "cashback_data" / "knowledge_base.json"
    
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
            print(f"  📚 Loaded knowledge base (updated: {self.knowledge_base.get('last_updated', 'unknown')})")
        else:
            print(f"  ⚠️ Knowledge base not found at {self.KNOWLEDGE_BASE_PATH}")
            self.knowledge_base = {}
    
    def _create_documents(self) -> list[Document]:
        """Convert knowledge base to LangChain documents for embedding."""
        documents = []
        
        portals = self.knowledge_base.get("portals", {})
        
        for portal_id, portal_data in portals.items():
            portal_name = portal_data.get("name", portal_id)
            exclusions = portal_data.get("exclusions", [])
            
            # Create document for each retailer in this portal
            for retailer_id, retailer_data in portal_data.get("retailers", {}).items():
                # Build comprehensive text for this retailer-portal combination
                base_rate = retailer_data.get("base_rate", "varies")
                categories = retailer_data.get("categories", {})
                notes = retailer_data.get("notes", "")
                
                # Format category rates
                category_text = ", ".join([f"{cat}: {rate}" for cat, rate in categories.items()])
                
                content = f"""
Portal: {portal_name}
Retailer: {retailer_id.replace('_', ' ').title()}
Base Cashback Rate: {base_rate}
Category-Specific Rates: {category_text if category_text else 'Same as base rate'}
Notes: {notes}
"""
                
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        "portal": portal_name,
                        "portal_id": portal_id,
                        "retailer": retailer_id,
                        "base_rate": base_rate,
                        "source": "knowledge_base"
                    }
                )
                documents.append(doc)
            
            # Create exclusion documents
            for excluded in exclusions:
                content = f"""
Portal: {portal_name}
Retailer: {excluded.replace('_', ' ').title()}
Cashback Rate: NO CASHBACK AVAILABLE
This retailer is explicitly excluded from {portal_name}'s cashback program.
"""
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        "portal": portal_name,
                        "portal_id": portal_id,
                        "retailer": excluded,
                        "base_rate": "0%",
                        "excluded": True,
                        "source": "knowledge_base"
                    }
                )
                documents.append(doc)
        
        # Add category guidance documents
        for category, guidance in self.knowledge_base.get("category_guidance", {}).items():
            content = f"""
Product Category: {category.title()}
Typical Cashback Range: {guidance.get('typical_range', 'varies')}
Best Portal for {category.title()}: {guidance.get('best_portal', 'varies by retailer')}
Notes: {guidance.get('notes', '')}
"""
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "category": category,
                    "source": "category_guidance"
                }
            )
            documents.append(doc)
        
        # Add universal exclusion documents
        for retailer, reason in self.knowledge_base.get("universal_exclusions", {}).items():
            content = f"""
Retailer: {retailer.replace('_', ' ').title()}
Cashback Status: UNIVERSALLY EXCLUDED
Reason: {reason}
This retailer does not participate in any major cashback portal programs.
"""
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "retailer": retailer,
                    "excluded": True,
                    "universal_exclusion": True,
                    "source": "universal_exclusions"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _init_vectorstore(self):
        """Initialize or load the vector store."""
        # Check if we have a persisted store
        if self.PERSIST_DIR.exists() and any(self.PERSIST_DIR.iterdir()):
            print(f"  📂 Loading existing vector store from {self.PERSIST_DIR}")
            self.vectorstore = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.PERSIST_DIR)
            )
        else:
            print("  🔨 Building new vector store...")
            self._build_vectorstore()
    
    def _build_vectorstore(self):
        """Build the vector store from knowledge base."""
        documents = self._create_documents()
        
        if not documents:
            print("  ⚠️ No documents to index")
            return
        
        print(f"  📝 Indexing {len(documents)} documents...")
        
        # Create persist directory
        self.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create vector store with documents
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            persist_directory=str(self.PERSIST_DIR)
        )
        
        print(f"  ✅ Vector store built and persisted to {self.PERSIST_DIR}")
    
    def rebuild_index(self):
        """Force rebuild of the vector store."""
        import shutil
        if self.PERSIST_DIR.exists():
            shutil.rmtree(self.PERSIST_DIR)
        self._load_knowledge_base()
        self._build_vectorstore()
    
    def lookup_cashback(self, retailers: list[str], product_category: str) -> str:
        """
        Look up cashback rates for given retailers and category.
        
        Args:
            retailers: List of retailer names
            product_category: Product category (electronics, clothing, etc.)
        
        Returns:
            Formatted cashback information string
        """
        if not self.vectorstore:
            return "Cashback information unavailable - vector store not initialized."
        
        if not retailers:
            return "No retailers provided for cashback lookup."
        
        print(f"\n💰 [Cashback RAG] Looking up rates for: {', '.join(retailers)}")
        print(f"   Category: {product_category}")
        
        # Collect relevant documents for each retailer
        all_results = []
        
        for retailer in retailers:
            # Create query for this retailer
            query = f"cashback rate for {retailer} {product_category}"
            
            # Retrieve relevant documents
            results = self.vectorstore.similarity_search(query, k=4)
            
            # Filter results relevant to this retailer
            retailer_lower = retailer.lower().replace(" ", "_").replace("-", "_")
            relevant_docs = []
            
            for doc in results:
                doc_retailer = doc.metadata.get("retailer", "").lower()
                # Check if document is about this retailer
                if doc_retailer == retailer_lower or retailer.lower() in doc.page_content.lower():
                    relevant_docs.append(doc)
            
            if relevant_docs:
                all_results.extend(relevant_docs)
        
        # Also get category guidance
        category_query = f"cashback rates for {product_category} products"
        category_docs = self.vectorstore.similarity_search(category_query, k=2)
        
        for doc in category_docs:
            if doc.metadata.get("source") == "category_guidance":
                all_results.append(doc)
        
        if not all_results:
            return self._fallback_lookup(retailers, product_category)
        
        # Use LLM to synthesize the results
        context = "\n\n---\n\n".join([doc.page_content for doc in all_results])
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a cashback specialist. Based on the retrieved information, 
provide accurate cashback rates for the specified retailers and product category.

RULES:
1. Only report rates that are explicitly mentioned in the context
2. If a retailer is marked as EXCLUDED, report "No cashback available"
3. Use category-specific rates when available
4. Format: [Retailer]: Rakuten X%, Capital One Shopping X%, ShopBack X%
5. If no info found for a retailer, say "No data available"
6. Be concise - one line per retailer"""),
            ("human", """Product Category: {category}
Retailers to look up: {retailers}

Retrieved Information:
{context}

Provide cashback rates for each retailer listed above.""")
        ])
        
        chain = synthesis_prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "category": product_category,
            "retailers": ", ".join(retailers),
            "context": context
        })
        
        print("  ✅ RAG lookup complete!")
        return result
    
    def _fallback_lookup(self, retailers: list[str], category: str) -> str:
        """Fallback when no vector store results found."""
        print("  ⚠️ No RAG results, using direct knowledge base lookup")
        
        results = []
        portals = self.knowledge_base.get("portals", {})
        universal_exclusions = self.knowledge_base.get("universal_exclusions", {})
        
        for retailer in retailers:
            retailer_key = retailer.lower().replace(" ", "_").replace("-", "_").replace("&", "")
            
            # Check universal exclusions first
            if retailer_key in universal_exclusions:
                results.append(f"{retailer}: No cashback (excluded from all portals)")
                continue
            
            # Check each portal
            portal_rates = []
            for portal_id, portal_data in portals.items():
                portal_name = portal_data.get("name", portal_id)
                exclusions = portal_data.get("exclusions", [])
                
                if retailer_key in exclusions:
                    portal_rates.append(f"{portal_name}: No cashback")
                elif retailer_key in portal_data.get("retailers", {}):
                    retailer_info = portal_data["retailers"][retailer_key]
                    categories = retailer_info.get("categories", {})
                    
                    # Try to get category-specific rate
                    rate = categories.get(category.lower(), retailer_info.get("base_rate", "varies"))
                    portal_rates.append(f"{portal_name} {rate}")
            
            if portal_rates:
                results.append(f"{retailer}: {', '.join(portal_rates)}")
            else:
                results.append(f"{retailer}: No cashback data available")
        
        return "\n".join(results)
    
    def get_category_guidance(self, category: str) -> str:
        """Get general guidance for a product category."""
        guidance = self.knowledge_base.get("category_guidance", {}).get(category.lower(), {})
        
        if guidance:
            return f"""
Category: {category}
Typical cashback range: {guidance.get('typical_range', 'varies')}
Best portal: {guidance.get('best_portal', 'varies')}
Notes: {guidance.get('notes', 'N/A')}
"""
        return f"No specific guidance for {category} category"


# Singleton instance for reuse
_cashback_rag_instance: Optional[CashbackRAG] = None


def get_cashback_rag() -> CashbackRAG:
    """Get or create the singleton CashbackRAG instance."""
    global _cashback_rag_instance
    if _cashback_rag_instance is None:
        _cashback_rag_instance = CashbackRAG()
    return _cashback_rag_instance


def lookup_cashback_rates_rag(retailers: list[str], product_category: str) -> str:
    """
    Convenience function to look up cashback rates using RAG.
    
    Args:
        retailers: List of retailer names
        product_category: Product category
    
    Returns:
        Formatted cashback rates string
    """
    rag = get_cashback_rag()
    return rag.lookup_cashback(retailers, product_category)


if __name__ == "__main__":
    # Load environment variables for testing
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test the RAG system
    print("Testing Cashback RAG System\n" + "=" * 40)
    
    rag = CashbackRAG()
    
    # Test lookup
    test_retailers = ["Amazon", "Best Buy", "Costco", "Target", "Walmart"]
    test_category = "electronics"
    
    print(f"\nLooking up cashback for: {test_retailers}")
    print(f"Category: {test_category}\n")
    
    result = rag.lookup_cashback(test_retailers, test_category)
    print("\n" + "=" * 40)
    print("RESULTS:")
    print("=" * 40)
    print(result)


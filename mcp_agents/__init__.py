"""
MCP-based Shopping Agents

This module implements the shopping agent as multiple MCP (Model Context Protocol) servers
that collaborate to provide comprehensive price comparison with cashback and credit card rewards.

Architecture:
- Product Search Server: Searches retailers for product prices
- Cashback Server: Looks up cashback rates from shopping portals
- Credit Card Server: Finds best credit card rewards
- Verification Server: Validates and cleans results
- Orchestrator: Coordinates all servers
"""


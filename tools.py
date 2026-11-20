import os
import logging
from langchain_core.tools import tool
from tavily import TavilyClient

logger = logging.getLogger(__name__)
logger.info(logging.INFO)

@tool
def web_search(query: str) -> dict:
    """Do a web search on Tavily

    Args:
        query (str): Query to be executed

    Returns:
        dict: Result of the web search.
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query)

    logger.info(f"Web search tool -> Query: {query}")

    return response
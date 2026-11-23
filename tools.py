import os
from langchain_core.tools import tool
from tavily import TavilyClient
from logger import get_logger

logger = get_logger(__name__)


@tool
def web_search(query: str) -> dict:
    """Do a web search on Tavily

    Args:
        query (str): Query to be executed

    Returns:
        dict: Result of the web search.
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=10,
        include_domains=[
            "pubmed.ncbi.nlm.nih.gov",
            "who.int",
            "ourworldindata.org",
            "data.cdc.gov",
            "scielo.br",
            "paho.org",
        ],
    )

    logger.info("Web search tool -> Search executed.")
    return response

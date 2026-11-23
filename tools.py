import os
import logging
from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from tavily import TavilyClient

logger = logging.getLogger(__name__)
logger.info(logging.INFO)


@tool
def set_patient_name(
    patient_name: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Save the patient's name if they introduce themselves

    Args:
        patient_name (str): The patient's name identified on the conversation.

    Returns:
        Command: Updates "patient_name" on the graph state.
    """

    logger.info('Patient name tool -> "patient_name" set')

    logger.info("P")
    return Command(
        update={
            "patient_name": patient_name.strip(),
            "messages": [ToolMessage("Patient name saved", tool_call_id=tool_call_id)],
        }
    )


@tool
def set_patient_knowledge_level(
    patient_knowledge_level: Literal["LOW", "MEDIUM", "HIGH"],
    tool_call_id: Annotated[str,  InjectedToolCallId],
) -> Command:
    """Save the patient's knowledge level after the quiz question based on the knowledge shared after assessing it on a quiz.
    It should take in consideration how hard was the question.

    Args:
        patient_knowledge_level (Literal["LOW", "MEDIUM", "HIGH"]): The patient's knowledege level after the "quiz question".

    Returns:
        Command: Updates "patient_knowledge_level" on the graph state.
    """

    logger.info(
        f'Knowledge level tool -> "patient_knowledge_level" set: {patient_knowledge_level}'
    )
    return Command(
        update={
            "patient_knowledge_level": patient_knowledge_level,
            "messages": [ToolMessage("Patient name saved", tool_call_id=tool_call_id)],
        }
    )


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
            "https://pubmed.ncbi.nlm.nih.gov",
            "https://www.who.int",
            "https://ourworldindata.org",
            "https://data.cdc.gov",
            "https://www.scielo.br",
            "https://www.paho.org",
        ],
    )

    logger.info("Web search tool -> Search executed.")
    return response

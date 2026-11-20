import os
import logging
import uuid
from dotenv import load_dotenv
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from helper import display_text_to_user, ask_user_for_input, print_spacing
from tools import web_search

load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None
assert os.getenv("TAVILY_API_KEY") is not None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class State(MessagesState):
    example: str


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=True)
workflow = StateGraph(State)


def entry_point(state: State, config: RunnableConfig):
    history = state["messages"]
    messages = []

    if not history:
        pass


def agent(state: State):

    return state


def router(state: State):
    return "agent"


workflow.add_node(agent)
workflow.add_node("tools", ToolNode([web_search]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(source="agent", path=router, path_map=["tools", END])
workflow.add_edge("tools", END)


memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

print("Graph: ")
app.get_graph().print_ascii()

thread_id = str(uuid.uuid4())
config = RunnableConfig(recursion_limit=2000, configurable={"thread_id": thread_id})


initial_state = {
    "messages": [],
}

app.invoke(
    initial_state,
    config,
)

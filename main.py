import os
from logger import setup_logging, get_logger, set_thread_id
import uuid
import operator
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    RemoveMessage,
)
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from helper import (
    display_text_to_user,
    get_few_shot_template,
    print_spacing,
)
from tools import set_patient_name, set_patient_knowledge_level, web_search

load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None
assert os.getenv("TAVILY_API_KEY") is not None

setup_logging()
logger = get_logger("main")


class State(MessagesState):
    patient_name: str | None
    patient_knowledge_level: Literal["LOW", "MEDIUM", "HIGH"]
    topic: str | None
    research: str
    quiz_question: str | None
    grade: Literal["F", "D", "C", "B", "A"] | None


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=True)

workflow = StateGraph(State)


def entry_point(state: State, config: RunnableConfig) -> State:
    grounding_prompt = SystemMessage(
        'Your name is "HealthBot" and you are a specialist in medical and public health information. '
        "Your primary objective is to provide accurate, evidence-based information grounded exclusively in reliable medical and scientific sources. "
        'You will be interacting with patients, and at first you should assume their knowledge of the topic is "LOW" (the patient\'s knowledge level can be "LOW", '
        '"MEDIUM", or "HIGH"). They will ask you for information about health topics or medical conditions. You will provide the patient with the information '
        "retrieved from a web search, phrasing and presenting it according to the patient's knowledge level. After presenting "
        "the information, wait for the patient to ask for a comprehension check, then generate a single, relevant quiz question based on the information you "
        "provided earlier. After the patient answers the quiz question, always assess their knowledge based on how difficult the question was and update it. "
        "Only present another topic after a comprehension check.\n"
        'Restrictions: DO NOT use general knowledge to "fill gaps" in the retrieved information. DO NOT invent missing information. DO NOT claim to have consulted a source that was '
        "not actually retrieved. DO NOT provide unsupported diagnoses or personalized treatment recommendations; your role is to present information and assess the patient's "
        "understanding through a comprehension check.\n"
        "Response style: Be objective, precise, and scientifically rigorous; use clear language without sacrificing medical accuracy; include citations and source URLs whenever "
        "available; prioritize evidence quality over the quantity of information."
    )

    logger.info("Entry point node.")
    return {"messages": [grounding_prompt], "patient_knowledge_level": "LOW"}


def agent_ask_topic(state: State) -> State:
    agent_question_topic = AIMessage("What would you like to learn about?")
    get_topic = interrupt(agent_question_topic.content)

    logger.info("Agent ask topic node.")
    return {
        "messages": [agent_question_topic, HumanMessage(get_topic)],
        "topic": get_topic,
    }


def agent_research(state: State):
    results = web_search.invoke({"query": state["topic"]})

    search = model.invoke(
        [
            state["messages"][0],
            SystemMessage(
                f"Summarize these results for a patient with {state['patient_knowledge_level']} "
                "knowledge, in at most 2 paragraphs, citing source URLs:\n"
                f"Results:\n {results}"
            ),
        ]
    )

    print(search.content)
    logger.info("Agent research node.")
    return {"messages": [search], "research": search.content}

    # if state["research"]:
    #     system_message = SystemMessage(
    #         "Analyze if the user question can be answered by the research it self,"
    #         " or get more information through the web search tool.\n"
    #         + basic_message
    #         + f"The result of the research is: {'\n'.join(state['research'])}"
    #     )

    #     agent_message = model_with_tools.invoke(state["messages"] + [system_message])
    # else:
    #     system_message = SystemMessage(basic_message)

    #     agent_message = model_with_tools.invoke(system_message + state["messages"])

    # print(agent_message.content)
    # return {"research": agent_message.content}


def agent_review_before_quiz(
    state: State,
) -> Command[
    Literal["agent_answer_question", "agent_quiz", "agent_review_before_quiz"]
]:
    agent_message = AIMessage(
        "Anything about this topic you want to review before I give you a question for a comprehension check? "
        "(Ask a question or say that you are ready)"
    )

    get_human_answer = interrupt(agent_message.content)
    patient_message = HumanMessage(get_human_answer)

    analyze_question_system_message = SystemMessage(
        "Analyze if the question below is actually some kind of question and if it is related to the topic following the rules:"
        "Rules: \n"
        '- Answer with "QUESTION" if it is a question and is related to the topic;\n'
        '- Answer "OFF_TOPIC" if it\'s not related to the topic;\n'
        "- Answer \"READY\" if it's not a question and the user want's to proceed."
        f'Question: "{get_human_answer}" \n'
        f"The topic: {state['research']}"
    )

    check = model.invoke(
        [analyze_question_system_message, agent_message, patient_message]
    )

    state_update = {"messages": [agent_message, patient_message]}

    logger.info("Agent review before quiz node.")

    if check.content == "READY":
        return Command(update=state_update, goto="agent_quiz")

    if check.content == "QUESTION":
        return Command(update=state_update, goto="agent_answer_question")

    no_answer_agent_message = AIMessage("The question is not related to the topic.")
    print(no_answer_agent_message.content)

    return Command(update=state_update, goto="agent_review_before_quiz")


def agent_answer_question(state: State) -> State:
    agent_answer = model.invoke(
        [
            state["messages"][0],
            SystemMessage(
                "Answer the patient's question using the research below. "
                "Try to answer with the informaton in it and only that. "
                "Do not try to use general information on the answer ."
                "If you feel that your answer is missing something, report "
                "to the user that this is not the complete information.\n"
                f"Research: {state['research']}"
            ),
            state["messages"][-1],
        ]
    )

    print(agent_answer.content)
    logger.info("Agent answer question node.")
    return {"messages": [agent_answer]}


def agent_quiz(state: State) -> State:
    system_message = SystemMessage(
        f"Generate ONE quiz question based on the knowledge {state['patient_knowledge_level']} and this research:\n"
        + "\n".join(state["research"])
    )
    agent_message = model.invoke([system_message])

    logger.info("Agent quiz node.")
    return {"messages": [agent_message], "quiz_question": agent_message.content}


def agent_grade_answer(state: State) -> None:
    question = state["messages"][-1].content
    patient_answer = interrupt(f"{question} \nWhat's the answer?")
    few_shot_messages = get_few_shot_template().invoke({}).to_messages()

    messages_with_few_shots = [
        *few_shot_messages,
        question,
        HumanMessage(patient_answer),
    ]

    # system_message = SystemMessage('Please evaluate the answer using a grading scale from "F" to "A", where "F" '
    #                                 'represents the lowest grade and "A" represents the highest grade.'
    #                                 'The only acceptable grades are: "F", "D", "C", "B", and "A".'
    #                                 "Return exactly one of these grades, with no additional text or explanation.")

    grade = model.invoke(messages_with_few_shots)
    print(grade.content)


def ask_user_about_finish(state: State) -> Command[Literal["summarize_and_clean", END]]:
    patient_input = interrupt(
        "Do you want to continue the learning experience and learn about another topic? (Y/N)"
    )

    if patient_input.strip().lower() in ("yes", "y"):
        return Command(goto="summarize_and_clean")

    return Command(goto=END)


def summarize_and_clean(state: State) -> State:
    system_message, *conversation = state["messages"]

    system_sumarization_message = SystemMessage(
        "Analyze this conversation and resume it in a short paragraph keeping only the essential and basic information."
    )
    summarization = model.invoke(
        [system_message] + conversation + [system_sumarization_message]
    )

    messages = (
        [system_message]
        + [RemoveMessage(id=message.id) for message in conversation]
        + [summarization]
    )

    state["research"] = []
    logger.info("Summarize and clean node.")
    return {"messages": messages, "quiz_question": None, "grade": None, "topic": None}


# def router(state: State):
#     logger.info("Router accessed.")

#     if state["messages"][-1].tool_calls:
#         return "tools"

#     return "agent"

for node in [
    entry_point,
    agent_ask_topic,
    agent_research,
    agent_review_before_quiz,
    agent_answer_question,
    agent_quiz,
    agent_grade_answer,
    ask_user_about_finish,
    summarize_and_clean,
]:
    workflow.add_node(node)

workflow.add_node(
    "tools", ToolNode([set_patient_name, set_patient_knowledge_level, web_search])
)

workflow.add_edge(START, "entry_point")
workflow.add_edge("entry_point", "agent_ask_topic")
workflow.add_edge("agent_ask_topic", "agent_research")
workflow.add_edge("agent_research", "agent_review_before_quiz")
workflow.add_edge("agent_answer_question", "agent_review_before_quiz")
workflow.add_edge("agent_quiz", "agent_grade_answer")
workflow.add_edge("agent_grade_answer", "ask_user_about_finish")
workflow.add_edge("summarize_and_clean", "agent_ask_topic")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# TODO: Check the problem in the graph print
# print("Graph: ")
# app.get_graph().print_ascii()

# print_spacing()

thread_id = str(uuid.uuid4())
set_thread_id(thread_id)
config = RunnableConfig(recursion_limit=2000, configurable={"thread_id": thread_id})


try:
    logger.info("Starting the run")
    run = app.invoke({"messages": []}, config)
    while "__interrupt__" in run:
        interrupt_message = run["__interrupt__"][0].value
        patient_input = input(interrupt_message + "\n")
        run = app.invoke(Command(resume=patient_input), config)
except Exception as e:
    basic_message = "Run failed"
    logger.exception(basic_message)
    print(f"{basic_message} - Error {e}")
finally:
    logger.info("Finishing the script")

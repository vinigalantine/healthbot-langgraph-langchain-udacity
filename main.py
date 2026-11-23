import os
from logger import setup_logging, get_logger, set_thread_id
import uuid
from typing import Literal
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from helper import get_few_shot_template, print_spacing, clear_console, show_graph
from tools import web_search
from output_schemas import AskTopicValidation, BeforeQuizOutput, GradeOutput

load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None
assert os.getenv("TAVILY_API_KEY") is not None

setup_logging()
logger = get_logger("main")


class State(MessagesState):
    patient_knowledge_level: Literal["LOW", "MEDIUM", "HIGH"]
    topic: str | None
    research: str
    quiz_question: str | None
    grade: Literal["F", "D", "C", "B", "A"] | None


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

workflow = StateGraph(State)


def entry_point(state: State, config: RunnableConfig) -> State:
    grounding_prompt = SystemMessage(
        'Your name is "HealthBot" and you are a specialist in medical or public health information. '
        "Your primary objective is to provide accurate, evidence-based information grounded exclusively in reliable medical and scientific sources. "
        'You will be interacting with patients, and at first you should assume their knowledge of the topic is "LOW" (the patient\'s knowledge level can be "LOW", '
        '"MEDIUM", or "HIGH"). They will ask you for information about health topics or medical conditions. You will provide the patient with the information '
        "retrieved from a web search, phrasing and presenting it according to the patient's knowledge level. After presenting the information, wait for the patient "
        "to say that they are ready for a comprehension check, then generate a single, relevant quiz question based on the information you provided earlier.\n"
        'Restrictions: DO NOT use general knowledge to "fill gaps" in the retrieved information. DO NOT invent missing information. DO NOT claim to have consulted a source that was '
        "not actually retrieved. DO NOT provide unsupported diagnoses or personalized treatment recommendations;\n"
        "Response style: Be objective, precise, and scientifically rigorous; use clear language without sacrificing medical accuracy; include citations and source URLs whenever "
        "available; prioritize evidence quality over the quantity of information."
    )

    logger.info("Entry point node.")
    return {"messages": [grounding_prompt], "patient_knowledge_level": "LOW"}


def agent_ask_topic(
    state: State,
) -> Command[Literal["agent_ask_topic", "agent_research"]]:
    agent_question_topic = AIMessage("What would you like to learn about?")
    get_topic = interrupt(agent_question_topic.content)

    check = model.with_structured_output(AskTopicValidation).invoke(
        [
            SystemMessage(
                "Validate if this message is related to medical or public health information."
            ),
            HumanMessage(get_topic),
        ]
    )

    if check.is_medical_related == "NO":
        print("This topic is not related to medical or public health information.")
        return Command(goto="agent_ask_topic")

    logger.info(f"Agent ask topic node -> {len(state['messages'])}")
    return Command(
        update={
            "messages": [agent_question_topic, HumanMessage(get_topic)],
            "topic": get_topic,
        },
        goto="agent_research",
    )


def agent_research(state: State):
    results = web_search.invoke({"query": state["topic"]})

    search = model.invoke(
        [
            state["messages"][0],
            SystemMessage(
                f"Summarize these results for a patient with {state['patient_knowledge_level']} "
                "knowledge, in at most 4 paragraphs, citing source URLs:\n"
                f"Results:\n {results}"
            ),
        ]
    )

    print_spacing()
    print(search.content)
    logger.info(f"Agent research node -> Number ofresults returned: {len(results)}")
    return {"messages": [search], "research": search.content}


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
        "Analyze if the patient answer to the question.\n"
        f"Consider this research: {state['research']}"
    )

    check = model.with_structured_output(BeforeQuizOutput).invoke(
        [analyze_question_system_message, agent_message, patient_message]
    )

    state_update = {"messages": [agent_message, patient_message]}

    logger.info(f"Agent review before quiz node -> Anlysis {check.analysis}")

    if check.analysis == "READY":
        return Command(update=state_update, goto="agent_quiz")

    if check.analysis == "QUESTION":
        return Command(update=state_update, goto="agent_answer_question")

    print("The question is not related to the topic.")

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
        "Generate ONE quiz question, do not share the answer now and it needs to be "
        "based on the knowledge {state['patient_knowledge_level']} and this research:\n"
        + "\n".join(state["research"])
    )
    agent_message = model.invoke([system_message])

    logger.info(f"Agent quiz node -> Question: {agent_message.content}")
    return {"messages": [agent_message], "quiz_question": agent_message.content}


def agent_grade_answer(state: State) -> State:
    question = state["messages"][-1].content
    patient_answer = interrupt(question)
    few_shot_messages = get_few_shot_template().invoke({}).to_messages()

    messages_with_few_shots = [
        *few_shot_messages,
        question,
        HumanMessage(patient_answer),
    ]

    grade = model.with_structured_output(GradeOutput).invoke(messages_with_few_shots)
    print(f"Grade: {grade.grade} - {grade.explanation}")

    logger.info(
        f"Agent grade answer node -> Grade: {grade.grade} - Explanation: {grade.explanation} - Patient Knowledge leve: {grade.patient_knowledge_level}"
    )
    return {"patient_knowledge_level": grade.patient_knowledge_level}


def ask_user_about_finish(state: State) -> Command[Literal["summarize_and_clean", END]]:
    patient_input = interrupt(
        "Do you want to continue the learning experience and learn about another topic? (Y/N)"
    )

    logger.info(f"Ask user about finish node -> Answer: {patient_input}")

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
    clear_console()

    return {"messages": messages, "quiz_question": None, "grade": None, "topic": None}


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

workflow.add_edge(START, "entry_point")
workflow.add_edge("entry_point", "agent_ask_topic")
workflow.add_edge("agent_research", "agent_review_before_quiz")
workflow.add_edge("agent_answer_question", "agent_review_before_quiz")
workflow.add_edge("agent_quiz", "agent_grade_answer")
workflow.add_edge("agent_grade_answer", "ask_user_about_finish")
workflow.add_edge("summarize_and_clean", "agent_ask_topic")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

print("Printing graph...")
show_graph(app.get_graph().draw_mermaid_png())

print_spacing()

thread_id = str(uuid.uuid4())
set_thread_id(thread_id)
config = RunnableConfig(recursion_limit=2000, configurable={"thread_id": thread_id})

try:
    logger.info("Starting the run")
    run = app.invoke({"messages": []}, config)
    while "__interrupt__" in run:
        interrupt_message = run["__interrupt__"][0].value
        patient_input = input(interrupt_message + "\n\nYour answer: ")
        print_spacing()
        run = app.invoke(Command(resume=patient_input), config)
except Exception as e:
    basic_message = "Run failed"
    logger.exception(basic_message)
    print(f"{basic_message} - Error {e}")
finally:
    logger.info("Finishing the script")

import os
import tkinter as tk
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from logger import get_logger

logger = get_logger(__name__)


def print_spacing(bar_multiplier: int = 50) -> None:
    print("\n" + "=" * bar_multiplier + "\n")


def clear_console() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def show_graph(png_bytes: bytes) -> None:
    root = tk.Tk()
    root.title("Mermaid graph")
    image = tk.PhotoImage(data=png_bytes)
    tk.Label(root, image=image).pack()
    root.mainloop()

def get_few_shot_template() -> FewShotChatMessagePromptTemplate:
    examples = [
        {
            "question": "What is the flu?",
            "user_answer": "A viral illness, with just fever and cough as symptoms",
            "is_user_answer_correct": 'WHOPS, THATS NOT IT... Your grade is "D".',
            "explanation": "The answer correctly classify it as a viral illness but is incomplete and kind of misleadin. The flu (influenza) is a contagious respiratory illness caused by influenza viruses. It affects the nose, throat, and sometimes the lungs. It can range from mild to severe."
            "Common symptoms include: fever, cough, sore throat, runny or stuffy nose, muscle and body aches, headache, fatigue, chills",
        },
        {
            "question": "What is chickenpox?",
            "user_answer": "It's a mid contagious disease caused by the varicella-zoster virus with symptoms like: itchy skin rash, fluid-filled blisters, fever, headache, fatigue, loss of appetite, general discomfort.",
            "is_user_answer_correct": 'ALMOST THERE! Your grade is "B"',
            "explanation": 'Answer is almost right, you lost points because of the "mid contagious" because chickenpox is a highly contagious disease caused by the varicella-zoster virus. It causes an itchy rash with fluid-filled blisters and is most common in children, although adults can also get it.'
            "Common symptoms include: Itchy skin rash, fluid-filled blisters, fever, headache, fatigue, loss of appetite, general discomfort",
        },
        {
            "question": "What are the normal ranges for blood pressure, heart rate, and body temperature in a healthy adult?",
            "user_answer": "Blood presure would be 120/80 mmHg, heart rate around 60 per minute, body temperature 36º celsius.",
            "is_user_answer_correct": 'RIGHT! Your grade is "A"',
            "explanation": "Perfect answer, all the information is right. A typical normal resting blood pressure is below 120/80 mmHg. A normal resting heart rate for most adults is 60–100 beats per minute. Normal body temperature averages around 37°C (98.6°F), although it can vary depending on the individual, time of day, and measurement method.",
        },
        {
            "question": "Which organ produces insulin, and what role does insulin play in the body?",
            "user_answer": "The organ is a the intestine and it's for running.",
            "is_user_answer_correct": 'WRONG! Your grade is "F"',
            "explanation": "The answer is COMPLETELY wrong... Insulin is a hormone produced by the beta cells of the pancreas. It helps regulate blood glucose levels by allowing cells, particularly muscle and fat cells, to absorb glucose from the bloodstream. It also promotes glucose storage and prevents excessive blood sugar levels.",
        },
    ]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "ai",
                "Now that you read the information that I presented you let's do a comprehension check.\n\tThe question is: {question}",
            ),
            ("human", "{user_answer}"),
            ("ai", "{is_user_answer_correct} - {explanation}"),
        ]
    )

    logger.info("Few-shots generated.")
    return FewShotChatMessagePromptTemplate(
        examples=examples, example_prompt=prompt_template
    )

import time
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

def display_text_to_user(text) -> None:
    print(text) 
    time.sleep(1)
    
def ask_user_for_input(input_description) -> str:
    return input(input_description)

def print_spacing(bar_multiplier: int = 50) -> None:
    print("\n"+"="*50+"\n")
    
# TODO: Explore this idea
def get_few_shot_template() -> FewShotChatMessagePromptTemplate:
    examples = [
        {
            "input": "",
            "output": ""
        }
    ]
    
    ChatPromptTemplate.from_messages(
        ("human", "")
        ("ai", "")
    )
    
    return FewShotChatMessagePromptTemplate()
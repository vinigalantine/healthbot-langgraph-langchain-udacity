from typing import Literal
from pydantic import BaseModel, Field

class AskTopicValidation(BaseModel):
    is_medical_related: Literal["YES", "NO"] = Field(description='"YES" if it is related to medical and "NO" if it is not.')

class BeforeQuizOutput(BaseModel):
    analysis: Literal["QUESTION", "READY", "OFF_TOPIC"] = Field(
        description='"QUESTION" if it\'s a question related to the topic. '
                    '"READY" if the patient is reqady for the quiz. '
                    '"OFF_TOPIC" if the question is not related to the topic or none of the other options.'
    )

class GradeOutput(BaseModel):
    grade: Literal["F", "D", "C", "B", "A"] = Field(description='"A" is fully correct. "F" is completely wrong.')
    explanation: str = Field(description="Explanation about the grading and correct answer.")
    patient_knowledge_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="The patient's knowledge level, considering the the question difficulty and grade.")
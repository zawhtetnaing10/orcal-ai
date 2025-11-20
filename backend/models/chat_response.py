from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    response: str = Field(..., description="Response from LLM.")

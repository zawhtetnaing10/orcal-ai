from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="Query from the user.")

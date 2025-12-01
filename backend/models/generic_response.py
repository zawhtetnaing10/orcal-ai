from pydantic import BaseModel, Field


class GenericResponse(BaseModel):
    message: str = Field(..., description="The response message.")

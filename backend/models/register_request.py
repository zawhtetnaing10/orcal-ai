from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(..., description="Email of the user.")
    password: str = Field(..., description="Password of the user.")
    username: str = Field(..., description="Display name for the user.")

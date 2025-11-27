from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(..., description="Email of the user.")
    password: str = Field(..., description="Password of the user.")

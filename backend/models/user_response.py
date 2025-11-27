from pydantic import BaseModel, Field


class UserResponse(BaseModel):
    uid: str = Field(..., description="User's unique id from.")
    email: str = Field(..., description="Email of the user")
    username: str = Field(..., description="Display name of the user.")

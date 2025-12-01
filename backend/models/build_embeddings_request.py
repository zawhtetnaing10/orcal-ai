from pydantic import BaseModel, Field


class InfoToEmbed(BaseModel):
    id: int = Field(..., description="Id of the data to embed")
    title: str = Field(..., description="Title of the data to embed")
    details: str = Field(..., description="Details of the data to embed")


class BuildEmbeddingsRequest(BaseModel):
    data: list[InfoToEmbed] = Field(
        ..., description="Information to embed and index for semantic and bm25 searches")

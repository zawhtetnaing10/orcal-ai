from fastapi import FastAPI
import uvicorn


app = FastAPI(
    title="Orcal AI",
    description="Personal Assistant RAG App"
)


@app.get("/")
def health_check():
    """
        Health Check Endpoint
    """
    return {"status": "ok", "message": "API is online"}


@app.get("/chat")
async def chat():
    return "The chat feature will be integrated later."


if __name__ == "__main__":
    # Restarts the server automatically on saving
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)

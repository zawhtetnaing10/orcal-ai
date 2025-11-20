from fastapi import FastAPI
import uvicorn
from backend.models.chat_request import ChatRequest
from backend.models.chat_response import ChatResponse

from lib.augmented_generation.rag import RAG
from lib.augmented_generation.rag import TurnHistory
import lib.utils.constants as constants


app = FastAPI(
    title="Orcal AI",
    description="Personal Assistant RAG App"
)

# RAG Instance
rag = RAG()

# Turn History. Replace with last 5 messages from Firestore later.
turn_history = TurnHistory()


@app.get("/")
def health_check():
    """
        Health Check Endpoint
    """
    return {"status": "ok", "message": "API is online"}


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query

    # Add user query to turn history. Replace with Firestore Turn History later.
    turn_history.add_to_turn_history(
        speaker=constants.SPEAKER_USER, text=query)

    # Add error handling here.
    llm_response = rag.discuss(
        query, turn_history=turn_history.get_turn_history_str())

    # Add llm_response to turn history. Replacewith Firestore Turn History later.
    turn_history.add_to_turn_history(
        speaker=constants.SPEAKER_MODEL, text=llm_response)

    chat_response = ChatResponse(response=llm_response)

    return chat_response

if __name__ == "__main__":
    # Restarts the server automatically on saving
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)

from fastapi import FastAPI, HTTPException, status, Header, Depends, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from backend.models.login_request import LoginRequest
from backend.models.chat_request import ChatRequest
from backend.models.chat_response import ChatResponse
from backend.models.register_request import RegisterRequest
from backend.models.user_response import UserResponse
from backend.models.generic_response import GenericResponse
from backend.models.build_embeddings_request import BuildEmbeddingsRequest
import backend.firebase.firebase_client as firebase_client

from lib.augmented_generation.rag import RAG
from lib.augmented_generation.rag import TurnHistory
from lib.augmented_generation.rag_external import RAGExternal
import lib.utils.constants as constants

from starlette.concurrency import run_in_threadpool
from firebase_admin import auth
from pydantic import BaseModel, Field

import firebase_admin.exceptions as firebase_exceptions
import asyncio
from google.cloud.firestore import SERVER_TIMESTAMP

import backend.utils.utils as utils


app = FastAPI(
    title="Orcal AI",
    description="Personal Assistant RAG App"
)
# To get bearer token
security = HTTPBearer()

# RAG Instance
# TODO: - Turn back on after testing Login and Register.
rag = RAG()

# Turn History. Replace with last 5 messages from Firestore later.
turn_history = TurnHistory()


@app.get("/")
def health_check():
    """
        Health Check Endpoint
    """
    return {"status": "ok", "message": "API is online"}


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    db = firebase_client.firestore_async

    if not db:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize firestore client."
        )

    if not request.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email must be provided"
        )

    if not request.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be provided"
        )

    if not request.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User name must be provided"
        )

    # Authenticate with Firebase
    try:
        user_record = await asyncio.to_thread(
            auth.create_user,
            email=request.email,
            password=request.password,
            display_name=request.username
        )
        user_uid = user_record.uid
    except firebase_exceptions.AlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with provided email already exists. Please try another email."
        )
    except Exception as exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{exception}"
        )

    # Create user info in firestore
    try:
        user_info = {
            "uid": user_uid,
            "email": request.email,
            "username": request.username,
            "created_at": SERVER_TIMESTAMP
        }
        user_doc_ref = db.collection("users").document(user_uid)
        user_info_doc_ref = user_doc_ref.collection(
            "profile").document("user_info")
        await user_info_doc_ref.set(user_info)
    except Exception as exception:

        # Create user in auth succeeded but firestore failed.
        # Rollback
        try:
            auth.delete_user(user_uid)
        except Exception as delete_user_exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{delete_user_exception}"
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{exception}"
        )

    # Return the response
    return UserResponse(
        uid=user_uid,
        email=request.email,
        username=request.username
    )


@app.post("/build-embeddings", response_model=GenericResponse, status_code=status.HTTP_200_OK)
async def build_embeddings(request_data: BuildEmbeddingsRequest,
                           credentials: HTTPAuthorizationCredentials = Security(security)):

    # Authorize firebase credentials with firebase auth
    id_token = credentials.credentials
    user_uid = await asyncio.to_thread(
        authenticate_user,
        id_token=id_token
    )

    # Initialize Firestore
    db = firebase_client.firestore_async
    if not db:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize firestore client."
        )
    batch = db.batch()

    try:
        # Bulk Insert knowledge base objects
        knowledge_objects = request_data.data
        user_doc_ref = db.collection("users").document(user_uid)
        knowledge_base_collection_ref = user_doc_ref.collection(
            "knowledge_base")

        for knowledge_object in knowledge_objects:
            doc_ref = knowledge_base_collection_ref.document(
                f"{knowledge_object.id}")
            batch.set(doc_ref, {
                "id": knowledge_object.id,
                "title": knowledge_object.title,
                "details": knowledge_object.details
            })
        await batch.commit()
    except Exception as e:
        print(f"Firebae bulk insert docs failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong building the embeddings and indices."
        )

    try:
        # Prepare docs from request
        docs = utils.convert_build_embeddings_request_to_docs(
            request=request_data, uid=user_uid)

        # Do the embeddings
        rag = RAGExternal()
        await asyncio.to_thread(
            rag.build_embeddings_and_indices,
            documents=docs,
            uid=user_uid
        )

        return GenericResponse(message="Successfully built the embeddings.")
    except Exception as e:
        print(f"Building indices and embeddings failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong building the embeddings and indices."
        )


def authenticate_user(id_token: str):
    """
        User firebase authentication to authenticate id_token from client.
    """
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token["uid"]
    except ValueError as _:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Firebase token must be provided")
    except auth.InvalidIdTokenError as _:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Firebase token is invalid")
    except auth.ExpiredIdTokenError as _:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Firebase token has expired")
    except auth.RevokedIdTokenError as _:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Firebase token has been revoked")
    except auth.CertificateFetchError as _:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Certification fetch failed.")
    except auth.UserDisabledError as _:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="User record has been disabled.")


@app.post("/chat")
async def chat(request: ChatRequest):
    print("Request Received")
    query = request.query

    # Add user query to turn history. Replace with Firestore Turn History later.
    turn_history.add_to_turn_history(
        speaker=constants.SPEAKER_USER, text=query)

    # Add error handling here.
    # llm_response = rag.discuss(
    #     query, turn_history=turn_history.get_turn_history_str())
    try:
        llm_response = await asyncio.to_thread(
            rag.discuss,
            query,
            turn_history.get_turn_history_str()
        )
    except Exception as e:
        print(f"ERROR processing RAG query: {e}")
        llm_response = "Apologies, an internal error occurred while processing the request."

    # Add llm_response to turn history. Replacewith Firestore Turn History later.
    turn_history.add_to_turn_history(
        speaker=constants.SPEAKER_MODEL, text=llm_response)

    chat_response = ChatResponse(response=llm_response)

    return chat_response

if __name__ == "__main__":
    # Restarts the server automatically on saving
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)

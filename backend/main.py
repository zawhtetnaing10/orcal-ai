from fastapi import FastAPI, HTTPException, status, Header, Depends, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from backend.models.chat_request import ChatRequest
from backend.models.register_request import RegisterRequest
from backend.models.user_response import UserResponse
from backend.models.generic_response import GenericResponse
from backend.models.build_embeddings_request import BuildEmbeddingsRequest
import backend.firebase.firebase_client as firebase_client

from lib.augmented_generation.rag_external import RAGExternal
import backend.utils.constants as backend_constants
import lib.utils.constants as constants

from starlette.concurrency import run_in_threadpool
from firebase_admin import auth
from pydantic import BaseModel, Field

import firebase_admin.exceptions as firebase_exceptions
import asyncio
from google.cloud.firestore import SERVER_TIMESTAMP

import backend.utils.utils as utils
from google.cloud.firestore import AsyncCollectionReference
from google.cloud.firestore import Query


app = FastAPI(
    title="Orcal AI",
    description="Personal Assistant RAG App"
)
# To get bearer token
security = HTTPBearer()


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
        user_info_doc_ref = db.collection("users").document(
            user_uid).collection("profile").document("user_info")
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

    # Update RAG Config. Set knowledge base built flag to false
    try:
        rag_info = {
            "is_knowledge_base_built": False,
        }
        user_rag_config_ref = db.collection("users").document(
            user_uid).collection("config").document("rag_config")
        await user_rag_config_ref.set(rag_info)
    except Exception as exception:
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
        knowledge_base_collection_ref = db.collection(
            "users").document(user_uid).collection("knowledge_base")

        for knowledge_object in knowledge_objects:
            doc_ref = knowledge_base_collection_ref.document(
                f"{knowledge_object.id}")
            batch.set(doc_ref, {
                "id": knowledge_object.id,
                "title": knowledge_object.title,
                "details": knowledge_object.details
            }, merge=True)
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
    except Exception as e:
        print(f"Building indices and embeddings failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong building the embeddings and indices."
        )

    # Update RAG Config. Set Knowledgebase built to true.
    try:
        rag_info = {
            "is_knowledge_base_built": True,
        }
        user_rag_config_ref = db.collection("users").document(
            user_uid).collection("config").document("rag_config")
        await user_rag_config_ref.set(rag_info)
    except Exception as exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{exception}"
        )

    return GenericResponse(message="Successfully built the embeddings.")


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
async def chat(request: ChatRequest, credentials: HTTPAuthorizationCredentials = Security(security)):
    # Authorize firebase credentials with firebase auth
    id_token = credentials.credentials
    user_uid = await asyncio.to_thread(
        authenticate_user,
        id_token=id_token
    )

    # Write the message in Firestore
    # Get Firestore
    db = firebase_client.firestore_async
    if not db:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize firestore client."
        )

    # Get Collection Ref
    messages_collection_ref: AsyncCollectionReference = db.collection("users").document(user_uid).collection(
        "chats").document(backend_constants.CONVERSATION_DOCUMENT_KEY).collection("messages")

    # Get the last 5 messages from firestore and sort them
    turn_history = []
    turn_history_query = messages_collection_ref.order_by(
        "timestamp",
        direction=Query.DESCENDING
    ).limit(5)
    turn_history_firebase = await turn_history_query.get()
    if len(turn_history_firebase) > 0:
        # Sort turn history in reverse. For turn history to work, it must be in chronological order.
        turn_history_dicts = [message.to_dict()
                              for message in turn_history_firebase]
        turn_history_dicts.sort(key=lambda message: message["timestamp"])

        # Append the messages to turn_history
        for message in turn_history_dicts:
            turn_history_dict = utils.create_turn_history_object(
                speaker=message["speaker"], text=message["content"])
            turn_history.append(turn_history_dict)

    # Get the message from the user.
    query = request.query
    # Write User's message in Cloud Firestore.
    user_message_timestamp = utils.get_current_time_milliseconds()
    message_dict = create_message_dict(
        uid=user_uid, speaker=constants.SPEAKER_USER, content=query, timestamp=user_message_timestamp)
    await messages_collection_ref.document(f"{user_message_timestamp}").set(message_dict)

    # Generate the response from LLM
    rag = RAGExternal()
    turn_history_str = ""
    if len(turn_history) > 0:
        turn_history_str = f"{turn_history}"
    llm_response = rag.discuss(
        uid=user_uid, query=query, turn_history=turn_history_str)

    # Write the LLM Response to Firestore.
    llm_response_timestamp = utils.get_current_time_milliseconds()
    message_dict = create_message_dict(
        uid=user_uid, speaker=constants.SPEAKER_MODEL, content=llm_response, timestamp=llm_response_timestamp)
    await messages_collection_ref.document(f"{llm_response_timestamp}").set(message_dict)

    # Return the LLM Response
    return GenericResponse(message=llm_response)


def create_message_dict(uid: str, speaker: str, timestamp: int, content: str):
    # Modify as needed later.
    return {
        "uid": uid,
        "speaker": speaker,
        "content": content,
        "timestamp": timestamp,
    }


if __name__ == "__main__":
    # Restarts the server automatically on saving
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)

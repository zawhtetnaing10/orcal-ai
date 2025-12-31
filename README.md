# Oracle-AI 🔮

Your Professional Persona RAG Assistant

## 🚀 Overview

Oracle-AI is a "Professional Persona" assistant. It uses a hybrid RAG (Retrieval-Augmented Generation) setup to make sure it doesn't just give generic AI answers—it gives your answers based on your specific career, skills, and work-life rules.

<img width="1620" height="500" alt="ai-graphic" src="https://github.com/user-attachments/assets/e746e696-0d5e-4828-abf2-51f20edf7e87" />


## 🛠 Technologies Used

### Backend + RAG 

Python & FastAPI: High-performance web service.

Hybrid Search: Combines BM25 (Keyword Search) and ChromaDB (Semantic Search) via LangChain.

RRF (Reciprocal Rank Fusion): Merges search results for superior context retrieval.

### Cloud & Security

Firebase Authentication: Secure user identity management.

Cloud Firestore: Scalable NoSQL storage for user profiles and chat history.

### Mobile 

Flutter Framework: Cross-platform excellence.

BLoC Pattern: Robust and predictable state management.

Retrofit: Type-safe networking for API communication.

Firebase SDK: Seamless integration for real-time Firestore updates.

## 📋 Pre-requisites

### LLM Options (Gemini or Local LLM)

#### Gemini (Cloud): - Create a .env file in the root directory.

Add: 
```
GEMINI_API_KEY=your_key_here
```

#### Ollama (Local):

Download Ollama.

Pull your preferred model (e.g., ollama pull llama3.1).

Update lib/utils/local_ai_utils.py with your chosen model name.

### Firebase Setup

Create a Firebase project.

Download your Private Key JSON from the Service Accounts tab.

Save it as private_keys/service_account_keys.json.

## 🏃 Running the Project

### Start the FastAPI server:

```
uv run -m backend.main
```

## 🔌 API Endpoints

POST /register: Handles Firebase Auth registration and initializes the user's Firestore document.

POST /build_embeddings: Takes the knowledge base objects sent from the frontend and turns them into a searchable Vector + Keyword indices.

POST /chat: The core RAG endpoint.

Receives query -> Performs RRF Search (Reciprocal Rank Fusion) -> Stuffs Context into Prompt -> Returns LLM Response.
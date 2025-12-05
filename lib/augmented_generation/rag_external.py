from lib.augmented_generation.rag import RAG
from lib.hybrid_search.rrf_search_external_docs import RRFSearchExternalDocs
from langchain_core.documents import Document


class RAGExternal(RAG):
    def __init__(self):
        self.rrf_search = RRFSearchExternalDocs()

    def build_embeddings_and_indices(self, documents: list[Document],  uid: str):
        """
            Build embeddings and indices from the given uid.
        """
        self.rrf_search.build_embeddings_and_index(
            documents=documents, uid=uid)

    def discuss(self, uid: str, query: str, turn_history: str = ""):
        # TODO: - Implement discuss for chat endpoint
        pass

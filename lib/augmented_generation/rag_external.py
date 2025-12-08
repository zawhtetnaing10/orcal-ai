from lib.augmented_generation.rag import RAG
from lib.hybrid_search.rrf_search_external_docs import RRFSearchExternalDocs
from langchain_core.documents import Document
import lib.utils.constants as constants
from lib.utils.genai_utils import generate_response
from lib.utils.prompt_utils import get_personal_assistant_rag_prompt


class RAGExternal(RAG):
    def __init__(self):
        self.rrf_search = RRFSearchExternalDocs()

    def build_embeddings_and_indices(self, documents: list[Document],  uid: str):
        """
            Build embeddings and indices from the given uid.
        """
        self.rrf_search.build_embeddings_and_index(
            documents=documents, uid=uid)

    def discuss(self, uid: str, query: str, turn_history: str = "") -> str:
        """
            Do a rrf search based on uid and query.
            Feed the results to LLM and get the result
        """
        # Get the search results
        results = self.rrf_search.rrf_search(
            uid=uid, query=query, limit=constants.DEFAULT_ITEM_LIMIT, k=constants.K_VALUE)

        # Get the prompt
        prompt = get_personal_assistant_rag_prompt(
            query=query, result=results, turn_history=turn_history)

        # Generate
        response = generate_response(prompt=prompt)

        return response.text

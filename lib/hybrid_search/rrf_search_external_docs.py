from lib.hybrid_search.rrf_search import RRFSearch
from lib.bm25_search.inverted_index_external_docs import InvertedIndexExternalDocs
from lib.semantic_search.semantic_search_external_docs import SemanticSearchExternalDocs

import lib.utils.constants as constants
from langchain_core.documents import Document


class RRFSearchExternalDocs(RRFSearch):
    def __init__(self):
        self.inverted_index = InvertedIndexExternalDocs()
        self.semantic_search = SemanticSearchExternalDocs()

    def build_embeddings_and_index(self, documents: list[Document], uid: str):
        """
            Build embeddings for both keyword search and semantic search
        """
        self.inverted_index.build(documents=documents, uid=uid)
        self.semantic_search.build_embeddings(
            documents=documents, uid=uid, is_chunked=False)

    def rrf_search(self, uid: str, query: str, limit: int, k: int = constants.K_VALUE):
        """
            Do a rrf search for given uid and query.
        """
        bm25_results = self.inverted_index.bm25_search(uid=uid,
                                                       query=query, limit=limit * 500)

        semantic_results = self.semantic_search.semantic_search(uid=uid,
                                                                query=query, limit=limit * 500)

        result_dict: dict[str: Document] = {}

        # Add bm25 search results
        for index, bm_25_doc in enumerate(bm25_results):
            # Get the rannk
            bm25_rank = index + 1
            # Get the doc id
            doc_id = bm_25_doc.metadata["id"]
            # Update the doc with bm25 rank
            bm_25_doc.metadata["bm25_rank"] = bm25_rank
            # Add placeholder for semantic rank
            bm_25_doc.metadata["semantic_rank"] = 0.0
            # Put in the result dict
            result_dict[doc_id] = bm_25_doc

        # Add semantic search results
        for index, semantic_search_doc in enumerate(semantic_results):
            # Get the rank
            semantic_rank = index + 1
            # Get the doc id
            doc_id = semantic_search_doc.metadata["id"]

            if doc_id in result_dict:
                existing_doc = result_dict[doc_id]
                existing_doc.metadata["semantic_rank"] = semantic_rank
            else:
                semantic_search_doc.metadata["semantic_rank"] = semantic_rank
                semantic_search_doc.metadata["bm25_rank"] = 0.0
                result_dict[doc_id] = semantic_search_doc

        # Get the values
        result = list(result_dict.values())

        # Calculate rrf and update the results
        for doc in result:
            bm25_rank = doc.metadata["bm25_rank"]
            semantic_rank = doc.metadata["semantic_rank"]

            rrf_bm25 = 0.0
            if bm25_rank != 0:
                rrf_bm25 = self.rrf_score(bm25_rank, k)

            rrf_semantic = 0.0
            if semantic_rank != 0:
                rrf_semantic = self.rrf_score(semantic_rank, k)

            rrf_score = round(rrf_bm25 + rrf_semantic, 5)
            doc.metadata["rrf_score"] = rrf_score

        # Sort the results
        result.sort(key=lambda doc: float(
            doc.metadata["rrf_score"]), reverse=True)

        # Return the limited results
        return result[:limit]

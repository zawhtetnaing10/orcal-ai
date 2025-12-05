from lib.semantic_search.semantic_search import SemanticSearch
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import lib.utils.data_loader_utils as data_loader_utils
import lib.utils.constants as constants

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.api.models.Collection import Collection
from lib.semantic_search.user_vector_store_cache import USER_VECTOR_STORE_CACHE


class SemanticSearchExternalDocs(SemanticSearch):
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=constants.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"}
        )

    def semantic_search(self, uid: str, query: str, limit=3) -> list[Document]:
        """
            Perform a semantic search according to query.
            Will prepare the embeddings first before searching.
        """
        vector_db = self.load_embeddings(uid)

        # Fetch double the amount of chunks since they'll have to be mapped back to documents
        docs_with_scores = vector_db.similarity_search_with_score(
            query, limit)

        # TODO: - If chunking is required later, add the logic back.
        # # Map the chunks to a dict with doc_idx as key and score as value
        # doc_idx_dict = {}
        # for chunk, score in chunks_with_scores:
        #     doc_idx = int(chunk.metadata["doc_idx"])
        #     if doc_idx in doc_idx_dict:
        #         # If doc_id already in dict. Only overwrite if it has lower score (better similarity)
        #         previous_score = doc_idx_dict[doc_idx]
        #         if score < previous_score:
        #             doc_idx_dict[doc_idx] = score
        #     else:
        #         # If not, just add the doc_idx
        #         doc_idx_dict[doc_idx] = score

        # # Sort the dict items with score
        # sorted_doc_idx_with_scores = sorted(
        #     doc_idx_dict.items(), key=lambda item: item[1])
        # # Get the doc_idx
        # sorted_doc_idx = [
        #     item[0] for item in sorted_doc_idx_with_scores
        # ]

        # # Map the doc_idx back to the original docs
        # result = [
        #     self.documents[doc_idx] for doc_idx in sorted_doc_idx
        # ]

        return docs_with_scores[:limit]

    def build_embeddings(self, documents, uid: str, is_chunked: bool = False):
        """
            Chunk all the documents and build embeddings out of it.
        """
        try:
            all_chunks = []
            if is_chunked:
                # If is_chunked chunk the docs by sentence
                for index, doc in enumerate(documents):
                    chunks = self.semantic_chunk(doc, index)
                    all_chunks.extend(chunks)

                print("All documents chunked successfully")
                for index, chunk in enumerate(all_chunks):
                    print(f"{index + 1}. {chunk}")
                    print("")

                # If all chunks are empty just return
                if not all_chunks:
                    print(f"There are no chunks to save.")
                    return
            else:
                # If not, chunks will be the documents themselves
                all_chunks = documents

            # Embed and Save to ChromaDB
            print(
                f"Embedding and saving {len(all_chunks)} chunks to {constants.CHROMA_PATH}...")
            Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=constants.CHROMA_PATH,
                collection_name=uid
            )

            print(
                f"Embedding process for {uid} complete. Vector database built successfully.")

        except Exception as e:
            print(f"Embedding failed with exception {e}")

    def load_embeddings(self, uid: str) -> Chroma:
        """
            Load the embeddings.
            If embeddings haven't been built yet, raise an exception
        """

        # Load the documents first. To be sure
        if self._check_chroma_client_collection_exists(uid):
            return Chroma(
                persist_directory=constants.CHROMA_PATH,
                embedding_function=self.embeddings,
                collection_name=uid
            )
        else:
            raise Exception("Embeddings haven't been built yet.")

    def _check_chroma_client_collection_exists(self, uid: str):
        """
            Check if chroma collection exists with the given uid
        """
        try:
            client = chromadb.PersistentClient(path=constants.CHROMA_PATH)
            collections: list[Collection] = client.list_collections()
            exists = any(c.name == uid for c in collections)
            return exists
        except Exception as e:
            print(f"Error checking chroma collection {e}")
            return False

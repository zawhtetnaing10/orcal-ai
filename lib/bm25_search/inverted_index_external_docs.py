from lib.bm25_search.inverted_index import InvertedIndex
import os
from lib.utils.text_utils import tokenize_text
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

import lib.utils.constants as constants
import lib.utils.data_loader_utils as data_loader_utils
import pickle


class InvertedIndexExternalDocs(InvertedIndex):
    def build(self, documents, uid: str):
        """
            Build and save the index for BM25 search
        """
        # Transform the documents. Add title before details in page content
        documents = self._transform_docs(documents)

        print(
            f"Successfully loaded json and created langchain docs")

        # Create the index
        index: BM25Retriever = BM25Retriever.from_documents(
            documents=documents,
            preprocess_func=tokenize_text,
            k=25
        )

        print(f"Indexing complete {index}")

        # Save the index into the pickle file
        try:
            with open(self._uid_to_file_path(uid), 'wb') as f:
                pickle.dump(index, f)
                print(f"Successfully saved the index")
        except Exception as e:
            print(f"Error saving the index. {e}")

    def _load_index(self, uid: str) -> BM25Retriever:
        """
            If the index hasn't been built yet, raise an Exception.
            If it's already built, just laod the index.
        """
        # If index does not exist, build it first.
        if not os.path.exists(self._uid_to_file_path(uid)):
            raise Exception(
                f"Index hasn't been built for Uid - {uid}. Please build the index first.")

        # Load the index and return
        try:
            with open(self._uid_to_file_path(uid), 'rb') as f:
                index = pickle.load(f)
                return index
        except Exception as e:
            print(f"Error loading the index {e}")

    def bm25_search(self, uid: str, query: str, limit: int):
        """
            Do a bm25 search from saved index.
        """
        retriever = self._load_index(uid=uid)

        # Set the top k results to the given limit.
        retriever.k = limit

        # Change the k value after testing
        docs = retriever.invoke(query)

        return docs

    def _uid_to_file_path(self, uid: str):
        return f"cache/{uid}.pkl"

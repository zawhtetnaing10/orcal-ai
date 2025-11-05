import os
from lib.utils.text_utils import tokenize_text
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_community.retrievers import BM25Retriever

import lib.utils.constants as constants
import pickle


def metadata_func(record: dict, metadata: dict) -> dict:
    """
        Extract id and title to use as metadata for the documents
    """
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    return metadata


class InvertedIndex:

    def build(self):
        """
            Build and save the index for BM25 search
        """
        try:

            # Load json and create documents
            loader = JSONLoader(
                file_path=constants.ABOUT_ME_FILE_PATH,
                jq_schema='.personal_info[]',
                content_key="details",
                metadata_func=metadata_func
            )
            documents = loader.load()

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
                with open(constants.INDEX_FILE_PATH, 'wb') as f:
                    pickle.dump(index, f)
                    print(f"Successfully saved the index")
            except Exception as e:
                print(f"Error saving the index. {e}")

        except FileNotFoundError:
            print(f"Cannot open file in f{constants.ABOUT_ME_FILE_PATH}")

    def _transform_docs(self, docs: list[Document]) -> list[Document]:
        """
            Transform the docs by adding the title before the details for page content.
            This is done for more accurate bm25 search results
        """
        result = []

        for doc in docs:
            title = doc.metadata.get("title", "")
            # Append title to details
            new_page_content = f"{title} {doc.page_content}"
            # Set new page content for the doc
            doc.page_content = new_page_content

            result.append(doc)

        return result

    def bm25_search(self, query: str, limit: int):
        """
            Do a bm25 search from saved index.
        """
        retriever = self._build_or_load_index()

        # Set the top k results to the given limit.
        retriever.k = limit

        # Change the k value after testing
        docs = retriever.invoke(query)

        return docs

    def _build_or_load_index(self) -> BM25Retriever:
        """
            If the index hasn't been built yet, build it first.
            Then load the index and return it.
        """
        # If index does not exist, build it first.
        if not os.path.exists(constants.INDEX_FILE_PATH):
            self.build()

        # Load the index and return
        try:
            with open(constants.INDEX_FILE_PATH, 'rb') as f:
                index = pickle.load(f)
                return index
        except Exception as e:
            print(f"Error loading the index {e}")

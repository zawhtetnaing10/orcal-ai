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
        try:

            # Load json and create documents
            loader = JSONLoader(
                file_path=constants.ABOUT_ME_FILE_PATH,
                jq_schema='.personal_info[]',
                content_key="details",
                metadata_func=metadata_func
            )
            documents = loader.load()

            print(
                f"Successfully loaded json and created langchain docs")

            # Create the index
            index = BM25Retriever.from_documents(
                documents=documents,
                preprocess_func=tokenize_text,
                k=5
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

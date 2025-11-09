from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader

from lib.utils.constants import ABOUT_ME_FILE_PATH


def metadata_func(record: dict, metadata: dict) -> dict:
    """
        Extract id and title to use as metadata for the documents
    """
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    return metadata


def load_about_me() -> list[Document]:
    """
        Load about_me.json and return the documents
    """
    # Load json and create documents
    loader = JSONLoader(
        file_path=ABOUT_ME_FILE_PATH,
        jq_schema='.personal_info[]',
        content_key="details",
        metadata_func=metadata_func
    )
    return loader.load()

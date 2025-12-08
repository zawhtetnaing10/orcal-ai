from langchain_core.documents import Document
from backend.models.build_embeddings_request import BuildEmbeddingsRequest

import time


def convert_build_embeddings_request_to_docs(request: BuildEmbeddingsRequest, uid: str) -> list[Document]:
    info_list = request.data

    result = []
    for data in info_list:
        page_content = data.details

        meta_data = {
            "id": data.id,
            "title": data.title,
            "uid": uid
        }

        document = Document(
            page_content=page_content,
        )
        document.metadata = meta_data

        result.append(document)

    return result


def create_turn_history_object(speaker: str, text: str) -> dict:
    """
        Creates Turn History object that LLM understands directly. 
        THe keys in the dict must not be changed.
    """
    item_dict = {}
    item_dict["role"] = speaker
    item_dict["parts"] = [
        {"text": text}
    ]
    return item_dict


def get_current_time_milliseconds() -> int:
    time_in_seconds = time.time()
    return int(round(time_in_seconds * 1000))

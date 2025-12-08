from langchain_core.documents import Document
from backend.models.build_embeddings_request import BuildEmbeddingsRequest


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
            meta_data=meta_data
        )

        result.append(document)

    return result

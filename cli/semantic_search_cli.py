import argparse

from lib.semantic_search.semantic_search import SemanticSearch
from langchain_core.documents import Document


def handle_semantic_chunk(semantic_search: SemanticSearch, text: str):
    """
        Handles semantic_chunk command
    """
    doc = Document(page_content=text)

    chunked_documents = semantic_search.semantic_chunk(
        document=doc, doc_idx=1, chunk_size=500, overlap=50)

    for index, chunked_doc in enumerate(chunked_documents):
        print(f"{index + 1}. {chunked_doc}")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Semantic Chunk
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk the given text and print out the results")
    semantic_chunk_parser.add_argument(
        "text", type=str, help="The text to chunk")

    # Semantic search object
    semantic_search = SemanticSearch()

    # Parse the args
    args = parser.parse_args()

    match args.command:
        case "semantic_chunk":
            handle_semantic_chunk(semantic_search, args.text)


if __name__ == "__main__":
    main()

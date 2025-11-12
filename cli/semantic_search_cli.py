import argparse

from lib.semantic_search.semantic_search import SemanticSearch
from langchain_core.documents import Document
import lib.utils.constants as constants


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


def handle_build_embeddings(semantic_search: SemanticSearch):
    """
        Handles build_embedding command
        Build embeddings for all documents inside about_me.json
    """
    semantic_search.build_embeddings()


def handle_semantic_search(semantic_search: SemanticSearch, query: str, limit: int):
    """
        Handles search command
    """
    docs = semantic_search.semantic_search(query, limit)

    for index, doc in enumerate(docs):
        print(f"{index + 1}. {doc}")
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

    # Build
    subparsers.add_parser(
        "build_embeddings", help="Build embeddings for the docs in about_me.json")

    # Search
    search_parser = subparsers.add_parser(
        "search", help="Run semantic search based on query.")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument(
        "--limit", type=int, default=constants.DEFAULT_ITEM_LIMIT, help="Number of items returned")

    # Semantic search object
    semantic_search = SemanticSearch()

    # Parse the args
    args = parser.parse_args()

    match args.command:
        case "semantic_chunk":
            handle_semantic_chunk(semantic_search, args.text)
        case "build_embeddings":
            handle_build_embeddings(semantic_search)
        case "search":
            handle_semantic_search(semantic_search, args.query, args.limit)


if __name__ == "__main__":
    main()

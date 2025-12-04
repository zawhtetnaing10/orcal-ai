import argparse
import nltk
from lib.bm25_search.inverted_index import InvertedIndex
from lib.bm25_search.inverted_index_external_docs import InvertedIndexExternalDocs

import lib.utils.data_loader_utils as data_loader_utils


def handle_download_nltk():
    """
        Downloads the necessary nltk files
    """
    nltk.download()


def handle_build(inv_index: InvertedIndex):
    """
        Handles build command. Builds the inverted index for BM25 search
    """
    inv_index.build()


def handle_build_external(inv_index: InvertedIndexExternalDocs, uid: str):
    """
        Handles build command. Documents and uid will be provided from an external source.
    """
    documents = data_loader_utils.load_about_me()
    inv_index.build(documents, uid)


def handle_search(inv_index: InvertedIndex, query: str, limit: int):
    """
        Handles search command. Searches the index for the given query and print out the results
    """
    results = inv_index.bm25_search(query, limit)
    for idx, result in enumerate(results):
        print("")
        print(f"{idx + 1}. {result.metadata['title']} - {result.metadata}")
        print(f"{result.page_content}")


def handle_search_external(inv_index: InvertedIndexExternalDocs, query: str, limit: int, uid: str):
    """
        Handles search command. Searches the index for the given query and print out the results.
        Documents and uid are provided externally.
    """
    results = inv_index.bm25_search(uid, query, limit)
    for idx, result in enumerate(results):
        print("")
        print(f"{idx + 1}. {result.metadata['title']} - {result.metadata}")
        print(f"{result.page_content}")


def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Download nltk
    subparsers.add_parser(
        "download_nltk", help="Downloads the necessary nltk data for tokenization")

    # Build
    subparsers.add_parser(
        "build", help="Build the inverted index for personal information")

    # Search
    bm25_search_parser = subparsers.add_parser(
        "search", help="Do a bm25 search ")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument(
        "--limit", type=int, help="Number of results.")

    # Build External
    build_external_parser = subparsers.add_parser(
        "build_external", help="Build the inverted index with documetns and uid provided from an external source.")
    build_external_parser.add_argument("uid", type=str, help="Uid of the user")

    # Search External
    bm25_search_external_parser = subparsers.add_parser(
        "search_external", help="Do a bm25 search with documents provided from external sources.")
    bm25_search_external_parser.add_argument(
        "query", type=str, help="Search query")
    bm25_search_external_parser.add_argument(
        "uid", type=str, help="Uid of the user")
    bm25_search_external_parser.add_argument(
        "--limit", type=int, help="Number of results.")

    # Keyword search object
    inv_index = InvertedIndex()
    inv_index_ext_docs = InvertedIndexExternalDocs()

    # Parse the args
    args = parser.parse_args()
    match args.command:
        case "download_nltk":
            handle_download_nltk()
        case "build":
            handle_build(inv_index)
        case "search":
            handle_search(inv_index, args.query, args.limit)
        case "build_external":
            handle_build_external(inv_index_ext_docs, args.uid)
        case "search_external":
            handle_search_external(
                inv_index_ext_docs, args.query, args.limit, args.uid)


if __name__ == "__main__":
    main()

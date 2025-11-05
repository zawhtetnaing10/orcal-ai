import argparse
import nltk
from lib.bm25_search.inverted_index import InvertedIndex


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


def handle_search(inv_index: InvertedIndex, query: str, limit: int):
    """
        Handles search command. Searches the index for the given query and print out the results
    """
    results = inv_index.bm25_search(query, limit)
    for idx, result in enumerate(results):
        print("")
        print(f"{idx + 1}. {result.metadata['title']}")
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

    # Keyword search object
    inv_index = InvertedIndex()

    # Parse the args
    args = parser.parse_args()
    match args.command:
        case "download_nltk":
            handle_download_nltk()
        case "build":
            handle_build(inv_index)
        case "search":
            handle_search(inv_index, args.query, args.limit)


if __name__ == "__main__":
    main()

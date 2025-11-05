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

    # Keyword search object
    inv_index = InvertedIndex()

    # Parse the args
    args = parser.parse_args()
    match args.command:
        case "download_nltk":
            handle_download_nltk()
        case "build":
            handle_build(inv_index)


if __name__ == "__main__":
    main()

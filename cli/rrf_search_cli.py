import argparse
from lib.hybrid_search.rrf_search import RRFSearch
from lib.hybrid_search.rrf_search_external_docs import RRFSearchExternalDocs

import lib.utils.data_loader_utils as data_loader_utils


def handle_rrf_search(rrf_search: RRFSearch, query: str, limit: int = 3):
    """
        Handles search command. Do a rrf search.
    """
    result = rrf_search.rrf_search(query=query, limit=limit)

    print("RRF search successful")
    for doc in result:
        print(f"{doc}")
        print("")


def handle_index_and_embeddings(rrf_search: RRFSearchExternalDocs, uid: str):
    """
        Builds embeddings and indices for uid
    """
    documents = data_loader_utils.load_about_me()
    rrf_search.build_embeddings_and_index(documents, uid)
    print(f"Successfully built embeddings and indices for uid - {uid}")


def handle_rrf_search_external(rrf_search: RRFSearchExternalDocs, uid: str, query: str, limit: int):
    """
        Handles rrf_search external command. Do a rrf search with given query and uid.
    """
    result = rrf_search.rrf_search(uid=uid, query=query, limit=limit)

    print("RRF search successful")
    for doc in result:
        print(f"{doc}")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Search
    rrf_search_parser = subparsers.add_parser(
        "search", help="Do a rrf search ")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "--limit", type=int, default=3, help="Number of results.")

    # Build Embeddings and index with external docs
    build_index_and_embeddings_parser = subparsers.add_parser(
        "build_index_and_embeddings", help="Build embeddings and index.")
    build_index_and_embeddings_parser.add_argument(
        "uid", type=str, help="Build embeddings and index for given documents and uid.")

    # Search with External Docs
    rrf_search_external_parser = subparsers.add_parser(
        "search_external", help="Do a rrf search with the provided query and uid")
    rrf_search_external_parser.add_argument(
        "query", type=str, help="Search query")
    rrf_search_external_parser.add_argument(
        "uid", type=str, help="Uid of the user.")
    rrf_search_external_parser.add_argument(
        "--limit", type=int, default=3, help="Number of results.")

    # RRF search object
    rrf_search = RRFSearch()
    rrf_search_external = RRFSearchExternalDocs()

    # Parse the args
    args = parser.parse_args()
    match args.command:
        case "search":
            handle_rrf_search(rrf_search, args.query, args.limit)
        case "build_index_and_embeddings":
            handle_index_and_embeddings(rrf_search_external, args.uid)
        case "search_external":
            handle_rrf_search_external(
                rrf_search=rrf_search_external, query=args.query, limit=args.limit, uid=args.uid)


if __name__ == "__main__":
    main()

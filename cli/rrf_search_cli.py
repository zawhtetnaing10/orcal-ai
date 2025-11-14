import argparse
from lib.hybrid_search.rrf_search import RRFSearch


def handle_rrf_search(rrf_search: RRFSearch, query: str, limit: int = 3):
    """
        Handles search command. Do a rrf search.
    """
    result = rrf_search.rrf_search(query=query, limit=limit)

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

    # RRF search object
    rrf_search = RRFSearch()

    # Parse the args
    args = parser.parse_args()
    match args.command:
        case "search":
            handle_rrf_search(rrf_search, args.query, args.limit)


if __name__ == "__main__":
    main()

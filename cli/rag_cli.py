import argparse
from lib.augmented_generation.rag import RAG


def handle_rag(rag: RAG, query: str):
    response = rag.discuss(query)
    print(f"AI Response:")
    print(response)


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Search
    discuss_parser = subparsers.add_parser(
        "discuss", help="Ask llm questions and discuss")
    discuss_parser.add_argument("query", type=str, help="Query from the user")

    # RRF search object
    rag = RAG()

    # Parse the args
    args = parser.parse_args()
    match args.command:
        case "discuss":
            handle_rag(rag, args.query)


if __name__ == "__main__":
    main()

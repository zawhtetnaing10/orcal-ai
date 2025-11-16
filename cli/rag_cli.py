import argparse
import sys
from lib.augmented_generation.rag import RAG


def handle_rag(rag: RAG):
    print("AI Assistant Online.....")
    while True:
        # Input
        sys.stdout.write("You: ")
        sys.stdout.flush()
        query = sys.stdin.readline().strip()

        # Response
        response = rag.discuss(query)
        print(f"Assistant: {response}")


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Search
    subparsers.add_parser(
        "discuss", help="Ask llm questions and discuss")

    # RRF search object
    rag = RAG()

    # Parse the args
    args = parser.parse_args()
    match args.command:
        case "discuss":
            handle_rag(rag)


if __name__ == "__main__":
    main()

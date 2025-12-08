import argparse
import sys
from lib.augmented_generation.rag import RAG
from lib.augmented_generation.rag import InMemoryTurnHistory
import lib.utils.constants as constants


def handle_rag(rag: RAG):
    print("AI Assistant Online.....")

    # Create the Turn History Object
    turn_history = InMemoryTurnHistory()

    while True:
        # Input
        sys.stdout.write("You: ")
        sys.stdout.flush()
        query = sys.stdin.readline().strip()

        # If the query is quit or exit, just exit. Do nothing
        if query.lower() == "exit" or query.lower() == "quit":
            print("Quitting the session.")
            exit(0)

        # Save the current query in Turn History
        turn_history.add_to_turn_history(
            speaker=constants.SPEAKER_USER, text=query)

        # Response
        response = rag.discuss(
            query, turn_history=turn_history.get_turn_history_str())

        # Add the response in Turn History
        turn_history.add_to_turn_history(
            speaker=constants.SPEAKER_MODEL, text=response)

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

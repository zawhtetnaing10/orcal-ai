from lib.hybrid_search.rrf_search import RRFSearch
import lib.utils.constants as constants
from lib.utils.genai_utils import generate_response
from lib.utils.prompt_utils import get_personal_assistant_rag_prompt


class RAG:
    def __init__(self):
        self.rrf_search = RRFSearch()

    def discuss(self, query: str, turn_history: str = "") -> str:
        """
            Do a rrf search based on query.
            Feed the results to LLM and get the result
        """
        # Do the search
        results = self.rrf_search.rrf_search(
            query, constants.DEFAULT_ITEM_LIMIT, constants.K_VALUE)

        # Get the prompt
        prompt = get_personal_assistant_rag_prompt(
            query=query, result=results, turn_history=turn_history)

        # Generate
        response = generate_response(prompt=prompt)

        return response.text


class TurnHistory:
    def __init__(self):
        # Turn history must be in chronological order. Latest messages must be at the end
        self.history: list[(str, str)] = []

    def add_to_turn_history(self, speaker: str, text: str):
        """
            Add item to turn history.
            If the turn history is already at the limit, remove the earliest one.
        """

        # If history is at the limit, remove the earliest one
        if len(self.history) >= constants.TURN_HISTORY_LIMIT:
            self.history.pop(0)

        # Create turn history object and save it in memory
        history_obj = self.create_turn_history_object(
            speaker=speaker, text=text)

        self.history.append(history_obj)

    def clear_turn_history(self):
        """
            Clears the current turn history.
            Should be called when switching conversations feature is implemented.
        """
        self.history.clear()

    def create_turn_history_object(self, speaker: str, text: str) -> dict:
        """
            Creates Turn History object that LLM understands directly. 
            THe keys in the dict must not be changed.
        """
        item_dict = {}
        item_dict["role"] = speaker
        item_dict["parts"] = [
            {"text": text}
        ]
        return item_dict

    def get_turn_history_str(self) -> str:
        return f"{self.history}"

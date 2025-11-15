from lib.hybrid_search.rrf_search import RRFSearch
import lib.utils.constants as constants
from lib.utils.genai_utils import generate_response
from lib.utils.prompt_utils import get_personal_assistant_rag_prompt


class RAG:
    def __init__(self):
        self.rrf_search = RRFSearch()

    def discuss(self, query: str) -> str:
        """
            Do a rrf search based on query.
            Feed the results to LLM and get the result
        """
        print("Discussing with LLM")

        # Do the search
        results = self.rrf_search.rrf_search(
            query, constants.DEFAULT_ITEM_LIMIT, constants.K_VALUE)

        # Get the prompt
        prompt = get_personal_assistant_rag_prompt(query=query, result=results)

        # Generate
        response = generate_response(prompt=prompt)

        return response.text

import os
from dotenv import load_dotenv
from google import genai
import lib.utils.constants as constants

load_dotenv()
gemini_client = None
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    print("API key not found. Check the environment carefully")
else:
    try:
        gemini_client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(
            f"Cannot initialize genai client: {e}")
        gemini_client = None


def get_client() -> genai.Client:
    """
        Return the already built gemini client
    """
    return gemini_client


def generate_response(prompt) -> str:
    """
        Generates resposne using a prompt
    """
    response = gemini_client.models.generate_content(
        model=constants.GEMINI_FLASH_MODEL, contents=prompt)
    return response.text

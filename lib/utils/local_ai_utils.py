from openai import AsyncOpenAI
from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def generate_local_llm_response(prompt) -> str:
    """
        Generates resposne using a prompt
    """
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="llama3.1",
        messages=messages
    )
    return response.choices[0].message.content

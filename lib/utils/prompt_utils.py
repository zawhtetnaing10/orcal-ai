def get_personal_assistant_rag_prompt(query: str, result: str, turn_history: str = "") -> str:
    """
    Generates a prompt for a personal assistant RAG system.

    This prompt guides the LLM to act as a knowledgeable personal assistant,
    answering queries based strictly on the provided context (from about_me.json).
    """

    # 1. System Instruction / Persona Setting
    # The initial block sets the rules for the model.
    # Hard constraint
    # system_instruction = (
    #     "You are a highly capable and concise Personal Assistant for the user. "
    #     "Your responses must be based **EXCLUSIVELY** on the 'PRIVATE CONTEXT' provided. "
    #     "Do not use external knowledge. "
    #     "If the answer is not contained within the context, you MUST politely state that the information is unavailable in the current knowledge base. "
    #     "Maintain a professional, helpful, and supportive tone."
    # )

    # Soft constraint
    system_instruction = (
        "You are a highly capable and concise Personal Assistant for the user. "
        "Your responses must be based on the 'PRIVATE CONTEXT' provided. "
        "Previous messages with you will be provided in 'TURN HISTORY' but it can also be empty "
        "Maintain a helpful, and supportive tone. "
        "Instead of just answering the question. Suggest the user on what they should do and try to solve their problem."
    )

    # 2. Main Prompt Structure
    # This structure clearly separates the context from the query for the model.
    full_prompt = (
        f"{system_instruction}\n\n"
        f"--- PRIVATE CONTEXT ---\n"
        f"{result}\n"
        f"--- TURN HISTORY ---\n"
        F"{turn_history}\n"
        f"-----------------------\n\n"
        f"USER QUERY:\n"
        f"{query}\n\n"
        f"ASSISTANT RESPONSE:"
    )

    return full_prompt

# prompt_builder.py

from memory import get_limited_conversation_history
from memory import SYSTEM_PROMPT

def build_final_prompt(user_id: str, retrieved_context: str, current_user_query: str) -> str:
    """
    Construct the final prompt for the LLM.
    """
    # 1. Start with system role
    prompt = f"SYSTEM: {SYSTEM_PROMPT}\n\n"

    # 2. Add previous conversation (short history)
    history = get_limited_conversation_history(user_id)
    for role, message in history:
        role_upper = role.upper()
        prompt += f"{role_upper}: {message}\n\n"

    # 3. Add retrieved document context
    prompt += f"CONTEXT:\n{retrieved_context}\n\n"

    # 4. Add the new user question
    prompt += f"USER: {current_user_query}\n\n"

    prompt += "ASSISTANT:"
    
    return prompt

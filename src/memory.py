SYSTEM_PROMPT = """
You are a helpful, knowledgeable AI chatbot for Bignalytics Coaching Institute, Indore.

Courses offered: Data Science, Machine Learning, AI, Python, Generative AI.
Answer questions related to courses, fees, placements, timings, and faculty.

Be supportive, precise, and polite.
If unsure, say "Sorry, I don't have that information."
"""

user_sessions = {}


MAX_HISTORY_TURNS = 3  # Only use last 3 user+assistant turns

def get_limited_conversation_history(user_id: str):
    """
    Fetch limited conversation history for building prompt.
    """
    history = user_sessions.get(user_id, [])
    # Skip 'system' prompt if present
    history = [turn for turn in history if turn[0] != "system"]
    return history[-(MAX_HISTORY_TURNS * 2):]  # (user, assistant) pairs

def initialize_user_session(user_id: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
        # Insert system prompt as first turn
        user_sessions[user_id].append(("system", SYSTEM_PROMPT))

def add_message_to_history(user_id: str, user_message: str, bot_response: str):
    if user_id not in user_sessions:
        initialize_user_session(user_id)
    user_sessions[user_id].append(("user", user_message))
    user_sessions[user_id].append(("assistant", bot_response))

def get_conversation_history(user_id: str):
    return user_sessions.get(user_id, [])

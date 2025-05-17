import tiktoken
import hashlib

def count_tokens(text: str, model_name="gpt2"):
    encoding = tiktoken.get_encoding(model_name)
    return len(encoding.encode(text))

def generate_user_id(ip: str, user_agent: str) -> str:
    raw_string = f"{ip}_{user_agent}"
    return hashlib.sha256(raw_string.encode()).hexdigest()[:16]

"""
query_transformer.py

Enhanced query transformation for Bignalytics chatbot with robust sub-question parsing and custom logging.

Key Improvements:
- Correct regex for topic detection and sub-question splitting.
- Fallback to original query if no sub-questions generated.
- Detailed debug and error logs around LLM invocations.
- Preserved custom logger setup.
- Environment-based configuration retained.
"""
import os
import re
import time
from typing import List, Dict
from functools import lru_cache

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from utils import count_tokens
from logger import log_pipeline_step, get_logger

# Initialize custom logger
logger = get_logger(__name__, "logs/query_transformer.log")

# Configuration via environment
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:1b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))  # currently not passed to Ollama

# Initialize Ollama LLM client

def init_llm() -> Ollama:
    return Ollama(
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0.0
    )

llm = init_llm()

# --- Prompt Templates ---
reword_prompts: Dict[str, PromptTemplate] = {
    "fee_structure": PromptTemplate.from_template(
        """
You are a professional assistant at Bignalytics Coaching Institute.
Reword the user query to clarify questions about course fees, discounts, and installments.
Original Query:
{original_query}

Reworded Query:
"""
    ),
    "placement_info": PromptTemplate.from_template(
        """
You are a professional assistant at Bignalytics.
Reword the user query to clarify questions about placement support, hiring companies, and salaries.
Original Query:
{original_query}

Reworded Query:
"""
    ),
    "course_overview": PromptTemplate.from_template(
        """
You are a professional assistant at Bignalytics.
Reword the user query to make questions about course syllabus, structure, and certification clearer.
Original Query:
{original_query}

Reworded Query:
"""
    ),
    "eligibility_criteria": PromptTemplate.from_template(
        """
You are a professional assistant at Bignalytics.
Reword the user query to clarify eligibility, prerequisites, and admission criteria.
Original Query:
{original_query}

Reworded Query:
"""
    ),
    "batch_timing": PromptTemplate.from_template(
        """
You are a professional assistant at Bignalytics.
Reword the user query to clarify batch timings, schedules, and start dates.
Original Query:
{original_query}

Reworded Query:
"""
    )
}

expand_prompt = PromptTemplate.from_template(
    """
You are an AI assistant for Bignalytics Educational Institute.
Expand the user's general query into 2–3 specific sub-questions about:
- Courses
- Fees
- Placements
- Batches
- Trainers
- Admission eligibility
- Discount offers

❌ Do NOT invent unrelated topics.
❌ Do NOT answer. Only expand.

✅ Sub-questions must be concise, specific, and faithful to the original query.

Example:
User Query: "Tell me about Data Science courses at Bignalytics"
Expanded Sub-Questions:
1. What Data Science courses are offered at Bignalytics?
2. What is the duration and structure of the Data Science courses?
3. What are the fees and any discounts for the Data Science courses?

Now expand the following:
Original User Query:
{original_query}

Expanded Sub-Questions:
1.
2.
3.
"""
)

# Correct regex patterns for topic detection
TOPIC_PATTERNS: Dict[str, re.Pattern] = {
    "fee_structure": re.compile(r"\b(fee|discount|installment|emi)s?\b", re.IGNORECASE),
    "placement_info": re.compile(r"\b(placement|job|hiring|salary)s?\b", re.IGNORECASE),
    "course_overview": re.compile(r"\b(course|program|syllabus|certification)s?\b", re.IGNORECASE),
    "eligibility_criteria": re.compile(r"\b(eligible|prerequisite|admission|requirement)s?\b", re.IGNORECASE),
    "batch_timing": re.compile(r"\b(batch|timing|schedule|start date)s?\b", re.IGNORECASE)
}

@lru_cache(maxsize=128)
def get_rewritten_query(original_query: str, mode: str) -> str:
    """Invoke the LLM chain; on failure, return the original query."""
    try:
        if mode == "expand":
            chain = LLMChain(llm=llm, prompt=expand_prompt)
            result = chain.run(original_query=original_query)
        else:
            prompt = reword_prompts.get(mode)
            if not prompt:
                logger.debug("No prompt found for mode '%s'; returning original query.", mode)
                return original_query
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(original_query=original_query)
        logger.debug("LLM output (mode=%s): %s", mode, result)
        return result
    except Exception as e:
        logger.error("LLM invocation error (mode=%s): %s", mode, str(e))
        return original_query


def detect_query_topic(user_query: str) -> str:
    """Detect the domain topic via regex patterns."""
    for topic, pattern in TOPIC_PATTERNS.items():
        if pattern.search(user_query):
            return topic
    return "general"


def is_query_specific(user_query: str) -> bool:
    """Return True if query matches any specific topic."""
    return detect_query_topic(user_query) != "general"


def rewrite_query_with_tracking(user_query: str, user_id: str) -> List[str]:
    """Rewrite or expand query; log tokens and duration; handle fallback."""
    logger.info("Starting transformation for user_id=%s", user_id)
    start_time = time.time()
    input_tokens = count_tokens(user_query)

    if is_query_specific(user_query):
        topic = detect_query_topic(user_query)
        logger.info("Rewording specific query; topic=%s", topic)
        raw = get_rewritten_query(user_query, topic)
    else:
        logger.info("Expanding general query into sub-questions.")
        raw = get_rewritten_query(user_query, "expand")

    output_tokens = count_tokens(raw)
    duration = time.time() - start_time
    log_pipeline_step("Query Transformation", user_query, input_tokens, output_tokens, duration, user_id=user_id)
    logger.debug("Raw transformed output: %s", raw)

    # Split out numbered sub-questions
    questions = split_rewritten_query(raw)
    if questions:
        return questions
    # Fallback
    logger.warning("No sub-questions found; falling back to original query.")
    return [user_query.strip()]


def split_rewritten_query(rewritten_query: str) -> List[str]:
    """Split '1. ... 2. ...' formatted text into a list of questions."""
    questions: List[str] = []
    for line in rewritten_query.splitlines():
        # Correct regex: match digit-dot-space then capture rest
        match = re.match(r"^\s*(\d+)\.\s*(.+)", line)
        if match:
            questions.append(match.group(2).strip())
    return questions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform user queries via LLM")
    parser.add_argument("query", type=str, help="Original user query")
    parser.add_argument("--user_id", type=str, default="unknown", help="User identifier")
    args = parser.parse_args()
    results = rewrite_query_with_tracking(args.query, args.user_id)
    for r in results:
        print(r)

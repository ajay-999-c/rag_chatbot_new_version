"""
Unified logging system for RAG application.

This module provides structured logging for both:
1. Application events (general system logs)
2. Performance metrics (for RAG system optimization)

It uses Python's logging module with custom formatters and handlers 
to ensure logs are properly organized and can be analyzed later.
"""

import logging
import os
from datetime import datetime
import csv
import json
from typing import Dict, List, Any, Optional, Union
import time
import uuid

# 1. Setup log directories if not exist
os.makedirs("logs", exist_ok=True)
os.makedirs("full_logs", exist_ok=True)
os.makedirs("logs/performance", exist_ok=True)

# Configure logging to prevent duplicate handlers
logging.basicConfig(level=logging.INFO)

# 2. Setup Logger Instances with proper configuration
def get_logger(name, log_file, log_level=logging.INFO, 
               console_level=logging.WARNING, json_format=False):
    """
    Create properly configured logger that won't duplicate logs
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level for file handler
        console_level: Logging level for console handler
        json_format: Whether to use JSON formatter (for performance logs)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Remove handlers if they exist to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(log_level)
    
    # File handler setup
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    
    if json_format:
        formatter = logging.Formatter("%(message)s")  # Raw message for JSON logs
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create loggers for different purposes
rag_logger = get_logger("rag_pipeline", "logs/rag_pipeline.log")
embedding_logger = get_logger("embedding_creation", "logs/embedding_creation.log")
performance_logger = get_logger("rag_performance", "logs/performance/rag_performance.log", 
                               json_format=True)

# --- CORE LOGGING CLASSES AND UTILITIES ---

def generate_request_id():
    """Generate a unique request ID for tracking a request across the pipeline"""
    return str(uuid.uuid4())

class RagTimer:
    """
    Utility class for timing RAG operations with support for nested timers
    
    This class is used as a context manager to time operations in the RAG pipeline.
    It can track both the operation time and additional metrics like token counts.
    
    Example:
        with RagTimer("embedding_generation", user_id="user123") as timer:
            # Do embedding operation
            timer.add_metadata("model", "all-MiniLM-L6-v2")
            timer.add_metadata("input_tokens", 512)
    """
    def __init__(self, step_name: str, user_id: Optional[str] = None, 
                 request_id: Optional[str] = None):
        self.step_name = step_name
        self.user_id = user_id
        self.request_id = request_id
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
        self.metadata = {}
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        # Log the timer data if we have a user_id
        if self.user_id and self.step_name:
            self.log_timer_data()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the timer for detailed analysis"""
        self.metadata[key] = value
    
    def log_timer_data(self) -> None:
        """Log the timer data with associated metadata"""
        metrics = {
            "step": self.step_name,
            "time_seconds": round(self.elapsed, 4),
            **self.metadata
        }
        
        query = self.metadata.get("query", "")
        log_performance_metrics(self.user_id, query, metrics, self.request_id)
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed

def log_event(message: str, level: str = "info", component: str = None, 
             request_id: Optional[str] = None):
    """
    Log a generic event or step inside the RAG pipeline system.
    Appends the message to logs/rag_pipeline.log with appropriate level.
    
    Args:
        message: The log message
        level: Log level (info, warning, error, debug, critical)
        component: Optional component name for better log organization
        request_id: Optional unique ID to track a request across the pipeline
    """
    # Format the log message with component and request_id if provided
    formatted_msg = message
    
    if component:
        formatted_msg = f"[{component}] {formatted_msg}"
    
    if request_id:
        formatted_msg = f"[Request: {request_id}] {formatted_msg}"
    
    # Log at the appropriate level
    if level.lower() == "info":
        rag_logger.info(formatted_msg)
    elif level.lower() == "warning":
        rag_logger.warning(formatted_msg)
    elif level.lower() == "error":
        rag_logger.error(formatted_msg)
    elif level.lower() == "debug":
        rag_logger.debug(formatted_msg)
    elif level.lower() == "critical":
        rag_logger.critical(formatted_msg)

def log_pipeline_step(
    step_name: str, 
    input_text: str, 
    input_tokens: int, 
    output_tokens: int, 
    time_taken: float, 
    section_type: Optional[str] = None, 
    retrieval_size: Optional[int] = None, 
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log a detailed structured step inside the pipeline, mentioning tokens, retrieval size, timing, etc.
    Appends info into logs/rag_pipeline.log.
    
    Args:
        step_name: Name of the pipeline step (query_rewrite, retrieval, generation, etc.)
        input_text: Input text for this step
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        time_taken: Time taken for this step in seconds
        section_type: Section/category filter used (if any)
        retrieval_size: Number of documents retrieved (if applicable)
        user_id: User identifier
        metadata: Additional metadata to log
    """
    # Basic information that's always logged
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "step": step_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "time_seconds": round(time_taken, 2),
        "section": section_type,
        "retrieved_docs": retrieval_size
    }
    
    # Add metadata if provided
    if metadata:
        log_data.update(metadata)
    
    # Format for plain text logging
    log_message = f"USER: {user_id} | STEP: {step_name} | INPUT TOKENS: {input_tokens} | OUTPUT TOKENS: {output_tokens} | TIME: {time_taken:.2f}s | SECTION: {section_type} | RETRIEVED DOCS: {retrieval_size} | INPUT: {input_text[:100]}..."
    
    # Log to both text and performance logs
    rag_logger.info(log_message)
    performance_logger.info(json.dumps(log_data))

def save_full_pipeline_log(log_data: Dict[str, Any], user_id: str):
    """
    Save full detailed RAG pipeline log for one user interaction.
    Appends into:
      - full_logs/full_pipeline_log.csv (all sessions together)
      - full_logs/full_pipeline_log.json (full session dump)
    
    Args:
        log_data: Dictionary containing all RAG pipeline data for this interaction
        user_id: User identifier
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- 1. Append to master CSV file ---
        csv_path = "full_logs/full_pipeline_log.csv"
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", encoding="utf-8", newline="") as csvfile:
            fieldnames = [
                "Timestamp", "User ID", "Question",
                "Rewritten Query", "Sub-Questions",
                "Chunks Retrieved", "Prompt Context",
                "Generated Answer", "Input Tokens", "Output Tokens", "Total Time (s)",
                "Section Type", "Model Name", "Success"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "Timestamp": timestamp,
                "User ID": user_id,
                "Question": log_data.get("user_query", ""),
                "Rewritten Query": log_data.get("rewritten_query", ""),
                "Sub-Questions": " | ".join(log_data.get("sub_questions", [])),
                "Chunks Retrieved": len(log_data.get("retrieved_chunks", [])),
                "Prompt Context": log_data.get("final_prompt", "")[:500],
                "Generated Answer": log_data.get("generated_answer", ""),
                "Input Tokens": log_data.get("input_tokens", 0),
                "Output Tokens": log_data.get("output_tokens", 0),
                "Total Time (s)": f"{log_data.get('total_time', 0):.2f}",
                "Section Type": log_data.get("section_type", "unknown"),
                "Model Name": log_data.get("model_name", "unknown"),
                "Success": log_data.get("success", True)
            })

        log_event(f"✅ Full CSV log updated at {csv_path}")

        # --- 2. Append to master JSON file ---
        json_path = "full_logs/full_pipeline_log.json"
        session_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "user_query": log_data.get("user_query", ""),
            "rewritten_query": log_data.get("rewritten_query", ""),
            "sub_questions": log_data.get("sub_questions", []),
            "retrieved_chunks": log_data.get("retrieved_chunks", []),
            "final_prompt": log_data.get("final_prompt", ""),
            "generated_answer": log_data.get("generated_answer", ""),
            "input_tokens": log_data.get("input_tokens", 0),
            "output_tokens": log_data.get("output_tokens", 0),
            "total_time_seconds": log_data.get("total_time", 0),
            "section_type": log_data.get("section_type", "unknown"),
            "model_name": log_data.get("model_name", "unknown"),
            "success": log_data.get("success", True),
            "pipeline_steps": log_data.get("pipeline_steps", {})
        }

        try:
            if not os.path.isfile(json_path):
                all_sessions = []
            else:
                with open(json_path, "r", encoding="utf-8") as jf:
                    try:
                        all_sessions = json.load(jf)
                    except json.decoder.JSONDecodeError:
                        all_sessions = []

            all_sessions.append(session_entry)

            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(all_sessions, jf, indent=2)

            log_event(f"✅ Full JSON log updated at {json_path}")
            
        except Exception as e:
            log_event(f"❌ Error saving JSON logs: {str(e)}", level="error")
            
    except Exception as e:
        log_event(f"❌ Error in save_full_pipeline_log: {str(e)}", level="error")

def log_embedding_event(message: str, level: str = "info"):
    """
    Log an embedding/vectorstore creation event.
    Appends the message to logs/embedding_creation.log.
    
    Args:
        message: The log message
        level: Log level (info, warning, error, debug, critical)
    """
    if level.lower() == "info":
        embedding_logger.info(message)
    elif level.lower() == "warning":
        embedding_logger.warning(message)
    elif level.lower() == "error":
        embedding_logger.error(message)
    elif level.lower() == "debug":
        embedding_logger.debug(message)
    elif level.lower() == "critical":
        embedding_logger.critical(message)

# --- COMBINED RAG PIPELINE LOGGING ---

class RagTracker:
    """
    Comprehensive tracking for an entire RAG pipeline request.
    
    This class manages both application logs and performance metrics
    for a complete RAG request from start to finish.
    
    It tracks:
    - Query processing steps
    - Retrieval performance
    - Token usage
    - Response generation time
    - Overall pipeline performance
    
    Example:
        tracker = RagTracker(user_id="user123", query="How does RAG work?")
        
        # Track query processing
        with tracker.track_step("query_rewrite"):
            rewritten_query = rewrite_query(tracker.query)
        
        # Track retrieval
        with tracker.track_step("retrieval") as retrieval_step:
            docs = retriever.retrieve(rewritten_query)
            retrieval_step.add_metadata("num_docs", len(docs))
        
        # Track LLM generation
        with tracker.track_step("llm_generation") as gen_step:
            response = llm.generate(docs, rewritten_query)
            gen_step.add_metadata("output_text", response)
        
        # Finalize and log complete pipeline
        tracker.finish(response)
    """
    def __init__(self, user_id: str, query: str, section_filter: Optional[str] = None):
        self.user_id = user_id
        self.query = query
        self.section_filter = section_filter
        self.request_id = generate_request_id()
        self.start_time = time.time()
        self.steps_data = {}
        self.pipeline_success = True
        
        # Log the start of the pipeline
        log_event(f"Started RAG pipeline for query: {query[:50]}...", 
                 component="RagTracker", request_id=self.request_id)
    
    def track_step(self, step_name: str):
        """
        Create a tracking context for a RAG pipeline step
        
        Returns:
            RagTimer: A context manager for timing this step
        """
        timer = RagTimer(step_name, self.user_id, self.request_id)
        timer.add_metadata("query", self.query)
        return timer
    
    def log_step_data(self, step_name: str, **kwargs):
        """
        Log data for a specific step without using context manager
        
        Args:
            step_name: Name of the pipeline step
            **kwargs: Additional data to log for this step
        """
        self.steps_data[step_name] = kwargs
        
        # Also log as a performance metric
        metrics = {
            "step": step_name,
            **kwargs
        }
        log_performance_metrics(self.user_id, self.query, metrics, self.request_id)
    
    def log_retrieval(self, docs, retrieval_time: float, retriever_type: str = "default"):
        """
        Log retrieval-specific metrics
        
        Args:
            docs: Retrieved documents
            retrieval_time: Time taken for retrieval
            retriever_type: Type of retrieval method used
        """
        metrics = {
            "step": "retrieval",
            "time_seconds": retrieval_time,
            "num_docs": len(docs),
            "retriever_type": retriever_type,
            "filter_used": self.section_filter is not None,
            "filter_value": self.section_filter
        }
        
        # If documents have metadata, gather statistics
        if docs and hasattr(docs[0], "metadata"):
            sources = set()
            sections = set()
            
            for doc in docs:
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
                if "section" in doc.metadata:
                    sections.add(doc.metadata["section"])
            
            metrics["unique_sources"] = len(sources)
            metrics["unique_sections"] = len(sections)
        
        self.log_step_data("retrieval", **metrics)
    
    def log_llm_generation(self, input_text: str, output_text: str, 
                          model_name: str, time_seconds: float):
        """
        Log LLM generation metrics including token counts
        
        Args:
            input_text: Full input prompt
            output_text: LLM generated response
            model_name: Name of the LLM model
            time_seconds: Generation time in seconds
        """
        from utils import count_tokens
        
        input_tokens = count_tokens(input_text)
        output_tokens = count_tokens(output_text)
        
        metrics = {
            "step": "llm_generation",
            "time_seconds": time_seconds,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tokens_per_second": output_tokens / time_seconds if time_seconds > 0 else 0
        }
        
        self.log_step_data("llm_generation", **metrics)
    
    def mark_error(self, error_message: str):
        """Mark the pipeline as having an error"""
        self.pipeline_success = False
        log_event(f"Error in RAG pipeline: {error_message}", 
                 level="error", component="RagTracker", request_id=self.request_id)
    
    def finish(self, final_response: Optional[str] = None):
        """
        Log the completion of the RAG pipeline with metrics
        
        Args:
            final_response: The final response generated by the RAG pipeline
        """
        total_time = time.time() - self.start_time
        
        # Create full pipeline log
        log_data = {
            "user_id": self.user_id,
            "request_id": self.request_id,
            "user_query": self.query,
            "section_type": self.section_filter,
            "total_time": total_time,
            "success": self.pipeline_success,
            "steps": self.steps_data
        }
        
        if final_response:
            log_data["response"] = final_response
        
        # Log final metrics
        metrics = {
            "step": "complete_pipeline",
            "time_seconds": total_time,
            "success": self.pipeline_success,
        }
        log_performance_metrics(self.user_id, self.query, metrics, self.request_id)
        
        # Save complete data to full pipeline log
        save_full_pipeline_log(log_data, self.user_id)
        
        log_event(f"Finished RAG pipeline in {total_time:.2f}s", 
                component="RagTracker", request_id=self.request_id)

# --- PERFORMANCE TRACKING FUNCTIONS ---

def log_performance_metrics(
    user_id: str,
    query: str,
    metrics: Dict[str, Any],
    request_id: Optional[str] = None
):
    """
    Log structured performance metrics for RAG components and pipeline.
    
    This function logs metrics in a structured JSON format for easy analysis.
    It creates daily rotating log files for performance metrics.
    
    Args:
        user_id: User identifier
        query: The query being processed
        metrics: Dictionary of performance metrics
        request_id: Optional unique request ID to track across the pipeline
    """
    # Create a structured log entry
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "request_id": request_id or generate_request_id(),
        "query": query[:100] + "..." if len(query) > 100 else query,
        **metrics
    }
    
    # Log to the performance logger
    performance_logger.info(json.dumps(log_data))
    
    # Also log to a daily performance file for easy analysis
    today = datetime.now().strftime("%Y-%m-%d")
    daily_log_path = f"logs/performance/performance_{today}.jsonl"
    
    try:
        with open(daily_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        rag_logger.error(f"Failed to write to daily performance log: {e}")

def log_token_metrics(
    user_id: str,
    request_id: str,
    step_name: str,
    input_text: str,
    output_text: Optional[str] = None,
    model_name: Optional[str] = None,
    time_seconds: Optional[float] = None
):
    """
    Log token-specific metrics for LLM interactions.
    
    Args:
        user_id: User identifier
        request_id: Unique request ID
        step_name: Name of the processing step
        input_text: Input text to count tokens for
        output_text: Optional output text to count tokens for
        model_name: Optional model name for token counting
        time_seconds: Optional time taken for this operation
    """
    from utils import count_tokens
    
    # Count tokens
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text) if output_text else 0
    
    # Create metrics
    metrics = {
        "step": step_name,
        "input_tokens": input_tokens,
        "input_chars": len(input_text),
        "time_seconds": time_seconds,
        "model": model_name,
    }
    
    if output_text:
        metrics.update({
            "output_tokens": output_tokens,
            "output_chars": len(output_text),
            "tokens_per_second": output_tokens / time_seconds if time_seconds else None
        })
    
    # Log the metrics
    log_performance_metrics(user_id, input_text[:50], metrics, request_id)

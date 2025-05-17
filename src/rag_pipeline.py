from query_transformer import rewrite_query_with_tracking, split_rewritten_query
from retriever import get_dynamic_retriever, retrieve_for_each_subquestion, merge_contexts
from generator import generator_chain
from memory import add_message_to_history, initialize_user_session
from utils import generate_user_id
from logger import log_event, save_full_pipeline_log
import time

def process_query_detailed(user_query: str, ip_address: str, user_agent: str, vectorstore_path="./chroma_db", db_type="chroma") -> dict:
    """
    Process a user query through the optimized RAG pipeline with ChromaDB
    """
    # Start timer for performance tracking
    start_time = time.time()
    
    # Track the user and create/get conversation history
    user_id = generate_user_id(ip_address, user_agent)
    chat_history = initialize_user_session(user_id)
    log_event(f"🔵 New query from user {user_id}")

    # Step 1: Query rewriting (enhance query for better retrieval)
    log_event("🔵 [Step 1] Starting query transformation...")
    # rewritten_query = rewrite_query_with_tracking(user_query, chat_history) # Temporarily disabled
    rewritten_query = user_query # Use original query
    sub_questions = [user_query] # Use original query as the only sub_question
    # sub_questions = split_rewritten_query(user_query) # Temporarily disabled
    print("---------------------------------")
    print(f"printing user query: {user_query}")
    print(f"printing rewritten query: {rewritten_query}")

    print(f"printing subquestion: {sub_questions}")
    print(f"type sub_questions: {type(sub_questions)}")
    print(f"type rewritten_query: {type(rewritten_query)}")
    print("---------------------------------")

    log_event(f"✅ [Step 1] Generated {len(sub_questions)} sub-questions")

    # Step 2: Retrieval with dynamic filtering based on query content
    log_event(f"🔵 [Step 2] Starting retrieval process...")
    retriever, query_category = get_dynamic_retriever(user_query, vectorstore_path=vectorstore_path, db_type=db_type)
    retrieved_docs = retrieve_for_each_subquestion(sub_questions, retriever, top_k=5)  
    log_event(f"✅ [Step 2] Retrieved {len(retrieved_docs)} documents with category: {query_category}")

    # Step 3: Context preparation and generation
    log_event("🔵 [Step 3] Preparing context and generating response...")
    context = merge_contexts(retrieved_docs)
    
    # Invoke the generator chain
    # Assuming generator_chain.invoke returns the generated text directly as a string
    print(f"printing user query: {user_query}")

    print(F"printing context: {context}")
    llm_output_string = generator_chain.invoke({
        "context": context, 
        "question": user_query,
        "chat_history": chat_history
    })
    log_event("✅ [Step 3] Response generated successfully")

    # Update conversation history with the raw LLM output string
    add_message_to_history(user_id, "user", user_query)
    add_message_to_history(user_id, "assistant", llm_output_string) # Use the string here
    
    # Calculate total time for performance monitoring
    processing_time = time.time() - start_time
    log_event(f"⏱️ Total processing time: {processing_time:.2f} seconds")
    
    # Prepare the detailed log for saving and for returning to app.py
    # This structure should align with what app.py expects for display
    # and what logger.py expects for saving.
    pipeline_output_dict = {
        "user_id": user_id,
        "query": user_query, # app.py uses this as 'query' in last_result, logger expects 'user_query'
        "user_query": user_query, # Explicitly add for logger.py
        "rewritten_query": user_query, # Set to user_query as transformation is disabled
        "sub_questions": sub_questions, # Reflects that only user_query is used
        "section_type": query_category, 
        "retrieved_chunks": [doc.page_content for doc in retrieved_docs if hasattr(doc, 'page_content')],
        "final_prompt": context,
        "generated_answer": llm_output_string, # This is the key app.py uses
        "total_time": processing_time,
        "input_tokens": "N/A", # Placeholder, update if available
        "output_tokens": "N/A", # Placeholder, update if available
        # Additional fields for detailed logging, not necessarily used by app.py display
        "num_retrieved_docs": len(retrieved_docs),
        "context_length": len(context),
        "response": llm_output_string, # Keep 'response' for logger.py if it expects it and for add_message_to_history backward compatibility if needed
    }
    
    # Save the structured log - corrected argument order
    save_full_pipeline_log(pipeline_output_dict, user_id)
    
    # Return the dictionary that app.py will use
    return pipeline_output_dict

def process_query(user_query: str, ip_address: str, user_agent: str, vectorstore_path="./chroma_db", db_type="chroma") -> str:
    """
    Simplified interface that returns just the response string
    """
    result = process_query_detailed(user_query, ip_address, user_agent, vectorstore_path, db_type=db_type)
    return result["generated_answer"] # Ensure this key exists in the returned dict

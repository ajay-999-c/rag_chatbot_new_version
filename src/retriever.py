# retriever.py

import numpy as np
import streamlit as st
from vectorstore import load_vectorstore, load_embedding_model

# 1. Predefined Section Labels
SECTION_LABELS = [
    "fee_structure",
    "placement_info",
    "course_overview",
    "hands_on_training",
    "eligibility_criteria",
    "course_duration",
    "batch_timing",
    "discount_offer",
    "general_info"
]

# 2. Cache embedding model and label embeddings
@st.cache_resource(show_spinner=False)
def get_embedding_model_and_label_embeddings():
    print("🔵 [System] Loading embedding model and computing section label embeddings...")
    embedding_model = load_embedding_model()
    label_embeddings = embedding_model.embed_documents(SECTION_LABELS)
    print(f"✅ [System] Embedding model loaded. {len(SECTION_LABELS)} section labels embedded.\n")
    return embedding_model, label_embeddings

# 3. Load once at startup
embedding_model, label_embeddings = get_embedding_model_and_label_embeddings()

# 4. Classify query into a section
def classify_query_by_embedding(user_query: str) -> str:
    print(f"🔵 [Classifier] Classifying query: {user_query}")
    query_embedding = embedding_model.embed_query(user_query)

    sims = np.dot(label_embeddings, query_embedding) / (
        np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    best_idx = np.argmax(sims)
    predicted_section = SECTION_LABELS[best_idx]

    print(f"✅ [Classifier] Predicted Section: {predicted_section} (Similarity: {sims[best_idx]:.4f})\n")
    return predicted_section

# 5. Load retriever based on section
def get_dynamic_retriever(user_query: str, vectorstore_path="./chroma_db", db_type="chroma"):
    print(f"🔵 [Retriever] Loading vectorstore from: {vectorstore_path}...")
    vectorstore = load_vectorstore(vectorstore_path)

    if vectorstore is None:
        error_message = f"Vectorstore not found at {vectorstore_path}. Please build it first by running the build_chroma_vectorstore.py script."
        print(f"❌ [Error] {error_message}")
        # Raising an exception to halt execution and clearly indicate the problem.
        # Streamlit will catch this and display an error to the user.
        raise FileNotFoundError(error_message)
    
    print(f"✅ [Retriever] Vectorstore loaded successfully.")

    section_type = classify_query_by_embedding(user_query)

    if db_type == "chroma":
        search_filter = {"section_type": section_type} if section_type != "general_info" else None # Pass None if no filter
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": search_filter
            }
        )
        print(f"✅ [Retriever] Dynamic retriever created with section filter: {section_type if search_filter else 'None'}\n")
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        print(f"✅ [Retriever] Simple retriever created without section filtering.\n")

    return retriever, section_type

# 6. Retrieve for each sub-question
def retrieve_for_each_subquestion(sub_questions, retriever, top_k=5):
    print(f"🔵 [Retrieval] Starting retrieval for {len(sub_questions)} sub-questions...")
    all_retrieved_docs = []
    unique_questions = list(set(sub_questions))  # Remove duplicates to avoid redundant queries
    print(f"🔍 Optimized: Reduced to {len(unique_questions)} unique questions from {len(sub_questions)} total")
    
    for idx, sub_q in enumerate(unique_questions, 1):
        retrieved = retriever.get_relevant_documents(sub_q)[:top_k]  # Actually use the top_k parameter
        print(f"✅ Retrieved {len(retrieved)} documents for sub-question {idx}: '{sub_q}'")
        all_retrieved_docs.extend(retrieved)

    # Remove duplicate documents based on content hash
    unique_docs = []
    seen_contents = set()
    for doc in all_retrieved_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_docs.append(doc)
    
    print(f"✅ [Retrieval] Optimized: {len(unique_docs)} unique documents from {len(all_retrieved_docs)} total retrieved.\n")
    return unique_docs

# 7. Merge retrieved contexts into a single text
def merge_contexts(docs):
    print(f"🔵 [Merging] Merging {len(docs)} retrieved documents into context...")
    
    # Sort documents by relevance score if available
    if docs and hasattr(docs[0], "metadata") and "score" in docs[0].metadata:
        docs = sorted(docs, key=lambda x: x.metadata.get("score", 0), reverse=True)
        print(f"🔍 Sorted documents by relevance score")
    
    merged_text = "\n\n".join(doc.page_content for doc in docs)
    print(f"✅ [Merging] Merging complete. Context size: {len(merged_text)} characters\n")
    return merged_text

# app.py

import streamlit as st
import time
from rag_pipeline import process_query_detailed
from utils import generate_user_id
from memory import initialize_user_session, get_conversation_history

# --- Streamlit Settings ---
st.set_page_config(page_title="Bignalytics RAG Chatbot 🔎", page_icon="🔎", layout="wide")

# --- Main Title ---
st.title("🔎 Bignalytics RAG Chatbot")
st.caption("Ask anything about Bignalytics courses, fees, placement support, batches!")

# Dummy IP/User-Agent (Streamlit limitation)
ip_address = "127.0.0.1"
user_agent = "Streamlit-Test-User"

# Initialize user session
if "user_id" not in st.session_state:
    st.session_state.user_id = generate_user_id(ip_address, user_agent)
    initialize_user_session(st.session_state.user_id)

# Layout Split: Chat (left) + Process Details (right)
col1, col2 = st.columns([3, 1])

# 💬 Left Side: Chat Interface
with col1:
    st.header("💬 Your BigAnalytics Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Type your question:", key="user_input")

    if st.button("Ask"):
        if user_query.strip():
            # Save user query
            st.session_state.chat_history.append(("user", user_query))

            with st.spinner("🔄 Processing your query..."):
                # Full RAG Processing
                result = process_query_detailed(
                    user_query, ip_address, user_agent,
                    vectorstore_path="./chroma_db", db_type="chroma"
                )

                st.session_state.last_result = result
                st.session_state.chat_history.append(("bot", result["generated_answer"]))

        else:
            st.warning("⚠️ Please type a question before submitting.")

    # Display Chat History
    for role, message in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(f"**You:** {message}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Bignalytics Bot:** {message}")

# 📊 Right Side: Metadata and Debugging
with col2:
    st.header("📊 Processing Steps")

    if "last_result" in st.session_state:
        result = st.session_state.last_result

        # Step 1: Query Rewriting
        with st.expander("1️⃣ Query Rewriting"):
            st.markdown(f"**Rewritten Query:**\n\n{result['rewritten_query']}")

        # Step 2: Sub-Question Expansion
        with st.expander("2️⃣ Sub-Questions"):
            for idx, q in enumerate(result["sub_questions"]):
                st.markdown(f"**{idx+1}.** {q}")

        # Step 3: Retrieval Details
        with st.expander("3️⃣ Retrieved Chunks"):
            st.markdown(f"**Chunks Retrieved:** {len(result['retrieved_chunks'])}")
            for idx, chunk in enumerate(result["retrieved_chunks"][:3]):
                st.markdown(f"**Chunk {idx+1}:** {chunk[:300]}...")

        # Step 4: Final Prompt (Context Sent to LLM)
        with st.expander("4️⃣ Final Prompt Context"):
            st.markdown(f"**Prompt to LLM:**\n\n{result['final_prompt'][:500]}...")  # Show first 500 chars

        # Step 5: Generated Answer
        with st.expander("5️⃣ Final Answer"):
            st.markdown(f"**Generated Answer:**\n\n{result['generated_answer']}")

        # Step 6: Performance Metrics
        with st.expander("6️⃣ Performance Metrics"):
            st.markdown(f"**Input Tokens:** {result['input_tokens']}")
            st.markdown(f"**Output Tokens:** {result['output_tokens']}")
            st.markdown(f"**Total Time Taken:** {result['total_time']:.2f} seconds")

        # Step 7: Section Type Predicted
        with st.expander("7️⃣ Section Classification"):
            st.markdown(f"**Predicted Section:** {result['section_type']}")

    else:
        st.info("ℹ️ Ask your first question to view all processing details.")

# vectorstore.py - Optimized for ChromaDB with CSV data

import os
import time
from langchain_chroma import Chroma # Updated import
from embedding import load_embedding_model
from logger import log_embedding_event

def create_chroma_vectorstore(documents, persist_path="./chroma_db"):
    """
    Create a ChromaDB vectorstore optimized for FAQ data from CSV.
    With newer langchain-chroma, client_settings and collection_metadata are often
    simplified or handled by default, or through the underlying chromadb client if needed.
    """
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)

    embedding_function = load_embedding_model()
    start = time.time()
    
    # Create vectorstore. persist_directory handles saving.
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_path
        # For more advanced Chroma settings, you might initialize a chromadb.PersistentClient first
        # and pass it to the Chroma constructor, or use specific parameters if available.
        # Example for advanced settings (if needed, consult langchain-chroma docs):
        # client = chromadb.PersistentClient(path=persist_path, settings=Settings(...))
        # vectordb = Chroma(
        #     client=client,
        #     collection_name="your_collection_name",
        #     embedding_function=embedding_function,
        # )
        # vectordb.add_documents(documents) # Then add documents
    )
    
    # Persist is usually handled by from_documents with persist_directory
    # vectordb.persist() # This might be redundant
    end = time.time()

    log_embedding_event(f"✅ Chroma vectorstore built successfully at {persist_path} in {end-start:.2f} seconds.")
    print(f"✅ ChromaDB vectorstore created with {len(documents)} documents in {end-start:.2f} seconds")
    return vectordb

def load_vectorstore(persist_path="./chroma_db"):
    """
    Load ChromaDB vectorstore.
    """
    embedding_function = load_embedding_model()
    print(f"🔵 Loading Chroma vectorstore from {persist_path}...")
    
    if not os.path.exists(persist_path):
        print(f"⚠️ Warning: Vectorstore path {persist_path} does not exist!")
        return None
        
    # Initialize Chroma with the persist_directory and embedding_function
    vectordb = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_function
        # client_settings are generally not passed directly here in newer versions.
        # If specific client settings are needed, you'd typically initialize a chromadb client separately.
    )
    
    # A way to check if the collection is loaded, count might vary based on direct client usage
    # This is a common way to try and access the underlying collection count.
    try:
        doc_count = vectordb._collection.count()
        print(f"✅ ChromaDB loaded successfully with {doc_count} documents!")
    except Exception as e:
        print(f"✅ ChromaDB loaded. Could not get document count directly (normal for some setups or empty DB): {e}")
        # If the above fails, it might be that the collection needs to be explicitly named or client handled differently
        # For a simple load, if from_documents was used to save, this should generally work if data exists.

    return vectordb

def create_vectorstore_from_documents(documents, persist_path="./chroma_db"):
    """
    Process documents and create ChromaDB vectorstore with optimized chunking for FAQ data
    """
    print(f"✅ Starting vectorstore creation...")
    total_start = time.time()

    # For CSV FAQ data, we can generally skip the chunking step as each QA pair 
    # is already a good-sized unit for retrieval
    print("✅ Preparing documents with metadata...")
    start = time.time()
    
    # Make sure each document has the right metadata
    for doc in documents:
        if "section_type" not in doc.metadata:
            doc.metadata["section_type"] = "general_info"
    
    end = time.time()
    print(f"✅ Documents prepared. ⏱️ {end-start:.2f} seconds")

    # Create the vectorstore
    vectordb = create_chroma_vectorstore(documents, persist_path)

    total_end = time.time()
    print(f"🎯 Total Time Taken: {total_end-total_start:.2f} seconds")
    return vectordb

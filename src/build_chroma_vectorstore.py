import pandas as pd
from langchain_chroma import Chroma # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time
import json
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("build_chromadb")

# Start timing
start_time = time.time()

def get_section_stats(df):
    """Get statistics on section tags for logging"""
    section_counts = Counter(df['Tagging'].str.strip())
    return {
        "total_sections": len(section_counts),
        "section_distribution": section_counts
    }

# 1. Load the tagged FAQ CSV
try:
    df = pd.read_csv("data/final_faq_ba.csv")
    logger.info(f"✅ Loaded {len(df)} FAQs.")
    
    # Log section statistics
    stats = get_section_stats(df)
    logger.info(f"Found {stats['total_sections']} unique section tags")
    logger.info(f"Section distribution: {json.dumps(stats['section_distribution'], indent=2)}")
except Exception as e:
    logger.error(f"Failed to load CSV: {e}")
    raise

# 2. Prepare Chunks and Metadata
texts = []
metadatas = []

for idx, row in df.iterrows():
    question = str(row['Question']).strip()
    reply = str(row['Reply']).strip()
    tag = str(row['Tagging']).strip()
    
    if not tag:
        logger.warning(f"Missing section tag for row {idx}, defaulting to 'Unclassified'")
        tag = "Unclassified"
    
    combined_text = f"Question: {question}\nAnswer: {reply}"
    texts.append(combined_text)

    # Enhanced metadata with additional fields
    metadata = {
        "section": tag,
        "question": question,
        "doc_id": f"qa_{idx}",
        "source_type": "faq"
    }
    metadatas.append(metadata)

logger.info(f"✅ Prepared {len(texts)} chunks and metadata.")

# Save metadata statistics for future reference
try:
    section_stats_path = "./data/section_stats.json"
    os.makedirs(os.path.dirname(section_stats_path), exist_ok=True)
    
    sections = [meta["section"] for meta in metadatas]
    section_counts = Counter(sections)
    
    with open(section_stats_path, 'w') as f:
        json.dump({
            "total_documents": len(texts),
            "total_sections": len(section_counts),
            "section_distribution": {k: v for k, v in section_counts.items()},
            "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    logger.info(f"Section statistics saved to {section_stats_path}")
except Exception as e:
    logger.warning(f"Could not save section statistics: {e}")

# 3. Generate Embeddings
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ HuggingFace embedding model loaded.")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise

# 4. Create Chroma Vectorstore
persist_directory = "./chroma_db"

# Ensure the directory exists
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    logger.info(f"Created directory: {persist_directory}")

try:
    embedding_start = time.time()
    # Chroma client initialization for creating a new DB
    # The persist_directory argument handles saving.
    # No explicit client_settings needed here for basic creation with from_texts
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    embedding_time = time.time() - embedding_start
    logger.info(f"✅ Generated embeddings in {embedding_time:.2f} seconds")
    
    # Persisting is generally handled by from_texts when persist_directory is set,
    # but an explicit call doesn't harm.
    # vectorstore.persist() # This line is often redundant now but kept for safety/explicitness
    logger.info(f"✅ Vectorstore created and persisted successfully at {persist_directory}")
    
except Exception as e:
    logger.error(f"Failed to create vectorstore: {e}")
    raise

end_time = time.time()
total_time = end_time - start_time
logger.info(f"🎯 Vectorstore created successfully! Total time: {total_time:.2f} seconds.")
logger.info(f"📂 Chroma DB saved at: {persist_directory}")

# 6. Verify database contents by getting collection info
try:
    collection = vectorstore._collection
    collection_stats = {
        "collection_name": collection.name,
        "collection_count": collection.count(),
        "metadata_schema": {k: type(v).__name__ for k, v in metadatas[0].items() if metadatas}
    }
    logger.info(f"📊 Collection verification: {json.dumps(collection_stats, indent=2)}")
    
    # Check if sections are properly indexed
    sample_section = metadatas[0]["section"] if metadatas else None
    if sample_section:
        try:
            # Similarity search remains the same
            results = vectorstore.similarity_search(
                "test query", 
                k=1, 
                filter={"section": sample_section}
            )
            logger.info(f"✅ Section filtering works. Sample section '{sample_section}' is properly indexed.")
        except Exception as e:
            logger.warning(f"⚠️ Section filtering test failed: {e}")
    
except Exception as e:
    logger.warning(f"Could not verify database contents: {e}")
    
logger.info("Build process completed successfully!")
print(f"📂 Chroma DB saved at: {persist_directory}")


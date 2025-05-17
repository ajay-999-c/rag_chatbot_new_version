
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.pydantic_v1 import Field
import math
import os
from dotenv import load_dotenv

load_dotenv()

class BatchingHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    batch_size: int = Field(default=32, description="Batch size for embeddings.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents for efficiency."""
        all_embeddings = []
        total = len(texts)
        num_batches = math.ceil(total / self.batch_size)

        print(f"⚡ Total {total} chunks to embed in {num_batches} batches of {self.batch_size} each.")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, total)
            batch_texts = texts[batch_start:batch_end]

            batch_embeddings = super().embed_documents(batch_texts)

            all_embeddings.extend(batch_embeddings)
            print(f"✅ Embedded batch {batch_idx+1}/{num_batches} [{batch_start}-{batch_end}]")

        return all_embeddings

def load_embedding_model():
    # Get model name from environment variable or use default
    model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    return BatchingHuggingFaceEmbeddings(
        model_name=model_name,  # You can also try "BAAI/bge-small-en-v1.5"
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True},  # Normalize for better similarity search
        batch_size=32  # Optimal batch size will depend on your hardware
    )

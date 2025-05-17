from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def detect_section_type(text: str) -> str:
    text_lower = text.lower()
    if "fee" in text_lower or "pricing" in text_lower:
        return "fee_structure"
    elif "placement" in text_lower or "hiring" in text_lower:
        return "placement_info"
    elif "overview" in text_lower or "syllabus" in text_lower:
        return "course_overview"
    elif "lab" in text_lower or "hands-on" in text_lower:
        return "hands_on_training"
    elif "eligibility" in text_lower or "prerequisite" in text_lower:
        return "eligibility_criteria"
    elif "duration" in text_lower:
        return "course_duration"
    elif "batch" in text_lower or "timing" in text_lower:
        return "batch_timing"
    elif "discount" in text_lower or "offer" in text_lower:
        return "discount_offer"
    else:
        return "general_info"

def chunk_documents_with_metadata(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_doc = Document(
                page_content=chunk,
                metadata={**doc.metadata, "section_type": detect_section_type(chunk)}
            )
            chunked_docs.append(chunked_doc)
    return chunked_docs

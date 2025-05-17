from langchain_community.document_loaders import CSVLoader

def load_csv(file_path: str):
    """
    Load and process a CSV file with proper metadata tagging
    """
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    
    # Add important metadata to each document for better filtering capabilities
    for doc in docs:
        doc.metadata["source_file"] = file_path
        doc.metadata["document_type"] = "csv"
        
        # Try to identify the content type from the question for better retrieval
        question_text = doc.page_content.split("Question:")[1].split("Reply:")[0] if "Question:" in doc.page_content else ""
        
        if "fee" in question_text.lower() or "discount" in question_text.lower() or "cost" in question_text.lower():
            doc.metadata["section_type"] = "fee_structure"
        elif "placement" in question_text.lower() or "job" in question_text.lower() or "career" in question_text.lower():
            doc.metadata["section_type"] = "placement_info"
        elif "course" in question_text.lower() or "training" in question_text.lower():
            doc.metadata["section_type"] = "course_overview"
        elif "hands-on" in question_text.lower() or "practical" in question_text.lower() or "project" in question_text.lower():
            doc.metadata["section_type"] = "hands_on_training"
        elif "eligibility" in question_text.lower() or "requirement" in question_text.lower():
            doc.metadata["section_type"] = "eligibility_criteria"
        elif "duration" in question_text.lower() or "time" in question_text.lower() or "long" in question_text.lower():
            doc.metadata["section_type"] = "course_duration"
        elif "batch" in question_text.lower() or "timing" in question_text.lower() or "schedule" in question_text.lower():
            doc.metadata["section_type"] = "batch_timing"
        elif "discount" in question_text.lower() or "offer" in question_text.lower() or "scholarship" in question_text.lower():
            doc.metadata["section_type"] = "discount_offer"
        else:
            doc.metadata["section_type"] = "general_info"
    
    print(f"✅ Loaded {len(docs)} entries from CSV: {file_path}")
    return docs

def load_all_data(csv_paths: list):
    """
    Load data from multiple CSV files - simplified to focus only on CSV processing
    """
    all_docs = []
    for csv_path in csv_paths:
        try:
            all_docs.extend(load_csv(csv_path))
            print(f"✅ Successfully processed: {csv_path}")
        except Exception as e:
            print(f"❌ Error processing {csv_path}: {str(e)}")
    
    print(f"📊 Total documents loaded: {len(all_docs)}")
    return all_docs

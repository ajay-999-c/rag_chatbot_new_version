

from load_data import load_all_data
from vectorstore import create_vectorstore_from_documents
from logger import log_event  # we already have logger.py to save errors
import os

def validate_file_paths(file_paths: list, file_type: str):
    """
    Validate if given file paths exist and are correct.
    Logs error if file is missing.
    """
    valid_paths = []
    for path in file_paths:
        if os.path.exists(path) and os.path.isfile(path):
            valid_paths.append(path)
        else:
            log_event(f"❌ ERROR: {file_type} file not found: {path}")
            print(f"❌ WARNING: {file_type} file not found: {path}")  # Also show on console
    return valid_paths

def main():
    # Your Dataset paths (Update here)
    pdf_files = ["..\data\Dataset.pdf"]
    csv_files = ["..\data\FINAL-FAQ-FIXED.csv"]

    # Step 1: Validate files
    valid_pdfs = validate_file_paths(pdf_files, file_type="PDF")
    valid_csvs = validate_file_paths(csv_files, file_type="CSV")

    if not valid_pdfs and not valid_csvs:
        print("❌ No valid files found. Exiting...")
        log_event("❌ No valid input files found. Vectorstore creation aborted.")
        return

    # Step 2: Load data
    try:
        all_documents = load_all_data(valid_pdfs, valid_csvs)
    except Exception as e:
        log_event(f"❌ ERROR loading documents: {str(e)}")
        print(f"❌ Failed to load documents: {e}")
        return

    # Step 3: Create vectorstore
    try:
        create_vectorstore_from_documents(all_documents, save_path="./faiss_index")
        print("✅ Vectorstore created and saved at ./faiss_index")
        log_event("✅ Vectorstore successfully created.")
    except Exception as e:
        log_event(f"❌ ERROR creating vectorstore: {str(e)}")
        print(f"❌ Failed to create vectorstore: {e}")

if __name__ == "__main__":
    main()

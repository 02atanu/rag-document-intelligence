import os
import tempfile
import hashlib
import re
from datetime import datetime, timezone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """Extra whitespace, newlines, messy pdf character removal"""
    # Replacing whitespace characters 
    text = re.sub(r'\s+',' ', text)
    return text.strip()

def process_pdf(file_bytes: bytes, filename: str) -> list:
    """saves uploaded bytes, extracts text and cleans it and finally chunks it"""
    # Creates a temporary file in memory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        for doc in documents:
            doc.page_content = clean_text(doc.page_content) #Clean before chunking

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len) #Chunking
        chunks = text_splitter.split_documents(documents)

        upload_date = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(f'{filename}_{chunk.page_content}'.encode('utf-8')).hexdigest() # Create deterministic id using filename and chnk text
        
            chunk.metadata.update({ # update the document with metadata payload
                'source': filename,
                'chunk_index': i,
                'upload_date': upload_date,
                'chunk_hash': chunk_hash,
            })
        return chunks
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
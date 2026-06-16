import os
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")  # This is the crucial one for Docker!
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "rag_db")

CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "pdf_documents"

def get_vector_store() -> PGVector:
    """Initializes connection to postgreSQL and PGVector"""

    embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', api_key=os.getenv('GEMINI_API_KEY'))

    vector_store = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING, use_jsonb=True)

    return vector_store

def save_chunks_to_db(chunks: list):
    """Embades the chunks and saves them to postgres database"""
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)


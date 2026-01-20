import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Config:
    """Configuration settings for the RAG engine."""
    CHUNK_SIZE = 800                               # size of text chunks
    CHUNK_OVERLAP = 150                            # overlap between chunks
    TOP_K = 7                                      # number of top documents to retrieve
    FETCH_K = 20                                   # number of candidate documents to consider during retrieval
    Model_Name = "llama-3.3-70b-versatile"         # language model name
    Persist_Dir = "./../../chroma_db_lilianweng"         # directory for persistent storage
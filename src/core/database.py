import os
import bs4
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from src.config import Config

# Init embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def get_vectorstore():
    """Load or create the vectorstore."""
    # Loading of the existing vectorstore from disk if it exists
    if os.path.exists(Config.Persist_Dir):
        return Chroma(persist_directory=Config.Persist_Dir, embedding_function=embeddings)
    
    # Else create the vectorstore from scratch
    loader = WebBaseLoader(
        web_paths=(
            "https://lilianweng.github.io/posts/2023-06-23-agent/", 
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-reasoning/",
            "https://lilianweng.github.io/posts/2020-10-29-odqa/",
            "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/"
            "https://lilianweng.github.io/posts/2023-06-23-agent/", 
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-reasoning/",
            "https://lilianweng.github.io/posts/2020-10-29-odqa/",
            "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/"
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header") # specify which parts to parse
            )
        )
    )
    docs = loader.load()


    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,           # maximum size of each chunk
        chunk_overlap=Config.CHUNK_OVERLAP, # number of characters repeated between chunks
        separators=["\n\n", "\n", ".", " "] # hierarchy of separators to use when splitting
    )
    split_docs = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=Config.Persist_Dir
    )
    return vectorstore

def get_retriever(k=None):
    """Get a retriever from the vectorstore."""
    store = get_vectorstore()
    k = k or Config.TOP_K
    return store.as_retriever(search_kwargs={"k": k})
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class RAGPipeline(ABC):
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever()

    @abstractmethod
    def run(self, query: str):
        """Chaque stratégie DOIT implémenter cette méthode."""
        pass
    
    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format retrieved documents for inclusion in the prompt."""
        formatted = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        return formatted
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .base import RAGPipeline

class BasicRAG(RAGPipeline):
    """Implementation of the Basic RAG strategy.
    This strategy directly retrieves documents based on the user's query
    and generates an answer using those documents."""
    
    def run(self, query: str):
        """Run the Basic RAG pipeline."""
        
        system_template = """You are a knowledgeable AI Research Assistant. 
        Your goal is to answer the user's question based ONLY on the provided context.

        Rules:
        1. If the answer is not contained within the context, simply state that you do not know. 
        2. Do not use outside knowledge.
        3. Keep the answer concise and well-structured.
        4. Use professional technical English."""

        human_template = """Context:
        {context}

        Question: 
        {question}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        
        chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)
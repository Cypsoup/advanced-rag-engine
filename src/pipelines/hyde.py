from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .base import RAGPipeline

class HyDERAG(RAGPipeline):
    """Implementation of the HyDE (Hypothetical Document Embeddings) RAG strategy.
    This strategy involves generating a hypothetical answer to the user's query,
    retrieving documents based on that hypothetical answer, and then answering the
    original query using the retrieved documents."""
    
    def run(self, query: str):
        """Run the HyDE RAG pipeline. We separate the process into three distinct steps for clarity and debugging."""
        
        ### Build the hyde chain and generate an hypothetical answer ###
        hyde_system = (
            "You are an expert AI assistant. "
            "Your task is to generate a detailed, plausible, hypothetical answer "
            "to the user's question. Do not answer strictly, but generate content "
            "that typically appears in a document answering this question."
        )
        
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", hyde_system),
            ("human", "{question}"),
        ])
        
        hyde_chain = (
            {"question": RunnablePassthrough()}
            | hyde_prompt
            | self.llm
            | StrOutputParser()
        )
        
        hypothetical_answer = hyde_chain.invoke(query)
        
        print(f"Generated Hypothetical Answer : {hypothetical_answer}")
        
        
        ### Retrieve documents based on the hypothetical answer ###
        docs = self.retriever.invoke(hypothetical_answer)
        formatted_docs = self.format_docs(docs)
        
        
        ### Build the final chain to answer using retrieved docs ###
        final_system = (
            "You are a helpful assistant. Use the following retrieved context "
            "to answer the user's question. If the answer is not in the context, "
            "say so. Keep it concise."
        )
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", final_system),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ])
        
        final_chain = final_prompt | self.llm | StrOutputParser()

        return final_chain.invoke({
            "context": formatted_docs,
            "question": query
        })
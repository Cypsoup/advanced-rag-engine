import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.core.database import embeddings
from src.core.models import RouteQuery
from .base import RAGPipeline

class LogicalRouterRAG(RAGPipeline):
    """
    Route the query to the appropriate retriever using the LLM. 
    """
    def choose_retriever(self, datasource: str):
        """Choose appropriate retriever based on the selected datasource."""
        
        # In a real project, we would have self.python_retriever, self.js_retriever...
        if datasource in ["python_docs", "js_docs", "golang_docs"]:
            return self.retriever 
        else:
            # Fallback (safety)
            return self.retriever

    def run(self, query: str):
        """Run the Logical RAG pipeline."""
        structured_llm = self.llm.with_structured_output(RouteQuery)
        
        system_msg = (
            "You are an expert router. Route the user's question to the most relevant "
            "datasource based on the programming language mentioned."
        )
        
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{question}"),
        ])

        router_chain = route_prompt | structured_llm

        route_result = router_chain.invoke({"question": query})
        datasource = route_result.datasource
        print(f"Selected datasource: {datasource}")
        
        selected_retriever = self.choose_retriever(datasource)

        # Final chain
        final_chain = (
            {
                "context": selected_retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | ChatPromptTemplate.from_template("Answer based on context: {context}\nQuestion: {question}")
            | self.llm
            | StrOutputParser()
        )
        
        return final_chain.invoke(query)


class SemanticRouterRAG(RAGPipeline):
    """
    Route the query to the appropriate prompt using cosine similarity.
    """
    def __init__(self, llm, vectorstore):
        super().__init__(llm, vectorstore)
        self.prompts_templates = [
            """You are a very smart physics professor. You are great at answering questions about physics in a concise manner.""",
            
            """You are a very good mathematician. You break down hard problems into component parts."""
        ]
        # Compute prompts embeddings
        self.prompts_embeddings = embeddings.embed_documents(self.prompts_templates)

    def cosine_similarity(self, a: list, b: list) -> np.ndarray:
        """Compute cosine similarity between two arrays of vectors."""
        a = np.array(a)
        b = np.array(b)
        
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

    def semantic_route(self, query: str) -> str:
        # Embed the query
        query_embedding = embeddings.embed_query(query)
        
        # Compute similarities
        similarities = self.cosine_similarity([query_embedding], self.prompts_embeddings)[0]
        
        # Get best prompt
        best_idx = np.argmax(similarities)
        return self.prompts_templates[best_idx]

    def run(self, query: str):
        # Select prompt
        selected_template_system = self.semantic_route(query)
        print(f"Selected prompt: {selected_template_system}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", selected_template_system),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ])
        
        docs = self.retriever.invoke(query)
        formatted_docs = self.format_docs(docs)
        
        chain = prompt | self.llm | StrOutputParser()
        
        
        return chain.invoke({"context": formatted_docs, "question": query})
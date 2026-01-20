from typing import List
from pydantic import BaseModel, Field
from langchain_core.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.config import Config
from .base import RAGPipeline


class SearchQueries(BaseModel):
    """Structure to hold multiple search queries."""
    queries: List[str] = Field(
        description="A list of 3 distinct search queries to retrieve relevant documents.",
        min_length=3,
        max_length=3
    )
    

class FusionRAG(RAGPipeline):
    """Implementation of the Reciprocal Rank Fusion (RRF) RAG strategy.
    This strategy generates multiple search queries from the user's question,
    retrieves documents for each query, and then fuses the results using the RRF algorithm
    before generating a final answer."""
    
    @staticmethod
    def reciprocal_rank_fusion(results: List[List[Document]], k=60) -> List[Document]:
        """
        Apply Reciprocal Rank Fusion (RRF) to a list of document lists. Each inner list corresponds to
        the documents retrieved for a specific query.
        RRF algorithm: For each document, its score is the sum of 1 / (k + rank) across all lists,
        where rank is the position of the document in that list (0-based), and k is a constant to dampen the effect of lower-ranked documents.
        """
        fused_scores = {}
        
        # Compute RRF scores
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc) # serialize (turn Document into string) to use as a dict key
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                
                fused_scores[doc_str] += 1 / (rank + k) # RRF score contribution
        
        # Sort documents by their fused scores
        reranked_results = [
            (loads(doc), score) 
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Return only the documents (without the score) and limit to TOP_K
        return [doc for doc, score in reranked_results][:Config.TOP_K]



    def run(self, query: str):
        """Run the Reciprocal Rank Fusion RAG pipeline."""
        
        structured_llm = self.llm.with_structured_output(SearchQueries) # Force llm to use structured output (in this case, SearchQueries objects)
        
        ##### ---------- Multi-Query Generation ---------- #####
        query_gen_system = (
            "You are a helpful AI assistant. Your task is to generate 3 different "
            "search queries that aim to retrieve relevant documents from a vector database. "
            "These queries should look at the user's question from different angles "
            "(e.g., specific keywords, broader concepts, synonymous terms)."
        )
        
        query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", query_gen_system),
            ("human", "User question: {question}\nOutput (3 queries):"),
        ])
        
        generate_chain = query_gen_prompt | structured_llm
        
        result_object = generate_chain.invoke({"question": query})
        generated_queries = result_object.queries
        
        # Ensure the original query is included
        if query not in generated_queries:
            generated_queries.append(query)
            
        print(f"Generated queries ({len(generated_queries)}) :")
        for q in generated_queries:
            print(f"   - {q}")

        
        
        ##### ---------- Document Retrieval ---------- #####        
        results = []
        for q in generated_queries:
            docs = self.retriever.invoke(q)
            results.append(docs)
            
        # RRF Fusion
        final_docs = self.reciprocal_rank_fusion(results)


        
        ##### ---------- Final Answer Generation ---------- #####
        final_system = (
            "You are a knowledgeable AI assistant. Answer the user's question "
            "using the context provided. The context is a compilation of documents "
            "retrieved via multiple search strategies."
        )
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", final_system),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ])
        
        final_chain = final_prompt | self.llm | StrOutputParser()
        
        # Format the fused documents
        formatted_context = self.format_docs(final_docs)
        
        return final_chain.invoke({
            "context": formatted_context,
            "question": query
        })
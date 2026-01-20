from langchain_core.prompts import ChatPromptTemplate
from src.core.models import TutorialSearch
from .base import RAGPipeline

class QueryConstructionRAG(RAGPipeline):
    """
    Transform a query into a structured query (filters + content search).
    Example: "Videos of 2024" -> {"year": 2024}
    """
    
    def run(self, query: str):

        # Force llm to use structured output (in this case, TutorialSearch objects)
        structured_llm = self.llm.with_structured_output(TutorialSearch)

        # Prompt
        system_instruction = (
            "You are an expert at converting user questions into database queries. "
            "You have access to a database of tutorial videos about a software library. "
            "Always expand acronyms. extract specific filters like view count, duration, and date."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "{question}"),
        ])

        # Query structuring chain
        query_analyzer = prompt | structured_llm

        search_params = query_analyzer.invoke({"question": query})
        
        # Missing step
        # In a real pipeline, you would use the search_params object to construct a query to the database.

        # Print structured query
        search_params.pretty_print()
        
        return search_params
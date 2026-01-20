from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base import RAGPipeline
from typing import List
from pydantic import BaseModel, Field


class StepBackQuery(BaseModel):
    """Structure for the abstracted step-back question."""
    step_back_question: str = Field(
        description="The broader, more abstract question that helps retrieve background knowledge.",
    )
    
class StepBackRAG(RAGPipeline):
    """Implementation of the Step-Back RAG strategy.
    This strategy involves generating a broader "step-back" question to retrieve general principles,
    followed by retrieving specific facts related to the original question. The final answer is generated
    using both contexts."""
    
    def run(self, query: str):
      
        structured_llm = self.llm.with_structured_output(StepBackQuery)
        
        ### Prompt Construction ###
        
        # Step-Back Question Generation Prompt
        sb_system_msg = """You are an expert at deep reasoning. Your task is to take a specific, complex question and generate a much broader "step-back" question that retrieves the fundamental principles or background context necessary to answer the original question.

        Here are some examples:
        - Question: "Why is the neural network losing accuracy when I add 5 more layers to this specific PyTorch ResNet-18 implementation?"
          Step-back: "What are the common challenges and vanishing gradient issues associated with increasing depth in convolutional neural networks?"

        - Question: "Did the Esternay plant's production increase during the strike of June 1968?"
          Step-back: "What was the general impact of the May-June 1968 strikes on industrial production in France?"

        - Question: "How do I fix the 'IndexError: list index out of range' in this specific Python function?"
          Step-back: "How does indexing and memory management work for list structures in Python?"
        """

        sb_prompt = ChatPromptTemplate.from_messages([
            ("system", sb_system_msg),
            ("human", "{question}"),
        ])

        # Final Response Generation Prompt
        response_system_msg = """You are a knowledgeable assistant. 
        Answer the user's question using the provided context. 
        
        Instructions:
        - Use the GENERAL PRINCIPLES to set the stage and interpret the specific facts.
        - If the SPECIFIC FACTS are missing, rely on the general principles to provide a likely answer.
        - Keep the tone professional and structured."""

        response_human_msg = """
        1. GENERAL PRINCIPLES (Step-Back Context):
        {step_back_context}

        2. SPECIFIC FACTS (Original Context):
        {specific_context}

        User Question: {question}
        """

        response_prompt = ChatPromptTemplate.from_messages([
            ("system", response_system_msg),
            ("human", response_human_msg),
        ])

        
        ### Pipeline Execution ###

        # Generate the Step-Back Question
        sb_chain = sb_prompt | structured_llm
        step_back_query_object = sb_chain.invoke({"question": query})
        step_back_question = step_back_query_object.step_back_question
        print(f"Generated Step-Back Question: {step_back_question}")
        
        # Double Retrieval
        docs_step_back = self.retriever.invoke(step_back_question)
        step_back_context = self.format_docs(docs_step_back)

        docs_specific = self.retriever.invoke(query)
        specific_context = self.format_docs(docs_specific)

        # Final Answer Generation
        final_chain = response_prompt | self.llm | StrOutputParser()

        return final_chain.invoke({
            "step_back_context": step_back_context,
            "specific_context": specific_context,
            "question": query
        })
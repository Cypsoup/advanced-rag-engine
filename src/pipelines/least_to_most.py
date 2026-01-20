from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base import RAGPipeline
from typing import List
from pydantic import BaseModel, Field


class Decomposition(BaseModel):
    """Structure for decomposing a complex question."""
    sub_questions: List[str] = Field(
        description="A list of 3 distinct sub-questions to solve the problem step-by-step.",
        min_length=3,
        max_length=3
    )


class LeastToMostRAG(RAGPipeline):
    """Implementation of the Least-to-Most RAG strategy.
    This strategy involves decomposing a complex question into simpler sub-questions,
    answering them sequentially while accumulating context, and finally synthesizing
    a final answer based on all intermediate answers."""
    
    def run(self, query: str):
        
        ##### ---------- Decomposition Step ---------- #####
        structured_llm = self.llm.with_structured_output(Decomposition) # Force llm to use structured output
        decomp_system = (
            "You are a helpful assistant that breaks down complex questions. "
            "Your goal is to decompose the user's query into a set of distinct "
            "sub-problems that can be answered in isolation. "
            "Do not answer the questions. Do not explain why."
        )
        
        decomp_prompt = ChatPromptTemplate.from_messages([
            ("system", decomp_system),
            ("human", "Question to decompose: {question}"),
        ])

        decomp_chain = decomp_prompt | structured_llm 
        
        decomposition_result = decomp_chain.invoke({"question": query})
        sub_questions = decomposition_result.sub_questions # get the list of sub-questions
        
        print(f"Sub questions generated ({len(sub_questions)}) :")
        for q in sub_questions:
            print(f"   - {q}")
        


        ##### ---------- Sequential Answering Step ---------- #####
        intermediate_system = (
            "You are a knowledgeable assistant. Answer the current sub-question "
            "using the provided context and the history of previous answers."
        )
        
        intermediate_human = """
        PREVIOUS Q&A (History):
        {q_a_pairs}

        CONTEXT (Retrieved for current question):
        {context}

        CURRENT SUB-QUESTION: 
        {sub_question}
        """

        intermediate_prompt = ChatPromptTemplate.from_messages([
            ("system", intermediate_system),
            ("human", intermediate_human),
        ])
        
        intermediate_chain = intermediate_prompt | self.llm | StrOutputParser()

        q_a_pairs = "" # Q&A pairs accumulator

        # Iterate over each sub-question
        for i, sub_q in enumerate(sub_questions):
            
            # Retrieval for the current sub-question
            docs = self.retriever.invoke(sub_q)
            context = self.format_docs(docs)

            answer = intermediate_chain.invoke({
                "q_a_pairs": q_a_pairs if q_a_pairs else "None yet.",
                "context": context,
                "sub_question": sub_q
            })
            
            # Accumulate the Q&A pair
            q_a_pairs += f"Question: {sub_q}\nAnswer: {answer}\n\n"



        ##### ---------- Final Synthesis Step ---------- #####
        final_system = (
            "You are an expert synthesizer. Your job is to answer the original "
            "complex user question based ONLY on the accumulated Q&A pairs provided."
        )
        
        final_human = """
        Original Question: {question}

        Accumulated Detailed Logic:
        {q_a_pairs}

        Final Concise Answer:
        """

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", final_system),
            ("human", final_human),
        ])

        final_chain = final_prompt | self.llm | StrOutputParser()

        return final_chain.invoke({
            "question": query,
            "q_a_pairs": q_a_pairs
        })
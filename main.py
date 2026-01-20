import sys
import os
from src.core.llm import get_llm
from src.core.database import get_vectorstore
from src.pipelines.basic import BasicRAG
from src.pipelines.hyde import HyDERAG
from src.pipelines.fusion import FusionRAG
from src.pipelines.step_back import StepBackRAG
from src.pipelines.least_to_most import LeastToMostRAG
from src.pipelines.routing import LogicalRouterRAG, SemanticRouterRAG
from src.pipelines.construction import QueryConstructionRAG

def print_header(title):
    """For lisibility in the console."""
    print("\n" + "="*60)
    print(f"TEST : {title.upper()}")
    print("="*60 + "\n")

def main():
    
    ##### ---------- INIT ---------- #####    
    llm = get_llm(temperature=0) 
    vectorstore = get_vectorstore() # create or load db
    
    
    ##### ---------- TESTS ---------- #####
    q_simple = "What is an Agent in the context of LLMs?"
    
    q_complex = "How does the memory module influence the planning capability of an autonomous agent?"
    
    q_routing_python = "How to use decorators in Python?"
    q_routing_physics = "What is a Black Hole?"
    
    q_filters = "Find Python tutorials on decorators published after 2022 with at least 10k views."


    # BASIC RAG
    # Useful to check if the database works and compare with advanced methods
    # run_strategy(BasicRAG(llm, vectorstore), q_complex, "Basic RAG")

    # HyDE (Hypothetical Document Embeddings)
    # Useful for small and unclear questions that require hallucinated context
    # run_strategy(HyDERAG(llm, vectorstore), q_simple, "HyDE RAG")

    # RRF FUSION (Multi-Query)
    # Useful for large and unclear questions that require multiple queries
    # run_strategy(FusionRAG(llm, vectorstore), q_complex, "RRF Fusion")

    # STEP-BACK PROMPTING
    # Useful for very specific questions. At first, search for general principles.
    # run_strategy(StepBackRAG(llm, vectorstore), q_complex, "Step-Back RAG")

    # LEAST-TO-MOST (Decomposition)
    # Useful for complex questions that require a step-by-step reasonning.
    # run_strategy(LeastToMostRAG(llm, vectorstore), q_complex, "Least-To-Most RAG")

    # ROUTING (Logical & Semantic)
    # Test routing to the right source or prompt
    # run_strategy(LogicalRouterRAG(llm, vectorstore), q_routing_python, "Logical Router")
    # run_strategy(SemanticRouterRAG(llm, vectorstore), q_routing_physics, "Semantic Router")

    # QUERY CONSTRUCTION (Filters)
    # Transform a natural language query into a structured query
    # run_strategy(QueryConstructionRAG(llm, vectorstore), q_filters, "Query Construction")


def run_strategy(pipeline, question, name):
    """To execute et print a pipeline."""
    print_header(name)
    print(f"Question : {question}")
    print("-" * 30)
    
    # Pipeline execution
    try:
        response = pipeline.run(question)
        
        print("\nFINAL ANSWER :")
        print("-" * 30)
        print(response)
        print("-" * 30)
        
    except Exception as e:
        print(f"ERROR during execution of {name} : {e}")

if __name__ == "__main__":
    main()
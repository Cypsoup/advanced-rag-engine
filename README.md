# Advanced RAG Architectures

This repository features advanced implementations of **Retrieval-Augmented Generation (RAG)** systems. The project focuses on moving beyond naive RAG architectures by integrating query translation, iterative reasoning, and sophisticated indexing strategies to handle complex technical inquiries.

## Technical Stack

The system is built on an optimized stack designed for precision and semantic diversity:

* **LLM:** Llama 3.3 70B (via Groq) for high-order logic and reasoning.
* **Embeddings:** Google Gemini `text-embedding-004`.
* **Vector Store:** ChromaDB with local persistence.
* **Retriever:** **Maximal Marginal Relevance (MMR)** search to ensure context diversity and minimize redundancy.

## Current Status: Test Case
To validate the initial implementation of the different strategies, this engine is currently indexed on **Lilian Weng’s technical blog posts**. This dataset provides a high-density environment of complex AI concepts (Memory, Planning, Tool Use) ideal for testing the system's ability to cross-reference and synthesize technical documentation.

## Project Structure

The project follows a modular **Object-Oriented Architecture**, separating infrastructure (Core) from retrieval strategies (Pipelines). This design allows for easy extension and independent testing of each RAG method.

```text
advanced_rag_engine/
│
├── .env                    # Environment variables (API Keys)
├── main.py                 # Entry point & Strategy Orchestrator
├── requirements.txt        # Python dependencies
│
└── src/
    ├── core/               # Infrastructure & Shared Components
    │   ├── llm.py          # LLM Initialization (Groq)
    │   ├── database.py     # ChromaDB Vector Store & Embeddings
    │   └── models.py       # Pydantic Models for Structured Output
    │
    └── pipelines/          # RAG Strategies (Business Logic)
        ├── base.py         # Abstract Base Class (RAGPipeline Interface)
        ├── basic.py        # Naive RAG implementation
        ├── hyde.py         # Hypothetical Document Embeddings
        ├── fusion.py       # RRF (Reciprocal Rank Fusion)
        ├── stepback.py     # Step-Back Prompting (Abstraction)
        ├── least_to_most.py# Sequential Sub-query Decomposition
        ├── routing.py      # Logical & Semantic Routing
        └── construction.py # Self-Querying (Natural Language to Metadata Filters)

```

## Implemented Strategies

The engine implements a variety of advanced RAG patterns, each encapsulated in its own pipeline:

### 1. Basic RAG

A standard retrieval baseline using MMR search to fetch relevant context and generate an answer.

### 2. Hypothetical Document Embeddings (HyDE)

Generates a hypothetical answer to the user's query first, then uses that hallucination to retrieve real documents. This improves retrieval when the query and the documents are in different semantic spaces.

### 3. Reciprocal Rank Fusion (RRF)

A multi-query fusion architecture. By generating variations of a single prompt, the system queries the vector space from multiple semantic angles and re-ranks the results using the RRF algorithm to ensure the most statistically relevant documents are prioritized.

### 4. Step-Back Prompting

Extracts high-level concepts and principles by generating a broader "step-back" question. The system retrieves context for both the abstract principle and the specific details to ground the final answer.

### 5. Least-to-Most Decomposition

Breaks down complex queries into a series of dependent sub-problems. Each sub-question is resolved sequentially, utilizing both the document context and the answers from previous steps. Ideal for multi-step reasoning.

### 6. Semantic & Logical Routing

* **Logical Routing:** Uses function calling to route queries to specific data sources (e.g., Python docs vs. JS docs) based on user intent.
* **Semantic Routing:** Uses embedding similarity to route queries to different prompt personas (e.g., "Physics Professor" vs. "Mathematician").

### 7. Query Construction (Self-Querying)

Translates natural language questions into structured database queries with metadata filters (e.g., "videos from 2024" -> `filter={year: 2024}`).

## Installation

### Prerequisites

* Python 3.9+
* Groq API Key
* Google Gemini API Key (for Embeddings)

### Configuration

1. Clone the repository:

```bash
git clone [https://github.com/cypsoup/advanced-rag-engine.git](https://github.com/cypsoup/advanced-rag-engine.git)
cd advanced-rag-engine

```

2. Install dependencies:

```bash
pip install -r requirements.txt

```

3. Create a `.env` file in the root directory:

```text
GROQ_API_KEY=your_api_key
GOOGLE_API_KEY=your_api_key

```

## Usage

The `main.py` file serves as the laboratory for testing different pipelines.

1. Open `main.py`.
2. Locate the **"ZONE DE TEST"** section.
3. Uncomment the strategy you wish to run:

```python
# Example in main.py
# run_strategy(BasicRAG(llm, vectorstore), q_complex, "Basic RAG")
run_strategy(FusionRAG(llm, vectorstore), q_complex, "RRF Fusion")

```

4. Run the script:

```bash
python main.py

```

## Roadmap

The project evolves towards **Agentic and Adaptive RAG** systems. Future updates will include:

* **Advanced Indexing:** Integration of RAPTOR (tree-organized retrieval) and ColBERT for late-interaction token matching.
* **Corrective & Adaptive Architectures:** Development of **CRAG** (Corrective RAG) for retrieval quality control and **Adaptive RAG** for dynamic self-correction loops.

```

```
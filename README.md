# Advanced RAG Architectures

This repository features advanced implementations of **Retrieval-Augmented Generation (RAG)** systems. The project focuses on moving beyond naive RAG architectures by integrating query translation, iterative reasoning, and sophisticated indexing strategies to handle complex technical inquiries.

## Technical Stack

The system is built on an optimized stack designed for precision and semantic diversity:

* **LLM:** Llama 3.3 70B (via Groq) for high-order logic and reasoning.
* **Embeddings:** Google Gemini `text-embedding-004`.
* **Vector Store:** ChromaDB with local persistence.
* **Retriever:** **Maximal Marginal Relevance (MMR)** search to ensure context diversity and minimize redundancy.

## Current Status: Test Case
To validate the initial implementation of the **Sequential Decomposition** and **MMR Retrieval** strategies, this engine is currently indexed on **Lilian Weng’s technical blog posts**. This dataset provides a high-density environment of complex AI concepts (Memory, Planning, Tool Use) ideal for testing the system's ability to cross-reference and synthesize technical documentation.

## Implemented Strategies

### 1. Least-to-Most Decomposition

This method breaks down complex queries into a series of dependent sub-problems. Each sub-question is resolved sequentially, utilizing both the document context and the answers from previous steps. This approach is specifically designed for tasks requiring a chain of causality or multi-step mathematical reasoning.

### 2. Reciprocal Rank Fusion (RRF)

The engine includes a multi-query fusion architecture. By generating variations of a single prompt, the system queries the vector space from multiple semantic angles and re-ranks the results using the RRF algorithm to ensure the most statistically relevant documents are prioritized.

### 3. Contextual Synthesis

A final synthesis stage consolidates intermediate reasoning chains into a structured, technical response, effectively filtering noise and contradictions found in the raw source material.

## Installation

### Prerequisites

* Python 3.9+
* Groq API Key
* Google Gemini API Key (for Embeddings)

### Configuration

1. Clone the repository:
```bash
git clone https://github.com/cypsoup/advanced-rag-engine.git

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



## Roadmap

The project follows a modular evolution towards **Agentic and Adaptive RAG** systems. Future updates will include:

* **Logic & Routing:** Implementation of intent-based routing and structured query construction (metadata filtering).
* **Advanced Indexing:** Integration of multi-representation indexing, RAPTOR (tree-organized retrieval), and ColBERT for late-interaction token matching.
* **Corrective & Adaptive Architectures:** Development of **CRAG** (Corrective RAG) for retrieval quality control and **Adaptive RAG** for dynamic self-correction loops.

import bs4
import os
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from operator import itemgetter


# Load environment variables (API keys) from .env file
load_dotenv()


##### HYPERPARAMETERS #####
CHUNK_SIZE = 800          # Size of each text chunk
CHUNK_OVERLAP = 150       # Overlap between chunks
TOP_K = 7                 # Number of documents to retrieve
FETCH_K = 20              # Number of candidate documents to consider during retrieval

##### INDEXING #####
# Load documents from web pages about AI topics
loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/", 
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-reasoning/",
        "https://lilianweng.github.io/posts/2020-10-29-odqa/",
        "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/"
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header") # specify which parts to parse
        )
    )
)
docs = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,           # maximum size of each chunk
    chunk_overlap=CHUNK_OVERLAP, # number of characters repeated between chunks
    separators=["\n\n", "\n", ".", " "] # hierarchy of separators to use when splitting
)
split_docs = text_splitter.split_documents(docs)

# Embed the document chunks
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
persist_directory = "./chroma_db_lilianweng" # directory to store the vector database
if not os.path.exists(persist_directory): # if the vector database does not exist, create it
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
else: # if it exists, load it
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )


# Create a retriever with specific search parameters
retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K, "fetch_k": FETCH_K}, # retrieve top TOP_K results from top FETCH_K candidates
    search_type="mmr", # use Maximal Marginal Relevance for retrieval (diversity in results)
)



##### RETRIEVAL AND GENERATION #####
# Prompt
template = """You are a knowledgeable AI Research Assistant. 
Use the following pieces of retrieved context to answer the user's question.

Rules:
1. If the answer is not contained within the context, simply state that you do not know. 
2. Do not use outside knowledge.
3. Keep the answer concise and well-structured.
4. Use professional technical English.

Context:
{context}

Question: 
{question}

Helpful Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile",
)

# Post processing
def format_docs(docs: list[str]) -> str:
    """Format retrieved documents for inclusion in the prompt."""
    formatted = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    return formatted


# --- CONSTRUCTION DE LA RAG CHAIN (FLUX DE DONNÉES) ---
# 1. Préparation de l'entrée : 
#    - 'context' : La question passe par le retriever, puis les docs trouvés sont formatés en texte.
#    - 'question' : La question originale est transmise telle quelle (passe-plat).
# 2. Prompt : On injecte le contexte et la question dans la variable prompt qui genere un prompt textuel final.
# 3. LLM : On envoie le prompt final au llm pour générer la réponse.
# 4. Parser : On extrait uniquement le texte de la réponse (nettoyage des métadonnées).
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Question
response = rag_chain.invoke("Explique comment la 'Memory' (mémoire) d'un agent influence ses capacités de 'Planning' (planification), et donne un exemple de technique citée par l'auteur qui combine ces deux aspects")
print("----- RAG RESPONSE (basic) -----")
print(response)
print("--------------------------------")


##### ALTERNATIVE RRF FUSION STRATEGY #####
# from langchain.load import dumps, loads

# # --- FUSION FUNCTION (RRF) ---
# def reciprocal_rank_fusion(results: list[list], k=60) -> list:
#     """Using Reciprocal Rank Fusion (RRF) algorithm, rerank documents from multiple retrieval results."""
#     fused_scores = {}
#     for docs in results:
#         for rank, doc in enumerate(docs):
#             doc_str = dumps(doc) # Serialize document to string to use as a key
#             if doc_str not in fused_scores:
#                 fused_scores[doc_str] = 0
#             fused_scores[doc_str] += 1 / (rank + k)
    
#     # Rank the documents based on fused scores (we use loads() to deserialize the document strings)
#     reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    
#     # Return only the TOP_K top documents, discarding scores
#     return [doc for doc, score in reranked_results][:TOP_K]

# generate_queries_template = """You are an AI assistant. Generate 3 different versions 
# of the following user question to retrieve relevant documents from a vector database.
# Original question: {question}"""
# generate_queries_prompt = ChatPromptTemplate.from_template(generate_queries_template)

# generate_queries_chain = (
#     generate_queries_prompt 
#     | llm 
#     | StrOutputParser() 
#     | (lambda x: x.split("\n")) # Split the output into separate questions
# )

# fusion_chain = (
#     generate_queries_chain 
#     | retriever.map() # Get documents for each generated question
#     | reciprocal_rank_fusion # Fuse and rank the retrieved documents
#     | format_docs # Format the final documents into a single context string
# )

# rag_chain = (
#     {"context": fusion_chain, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )



##### LEAST-TO-MOST RAG (sequential sub-query decomposition) #####

### GENERATE SUB-QUESTIONS ###
# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

# Chain
generate_queries_decomposition = ( 
    prompt_decomposition 
    | llm 
    | StrOutputParser() 
    | (lambda x: [q.strip("1234567890. ") for q in x.split("\n") if q.strip()]) # Split and clean output into list of questions
)

# Run
question = "Explique comment la 'Memory' (mémoire) d'un agent influence ses capacités de 'Planning' (planification), et donne un exemple de technique citée par l'auteur qui combine ces deux aspects"
questions = generate_queries_decomposition.invoke({"question":question})


### GENERATE INTERMEDIATE ANSWERS ###
# Prompt
template = """Here is the question you need to answer:
\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:
\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 
\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

intermediate_prompt = ChatPromptTemplate.from_template(template)

# RAG CHAIN FOR EACH SUB-QUESTION
rag_chain_intermediate = (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")
    } 
    | intermediate_prompt 
    | llm 
    | StrOutputParser()
)

q_a_pairs = "" # Initialize empty string to accumulate Q&A pairs

# Iterate over each sub-question, get answer, and accumulate Q&A pairs
for q in questions:
    if q.strip() == "": # skip empty questions
        continue
    
    answer = rag_chain_intermediate.invoke({
            "question": q, 
            "q_a_pairs": q_a_pairs
        })    
    
    q_a_pairs += f"\nQ: {q}\nA: {answer}\n---\n"

    

### FINAL ANSWER GENERATION ###
template_final = """Based on the following accumulated question and answer pairs, provide a concise and well-structured final answer to the original question:
{question}

Accumulated Q&A pairs:
{q_a_pairs}

Final answer :"""

final_prompt = ChatPromptTemplate.from_template(template_final)

# Final chain
final_chain = final_prompt | llm | StrOutputParser()

# On lance la synthèse
final_res = final_chain.invoke({
    "question": question, # original question
    "q_a_pairs": q_a_pairs
})

print("----- LEAST-TO-MOST RAG RESPONSE -----")
print(final_res)
print("--------------------------------------")
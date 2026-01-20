from langchain_groq import ChatGroq
from src.config import Config

def get_llm(temperature=0):
    return ChatGroq(
        temperature=temperature,
        model_name=Config.Model_Name
    )
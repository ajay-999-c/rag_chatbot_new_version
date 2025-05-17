from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv() 
import os
generator_llm = ChatGroq(
    model_name="llama-3-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = "what is the fee structure for the data science course?"
response = generator_llm(prompt)
print(response.text)
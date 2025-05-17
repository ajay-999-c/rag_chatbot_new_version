from dotenv import load_dotenv
load_dotenv() 
import os
import httpx # Add httpx import

from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
from logger import log_event  # For logging events into file

print("\n🔵 [System] Starting Generator Initialization...\n")

# Load optional environment config - priority order: Groq > HuggingFace > Ollama
USE_GROQ = os.getenv("USE_GROQ", "true").lower() == "true"  # Default True (Groq as default)
USE_HUGGINGFACE = os.getenv("USE_HUGGINGFACE", "false").lower() == "true"  # Default False
USE_LOCAL_HF = os.getenv("USE_LOCAL_HF", "false").lower() == "true"  # Use local HF model instead of API

# Initialize generator_llm to None for error handling
generator_llm = None
print(f"Initializing generator_llm to None for error handling...\\n") # Fixed typo
print(f"🔵 [Config] Attempting to use Groq: {USE_GROQ}") # Made message clearer

try:
    # Option 1: Try to use Groq if enabled
    if USE_GROQ:
        print("🔵 [Flow] Entering Groq setup block.")
        try:
            # Import Groq integration
            import importlib.util
            if importlib.util.find_spec("langchain_groq") and importlib.util.find_spec("groq"):
                from langchain_groq import ChatGroq
                # from groq import Groq as GroqSDKClient # No longer needed for this simplified approach
                
                print("🔵 [Loading] Attempting to load Groq LLM model (simplified initialization)...")

                # Simplified ChatGroq initialization
                # Remove custom httpx.Client and GroqSDKClient instantiation
                # custom_httpx_client = httpx.Client(proxies=None)
                # groq_sdk_client = GroqSDKClient(
                #     api_key=os.getenv("GROQ_API_KEY"),
                #     http_client=custom_httpx_client
                # )

                generator_llm = ChatGroq(
                    # client=groq_sdk_client, # Removed: Pass parameters directly
                    model_name="llama-3.1-8b-instant",  # Or "llama3-70b-8192"
                    temperature=0.2,
                    groq_api_key=os.getenv("GROQ_API_KEY") # ChatGroq will use this to create its internal Groq client
                                                          # It will also pick up GROQ_API_KEY from env if this is not passed
                )
                log_event("✅ Initialized Groq ChatGroq LLM: llama-3.1-8b-instant")
                print("✅ [Success] Groq model loaded successfully.\\n")
                print("💡 RAG System is using: Groq (llama-3.1-8b-instant)")
            else:
                print("⚠️ langchain_groq or groq package not found. Install with: pip install langchain-groq groq")
                USE_GROQ = False # Update flag
        except Exception as e:
            print(f"⚠️ Failed to load Groq model: {str(e)}. Falling back to alternative model.")
            USE_GROQ = False # Update flag
            # USE_HUGGINGFACE = True # This line is already present in your file

    # Option 2: Try to use HuggingFace if enabled or if Groq failed
    print(f"🔵 [Config] After Groq attempt, USE_GROQ: {USE_GROQ}, USE_HUGGINGFACE: {USE_HUGGINGFACE}")
    if not USE_GROQ and USE_HUGGINGFACE:
        print("🔵 [Flow] Entering HuggingFace setup block.")
        if USE_LOCAL_HF:
            print("🔵 [Flow] Attempting Local HuggingFace.")
            try:
                from langchain_community.llms import HuggingFacePipeline
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                print("🔵 [Loading] Attempting to load local HuggingFace model...")
                
                # Use specified model or default to a small model
                model_name = os.getenv("HF_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                
                # Determine device (CPU or GPU)
                device = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
                print(f"  Using device: {device}")
                
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map=device, 
                    torch_dtype="auto"
                )
                
                # Create a text generation pipeline
                text_pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.2,
                    top_p=0.95,
                    repetition_penalty=1.15
                )
                
                # Create LangChain wrapper around the pipeline
                generator_llm = HuggingFacePipeline(pipeline=text_pipe)
                
                log_event(f"✅ Initialized local HuggingFace model: {model_name}")
                print(f"✅ [Success] Local HuggingFace model {model_name} loaded successfully.\n")
                print(f"💡 RAG System is using: Local HuggingFace ({model_name})")
            except Exception as e:
                print(f"⚠️ Failed to load local HuggingFace model: {str(e)}. Trying Inference API.")
                USE_LOCAL_HF = False
        
        # Try HuggingFace Inference API if local model failed or wasn't requested
        if not USE_LOCAL_HF:
            print("🔵 [Flow] Attempting HuggingFace Inference API.")
            try:
                import importlib.util # Already present
                if importlib.util.find_spec("langchain_huggingface"):
                    from langchain_huggingface import HuggingFaceEndpoint
                    
                    print("🔵 [Loading] Attempting to load HuggingFace Inference API model...")
                    
                    # Get HF API token from environment
                    hf_api_key = os.getenv("HF_API_KEY")
                    if not hf_api_key:
                        raise ValueError("HF_API_KEY environment variable not set")
                        
                    # Use the HuggingFace Inference API with default model
                    model_name = os.getenv("HF_MODEL_NAME", "google/gemma-2b-it")
                    generator_llm = HuggingFaceEndpoint(
                        endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}",
                        huggingfacehub_api_token=hf_api_key,
                        task="text-generation",
                        max_length=512,
                        temperature=0.2
                    )
                    log_event(f"✅ Initialized HuggingFace Inference API LLM: {model_name}")
                    print(f"✅ [Success] HuggingFace Inference API model {model_name} loaded successfully.\n")
                    print(f"💡 RAG System is using: HuggingFace Inference API ({model_name})")
                else:
                    print("⚠️ langchain_huggingface not found. Install with: pip install langchain-huggingface")
                    USE_HUGGINGFACE = False
            except Exception as e:
                print(f"⚠️ Failed to load HuggingFace Inference API: {str(e)}. Falling back to Ollama.")
                # USE_HUGGINGFACE = False # This line is already present in your file
    
    # Option 3: Fallback to Ollama if both Groq and HuggingFace failed
    print(f"🔵 [State] Before Ollama fallback, generator_llm is: {type(generator_llm)}")
    if generator_llm is None:
        print("🔵 [Flow] Entering Ollama fallback block.")
        print("🔵 [Loading] Attempting to load Ollama LLM model (gemma3:1b)...")
        
        from langchain_community.llms import Ollama

        generator_llm = Ollama(
            model="gemma3:1b",  # ✅ Fallback model name!
            base_url="http://localhost:11434",
            temperature=0.2
        )
        log_event("✅ Initialized Ollama local LLM: gemma3:1b")
        print("✅ [Success] Ollama local model loaded successfully.\n")
        print("💡 RAG System is using: Ollama (gemma3:1b)")

except Exception as e:
    log_event(f"❌ Failed to initialize LLM: {str(e)}")
    print(f"❌ [Error] Failed to initialize LLM: {str(e)}")
    raise e

# Build Prompt Template
print("🔵 [Building] Creating Generation Prompt Template...")

from langchain_core.prompts import PromptTemplate

generation_prompt = PromptTemplate.from_template("""
You are Bignalytics' official AI assistant for prospective students and visitors.

Answer the user's question based **only on the information provided in the context below** (which is extracted from Bignalytics' verified knowledge base). 
Do not make up answers. If the context does not have enough information, respond politely with: "Sorry, not enough information."

Instructions:
- Give clear, concise, and factual answers relevant to the coaching center (courses, admissions, fees, placements, batches, trainers, etc.).
- Use bullet points for lists or multiple items if helpful.
- If a question is not directly covered in the context, do **not** attempt to infer or guess.

Context:
{context}

Question:
{question}

Answer:
""")
print("✅ [Success] Generation Prompt Template ready.\n")

# Build Generator Chain
print(f"🔵 [Building] About to combine Prompt and LLM. Current generator_llm type: {type(generator_llm)}, Object: {generator_llm}")
print('printing generator prompt------------------')
print(generation_prompt)
generator_chain = generation_prompt | generator_llm

# generation_prompt = PromptTemplate.from_template("""
# You are an AI assistant helping users.

# Use the provided context to answer the question accurately.
# If the context is insufficient, politely respond "Sorry, not enough information."

# Context:
# {context}

# Question:
# {question}
# """)
# print("✅ [Success] Generation Prompt Template ready.\\n")

# # Build Generator Chain
# print(f"🔵 [Building] About to combine Prompt and LLM. Current generator_llm type: {type(generator_llm)}, Object: {generator_llm}")
# print('printing generator prompt------------------')
# print(generation_prompt)
# generator_chain = generation_prompt | generator_llm
print("✅ [Success] Generator Chain ready for use.\\n")

print("✅ [System] Generator initialization completed.\\n")

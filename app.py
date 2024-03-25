from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama, openai
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ['HF_TOKEN'] = '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'

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "arthurwangheng/Llama-2-7b-FAQ-chatbot-finetune"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,  
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a Emmorce FAQ assistant.You are going to answer the question customer asks'),
    ('user', '{input}'),
])

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

chain = prompt | model

add_routes(
    app,
    chain,
    path="/chatbot",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=9000)
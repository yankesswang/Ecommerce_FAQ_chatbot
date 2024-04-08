# Standard library imports
import os
import json
from pathlib import Path
from pprint import pprint
import tempfile

# Third-party library imports for environment management and web application
from dotenv import load_dotenv
import uvicorn
import streamlit as st

# FastAPI for web server
from fastapi import FastAPI
from langserve import add_routes
# Langchain imports for conversational retrieval and embeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms import Ollama, openai
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Llama Index imports for document processing and vector storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Streamlit components
from streamlit_chat import message

# Transformers for NLP models and tokenization
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Environment variable setup
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def load_llm():
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
    return model

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server for FAQ ChatBot",
)

def Runnable():
    llm = load_llm()
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a Emmorce FAQ assistant.You are going to answer the question customer asks'),
        ('user', '{input}'),
    ])
    chain = prompt | llm 
    return chain

add_routes(
    app,
    Runnable(),
    path="/chatbot"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=1900)

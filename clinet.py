from langchain.prompts import ChatPromptTemplate
from langserve import RemoteRunnable
import streamlit as st
import json
from pathlib import Path
from pprint import pprint
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
load_dotenv()
import json
from langchain.docstore.document import Document
import requests

st.title("FAQ Chatbot - 🤖")
# streamlit run clinet.py

# Function for conversational chat
def conversational_chat(query):
    response= requests.post("http://localhost:1900/chatbot/invoke",json= {'input':{'question':query}})
    result = str(response.json()['output']['answer'])
    st.session_state['history'].append((query, result))
    return result

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = [f"Hello! Ask me about any Ecommerce Question 🤗"]
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! I am Arthur👋"]

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Ask a question 👉", key='input')
        submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


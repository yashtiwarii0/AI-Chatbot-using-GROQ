import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
import langchain
## Langsmith tracking
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith tracking setup
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"   # <-- fixed typo
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with GROQ"

##prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_respopnse(question,model,temperature,max_tokens):
    llm=ChatGroq(
        #groq_api_key=groq_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

#title of the app

st.title("Enhanced Q&A chatbot with GROQ")

##sidebar for settings

st.sidebar.title("settings")
model=st.sidebar.selectbox("Select an GROQ AI model",["llama-3.1-8b-instant","llama-3.3-70b-versatile","openai/gpt-oss-120b","openai/gpt-oss-20b"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

##main interface for user interface
st.write("go ahead and ask any question")
user_input=st.text_input("You:")


if user_input:
    response=generate_respopnse(user_input,model,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide the query")
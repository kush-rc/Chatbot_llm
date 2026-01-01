import streamlit as st
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Initialize Groq
@st.cache_resource
def initialize_llm():
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )

# Initialize vector store
@st.cache_resource
def initialize_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(
        persist_directory="./vectorstore",
        embedding_function=embeddings
    )
    return vectorstore

# Initialize chatbot
llm = initialize_llm()
vectorstore = initialize_vectorstore()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

# UI
st.title("🤖 RAG Chatbot with Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

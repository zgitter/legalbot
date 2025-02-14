from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import streamlit as st

GOOGLE_API_KEY = st.secrets.GOOGLE_API_KEY
PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
PINECONE_ENVIRONMENT = st.secrets.PINECONE_ENVIRONMENT
PINECONE_INDEX_NAME = st.secrets.PINECONE_INDEX_NAME

# Initialize components
def init_rag_pipeline():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    return vector_store, llm

# Define RAG pipeline
def get_rag_response(vector_store, llm, query):
    prompt_template = """
    Use the following context to answer the question. If you don't know the answer, just say you don't know.
    Keep the answer concise and relevant.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )
    
    response = qa_chain.invoke({"query": query})
    return response["result"]

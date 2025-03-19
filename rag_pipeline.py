from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


import streamlit as st
import re

import pickle
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


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



# Define RAG pipeline + ?
# 

def load_bm25_index():
    """Loads the precomputed BM25 index from file."""
    with open("data/bm25_index.pkl", "rb") as f:
        bm25, corpus_texts = pickle.load(f)
    return bm25, corpus_texts

def get_bm25_results(query: str, bm25: BM25Okapi, corpus_texts: list, top_n: int = 5):
    """Returns a set of top-n document texts from BM25 retrieval."""
    tokenized_query = word_tokenize(query.lower())
    return set(bm25.get_top_n(tokenized_query, corpus_texts, n=top_n))

def get_vector_results(query: str, vector_store, top_k: int = 5):
    """Returns a set of top-k document texts from vector search."""
    return set(doc.page_content for doc in vector_store.similarity_search(query, k=top_k))

def get_hybrid_context(query: str, vector_store, bm25: BM25Okapi, corpus_texts: list, top_k: int = 5):
    """Retrieve contexts using both vector search and BM25, then combine them."""
    vector_contexts = get_vector_results(query, vector_store, top_k=top_k)
    bm25_contexts = get_bm25_results(query, bm25, corpus_texts, top_n=top_k)
    return "\n\n".join(vector_contexts.union(bm25_contexts))

def get_rag_response(vector_store, llm, query):
    """Retrieves relevant context using hybrid search and generates an LLM response."""
    bm25, corpus_texts = load_bm25_index()
    combined_context = get_hybrid_context(query, vector_store, bm25, corpus_texts, top_k=5)
    
    prompt_template = """
    You are a helpful legal assistant. Use the following legal context to answer the question accurately.
    If the answer is not present in the context, say "I don't know."
    
    Legal Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )
    
    response = qa_chain.invoke({"query": query, "context": combined_context})
    return response["result"]



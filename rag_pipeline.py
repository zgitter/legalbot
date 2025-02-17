from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import re

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


def get_rag_response(vector_store, llm, query):
    """Retrieve relevant legal text from Pinecone and generate a response."""
    
    # Step 1: Detect specific legal references in query
    article_match = re.search(r"Article (\d+)", query, re.IGNORECASE)
    chapter_match = re.search(r"Chapter (\d+)", query, re.IGNORECASE)

    # Build metadata filters
    filters = {}
    if article_match:
        filters["article_number"] = article_match.group(1)  # Filter by article number
    if chapter_match:
        filters["chapter"] = f"Chapter {chapter_match.group(1)}"  # Filter by chapter title
    
    # Step 2: Retrieve documents with filtering if applicable
    if filters:
        retrieved_docs = vector_store.similarity_search(query, k=10, filter=filters)
    else:
        retrieved_docs = vector_store.similarity_search(query, k=10)  # No filter, normal search

    # Step 3: Merge retrieved texts
    merged_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 4: Construct prompt
    prompt_template = """
    Use the following legal text to answer accurately. If an article is missing, say so.

    Legal Context: {context}
    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Step 5: Run the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
    )

    response = qa_chain.invoke({"query": query, "context": merged_context})
    return response["result"]


'''
# Oldy?
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
'''
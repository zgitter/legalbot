from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import re

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

<<<<<<< HEAD
# Define RAG pipeline


def get_rag_response(vector_store, llm, query):
    retriever = vector_store.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": 20}  # Increase k to 20 to fetch more results
    )
    
    #retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_docs = retriever.invoke(query) # the above is deprecated

    # Helper function to calculate relevance based on keyword matches
    def relevance_score(doc, query):
        query_words = set(query.lower().split())  # Split query into words
        doc_words = set(doc.page_content.lower().split())  # Split document text
        return len(query_words & doc_words)  # Count common words

    # Sort documents by relevance score (higher = more relevant)
    sorted_docs = sorted(retrieved_docs, key=lambda doc: relevance_score(doc, query), reverse=True)

    # Merge top-ranked documents into a single context
    merged_context = "\n\n".join([doc.page_content for doc in sorted_docs[:10]])  # Use top 10 ranked results

    # Define prompt
    prompt_template = """
    Use the following expanded context to answer the question accurately. If some parts seem missing, infer from related context.

    Expanded Context: {context}
    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
    )

    response = qa_chain.invoke({"query": query, "context": merged_context})
    return response["result"]




'''
def get_rag_response(vector_store, llm, query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "fetch_k": 50}  # Fetch more documents for better ranking
    )
    
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Custom Re-Ranking: Prioritize documents that contain exact keyword matches
    def relevance_score(doc, query):
        query_words = set(query.lower().split())  # Split query into words
        doc_words = set(doc.page_content.lower().split())  # Split document text
        return len(query_words & doc_words)  # Count common words
    
    # Sort documents by how many words they share with the query
    sorted_docs = sorted(retrieved_docs, key=lambda doc: relevance_score(doc, query), reverse=True)
    
    # Merge top-ranked documents into a single context
    merged_context = "\n\n".join([doc.page_content for doc in sorted_docs[:10]])  # Use top 10 ranked results

    # Updated prompt with explicit context
    prompt_template = """
    Use the following expanded context to answer the question accurately. If some parts seem missing, infer from related context.

    Expanded Context: {context}
    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
    )

    response = qa_chain.invoke({"query": query, "context": merged_context})
    return response["result"]

'''



'''def get_rag_response_basic(vector_store, llm, query):
=======
<<<<<<< HEAD
import re

=======

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
>>>>>>> 08836fa7b73c9588450b3996d6962af4be20a02a
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
<<<<<<< HEAD
    
    response = qa_chain.invoke({"query": query})
    return response["result"]'''
=======

    response = qa_chain.invoke({"query": query, "context": merged_context})
    return response["result"]


'''
# Oldy?
# Define RAG pipeline
>>>>>>> 6bb4e5cb6d0c643ef4076f4af22bce989e27e9ec
def get_rag_response(vector_store, llm, query):
    """Retrieve relevant legal text from Pinecone and generate a response."""
    
    # Step 1: Detect specific legal references in query
    article_match = re.search(r"Article (\d+)", query, re.IGNORECASE)
    chapter_match = re.search(r"Chapter (\d+)", query, re.IGNORECASE)

    # Build metadata filters
    filters = {}
    if article_match:
        filters["article_number"] = {"$eq": str(article_match.group(1))}  # Exact match for article number
    if chapter_match:
        filters["chapter"] = {"$eq": f"Chapter {chapter_match.group(1)}"}  # Exact match for chapter title

    # Step 2: Try metadata filtering first, then fallback to similarity search
    try:
        if filters:
            retrieved_docs = vector_store.similarity_search(query, k=50, filter=filters)
        else:
            retrieved_docs = vector_store.similarity_search(query, k=50)  # No filter, normal search
    except Exception as e:
        print(f"Error in retrieval: {e}")
        retrieved_docs = vector_store.similarity_search(query, k=50)  # Fallback to normal search

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
>>>>>>> 08836fa7b73c9588450b3996d6962af4be20a02a

# rag_pipeline.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

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
    return response["result"]'''
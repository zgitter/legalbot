import os
import json
import re
from dotenv import load_dotenv
import pinecone
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import PineconeVectorStore

# Load environment variables
load_dotenv()

# Retrieve Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

def extract_article_number(article_title):
    """Extract article number from titles like 'Article 1. Title' → '1'."""
    match = re.search(r'Article (\d+)', article_title)
    return match.group(1) if match else None

def process_section(section):
    """Extract text content from section (handles dict, list, and nested structures)."""
    if isinstance(section, dict):
        return str(section.get("content", "")) + "\n" + process_section(section.get("section", ""))
    elif isinstance(section, list):
        return "\n".join(process_section(subsec) for subsec in section)
    return str(section)  # Ensure all values are strings

def process_chapter(chapter):
    """Process a chapter and yield Documents for each article."""
    chapter_title = chapter.get("content", "")
    for article in chapter.get("section", []):
        article_title = article.get("content", "")
        article_number = extract_article_number(article_title)
        
        # Extract article text, handling nested sections
        article_text = process_section(article.get("section", ""))

        # Combine title and text
        full_text = f"{article_title}\n\n{article_text}"
        
        # Build metadata dictionary
        metadata = {
            "chapter": chapter_title,
            "article": article_title,
            "article_number": article_number
        }
        
        yield Document(page_content=full_text, metadata=metadata)

# Load JSON law book
with open("/mnt/data/Ethiopia_1994.json", "r", encoding="utf-8") as f:
    law_data = json.load(f)

# Prepare Documents
documents = []
for chapter in law_data:
    documents.extend(list(process_chapter(chapter)))

print(f"Prepared {len(documents)} documents for embedding.")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store in Pinecone
vector_store = PineconeVectorStore.from_documents(
    documents,
    embeddings,
    index_name=PINECONE_INDEX_NAME
)

print("Successfully embedded and stored the law book in Pinecone!")

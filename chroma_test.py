import os
import chromadb
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
gemini_client = genai.Client(api_key=api_key)

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = chroma_client.get_or_create_collection(
    name="industrial_documents"
)

# Test document
text = "Industrial machines require regular maintenance."

# Create Gemini embedding
result = gemini_client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)

embedding = result.embeddings[0].values

# Store text + embedding in ChromaDB
collection.add(
    ids=["test_embedding_1"],
    documents=[text],
    embeddings=[embedding]
)

print("Gemini embedding stored in ChromaDB successfully!")
print("Number of documents:", collection.count())
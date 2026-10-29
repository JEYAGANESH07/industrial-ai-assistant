import os
import chromadb
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
gemini_client = genai.Client(api_key=api_key)

# Connect to existing ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get our existing collection
collection = chroma_client.get_collection(
    name="industrial_documents"
)

# User's question
question = "How often should machinery be maintained?"

# Convert question into an embedding
result = gemini_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = result.embeddings[0].values

# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=1
)

# Display result
print("Question:")
print(question)

print("\nMost relevant document:")
print(results["documents"][0][0])
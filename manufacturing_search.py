import os

import chromadb
from dotenv import load_dotenv
from google import genai


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(
    api_key=api_key
)


# -----------------------------
# Connect to ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="manufacturing_documents_v2"
)


# -----------------------------
# User question
# -----------------------------

question = "What are the different types of manufacturing processes?"


# -----------------------------
# Create embedding for question
# -----------------------------

result = gemini_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = result.embeddings[0].values


# -----------------------------
# Search ChromaDB
# -----------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=4,
    include=["documents", "metadatas", "distances"]
)


# -----------------------------
# Display results
# -----------------------------

print("Question:")
print(question)

print("\nMost relevant sections:\n")


for i, document in enumerate(results["documents"][0]):

    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print(f"--- Result {i + 1} ---")

    print("Source:", metadata["source"])
    print("Page:", metadata["page"])
    print("Distance:", distance)

    print("\nContent:")
    print(document)

    print()
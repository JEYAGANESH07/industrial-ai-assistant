
import chromadb

# Connect to your existing persistent database
client = chromadb.PersistentClient(path="./chroma_db")

# List all available collections
print("\n=== ChromaDB Collections ===")
for collection in client.list_collections():
    print(f"- {collection.name}")

# Inspect the main document-chunk collection
collection = client.get_collection(
    name="manufacturing_documents_local"
)

print("\n=== Collection Information ===")
print("Collection name:", collection.name)
print("Total chunks:", collection.count())

# Display a few stored records
data = collection.get(
    limit=3,
    include=["documents", "metadatas"]
)

print("\n=== Sample Stored Records ===")
for i, doc_id in enumerate(data["ids"]):
    print(f"\nRecord {i + 1}")
    print("ID:", doc_id)
    print("Metadata:", data["metadatas"][i])
    print("Text:", (data["documents"][i] or "")[:400])

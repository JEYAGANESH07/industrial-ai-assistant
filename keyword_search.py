import chromadb


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
# User keyword
# -----------------------------

keyword = "machining"


# -----------------------------
# Get all stored documents
# -----------------------------

results = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)


# -----------------------------
# Search for keyword
# -----------------------------

print("\nKeyword Search:")
print("Keyword:", keyword)

found = 0

for i, document in enumerate(results["documents"]):

    if keyword.lower() in document.lower():

        found += 1

        metadata = results["metadatas"][i]

        print("\n--- Match", found, "---")

        print(
            "Source:",
            metadata["source"]
        )

        print(
            "Page:",
            metadata["page"]
        )

        print("Content:")

        print(document)


# -----------------------------
# No results
# -----------------------------

if found == 0:

    print("\nNo matching documents found.")
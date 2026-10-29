import chromadb


chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = chroma_client.get_collection(
    name="manufacturing_documents_local"
)


results = collection.get(
    limit=20
)


print()
print("==============================")
print("CHECKING CLEANED CHROMADB TEXT")
print("==============================")
print()


for index, document in enumerate(
    results["documents"],
    start=1
):

    print(
        f"\n--- Chunk {index} ---"
    )

    print(
        document[:1000]
    )


print()
print("==============================")
print(
    f"Total chunks in collection: "
    f"{collection.count()}"
)
print("==============================")
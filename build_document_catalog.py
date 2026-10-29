"""One-time migration: create document-level vectors from existing chunks.

Use for your current small index before enabling document discovery. This
migration groups existing chunks by PDF and embeds a short per-document preview.
For huge collections, use an incremental migration strategy instead.
"""
import hashlib
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
CHUNK_COLLECTION = "manufacturing_documents_local"
CATALOG_COLLECTION = "manufacturing_document_catalog"


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chunks = client.get_collection(name=CHUNK_COLLECTION)
    catalog = client.get_or_create_collection(
        name=CATALOG_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    data = chunks.get(include=["documents", "metadatas"])
    grouped = defaultdict(lambda: {
        "category": "Unknown", "pages": set(), "chunks": 0, "samples": []
    })

    for document, metadata in zip(
        data.get("documents") or [],
        data.get("metadatas") or []
    ):
        metadata = metadata or {}
        source = metadata.get("source")
        if not source:
            continue
        entry = grouped[source]
        entry["category"] = metadata.get("category", entry["category"])
        page = metadata.get("page")
        if page is not None:
            entry["pages"].add(int(page))
        entry["chunks"] += 1
        if len(entry["samples"]) < 8 and document:
            entry["samples"].append(document[:700])

    if not grouped:
        print("No source documents found in the existing chunk collection.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    items = []
    for source, entry in grouped.items():
        text = (
            f"Document: {source}\n"
            f"Category: {entry['category']}\n"
            f"Pages: {len(entry['pages'])}\n"
            f"Content preview: " + "\n".join(entry["samples"])
        )[:7000]
        items.append((source, entry, text))

    batch_size = 64
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        texts = [item[2] for item in batch]
        vectors = model.encode(texts, convert_to_numpy=True).tolist()
        ids, metadatas = [], []
        for source, entry, _ in batch:
            ids.append("doc_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:24])
            metadatas.append({
                "source": source,
                "category": entry["category"],
                "pages": int(len(entry["pages"])),
                "chunks": int(entry["chunks"])
            })
        catalog.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        print(f"Indexed catalogue documents: {min(start + len(batch), len(items))}/{len(items)}")

    print(f"Document catalogue ready: {catalog.count()} PDFs.")


if __name__ == "__main__":
    main()

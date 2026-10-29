"""Document-level discovery for the Industrial AI RAG project."""
import chromadb

CHROMA_PATH = "./chroma_db"
CATALOG_NAME = "manufacturing_document_catalog"


def get_document_catalog():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=CATALOG_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def discover_documents(query_embedding, limit=10):
    """Return ranked candidate PDFs and their source names."""
    catalog = get_document_catalog()
    count = catalog.count()
    if count == 0:
        return [], []

    result = catalog.query(
        query_embeddings=[query_embedding],
        n_results=min(max(1, int(limit)), count),
        include=["metadatas", "distances", "documents"]
    )

    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]

    candidates, sources = [], []
    for metadata, distance, text in zip(metadatas, distances, documents):
        metadata = metadata or {}
        source = metadata.get("source")
        if not source:
            continue
        sources.append(source)
        candidates.append({
            "source": source,
            "category": metadata.get("category", "Unknown"),
            "pages": metadata.get("pages", 0),
            "chunks": metadata.get("chunks", 0),
            "distance": float(distance) if distance is not None else None,
            "catalog_preview": (text or "")[:500]
        })

    unique_sources = list(dict.fromkeys(sources))
    unique_docs, seen = [], set()
    for item in candidates:
        if item["source"] not in seen:
            unique_docs.append(item)
            seen.add(item["source"])
    return unique_sources, unique_docs

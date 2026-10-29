"""Batch-ingest PDFs from a local folder into the existing RAG index."""
from pathlib import Path

import chromadb

from upload_processor import process_uploaded_pdf

CHROMA_PATH = "./chroma_db"
CHUNK_COLLECTION = "manufacturing_documents_local"
CATALOG_COLLECTION = "manufacturing_document_catalog"


class NamedFile:
    """File wrapper giving the processor a stable relative source name."""
    def __init__(self, file_obj, name):
        self._file_obj = file_obj
        self.name = name

    def __getattr__(self, name):
        return getattr(self._file_obj, name)


def ingest_pdf_folder(folder_path, categorize=False, force_reindex=False):
    """Recursively ingest PDFs; skip indexed files unless force_reindex=True."""
    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"PDF folder does not exist or is not a directory: {root}")

    pdf_paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        return {"folder": str(root), "found": 0, "processed": [], "skipped": [], "errors": []}

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chunks = client.get_collection(name=CHUNK_COLLECTION)
    catalog = client.get_or_create_collection(
        name=CATALOG_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    processed, skipped, errors = [], [], []
    for pdf_path in pdf_paths:
        source_name = pdf_path.relative_to(root).as_posix()
        try:
            existing = catalog.get(
                where={"source": source_name},
                include=["metadatas"],
                limit=1
            )
            is_indexed = bool(existing.get("ids"))

            if is_indexed and not force_reindex:
                skipped.append(source_name)
                continue

            if is_indexed and force_reindex:
                chunks.delete(where={"source": source_name})
                catalog.delete(where={"source": source_name})

            with pdf_path.open("rb") as raw_file:
                named_file = NamedFile(raw_file, source_name)
                result = process_uploaded_pdf(named_file, categorize=categorize)
            processed.append(result)
        except Exception as exc:
            errors.append({"file": source_name, "error": str(exc)})

    return {
        "folder": str(root),
        "found": len(pdf_paths),
        "processed": processed,
        "skipped": skipped,
        "errors": errors
    }

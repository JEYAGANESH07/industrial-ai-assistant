import re
import chromadb
from pypdf import PdfReader
from embedding_model import get_embedding_model

from document_categorizer import categorize_document


# Shared model instance: also used by hybrid_search.py.
embedding_model = get_embedding_model()


chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="manufacturing_documents_local"
)

# One vector per PDF for document-level discovery.
document_catalog = chroma_client.get_or_create_collection(
    name="manufacturing_document_catalog",
    metadata={"hnsw:space": "cosine"}
)


def clean_extracted_text(text):
    """
    Clean PDF extraction artifacts while preserving
    ordinary word spacing and paragraph structure.
    """

    if not text:
        return ""

    # Normalize whitespace.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")

    # Remove encoded PDF glyph runs such as /G43/G48/G41/...
    text = re.sub(
        r"(?:/G[0-9A-Fa-f]{2,4}){3,}",
        " ",
        text
    )

    # Join hyphenated words split across lines.
    # Example: manufac-\nturing -> manufacturing
    text = re.sub(
        r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[A-Za-z])",
        "",
        text
    )

    # Repair confirmed word-splitting patterns.
    # These rules are intentionally specific to avoid
    # merging normal phrases such as "A station".
    known_repairs = {
        "th e": "the",
        "ass embly": "assembly",
        "opera tions": "operations",
        "relati onship": "relationship",
        "rela ti onship": "relationship",
        "v olumes": "volumes",
        "produc t": "product",
        "Wo rking": "Working",
        "wo rking": "working",
        "S chematic": "Schematic",
        "V arious": "Various",
        "T ailstock": "Tailstock",
        "W ork": "Work",
        "A rbor": "Arbor",
        "T able": "Table",
        "S traight": "Straight",
        "C ountersunk": "Countersunk",
        "R aised": "Raised",
    }

    for incorrect, corrected in known_repairs.items():
        text = re.sub(
            r"(?<!\w)" + re.escape(incorrect) + r"(?!\w)",
            corrected,
            text
        )

    # Join single line breaks while preserving paragraph breaks.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text
    )

    # Normalize repeated spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def create_chunks(text, chunk_size=1000, overlap=200):
    """
    Create overlapping chunks, preferring sentence boundaries.
    """

    text = clean_extracted_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller "
            "than chunk_size."
        )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # Split exceptionally long sentences by words.
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            words = sentence.split()
            part = ""

            for word in words:
                candidate = f"{part} {word}".strip()

                if len(candidate) <= chunk_size:
                    part = candidate
                else:
                    if part:
                        chunks.append(part)

                    part = word

            if part:
                current_chunk = part

            continue

        candidate = f"{current_chunk} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Preserve whole words from the previous chunk as overlap.
        overlap_words = []
        overlap_length = 0

        for word in reversed(current_chunk.split()):
            added_length = len(word) + (
                1 if overlap_words else 0
            )

            if overlap_length + added_length > overlap:
                break

            overlap_words.insert(0, word)
            overlap_length += added_length

        overlap_text = " ".join(overlap_words)

        current_chunk = (
            f"{overlap_text} {sentence}".strip()
        )

        if len(current_chunk) > chunk_size:
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def infer_category_locally(text, filename=""):
    """Assign a lightweight category without an LLM API call."""
    sample = f"{filename} {text[:12000]}".lower()
    rules = [
        ("Safety", ("safety", "hazard", "ppe", "accident", "risk assessment")),
        ("Maintenance", ("maintenance", "lubrication", "preventive maintenance", "repair")),
        ("Quality Control", ("quality control", "inspection", "defect", "tolerance", "quality assurance")),
        ("Machinery / Equipment", ("machine tool", "machinery", "equipment", "lathe", "milling machine")),
        ("Materials", ("material properties", "alloy", "polymer", "raw material", "composite")),
        ("Production", ("mass production", "batch production", "production planning", "production line")),
        ("Manufacturing Process", ("manufacturing process", "machining", "casting", "forming", "welding", "assembly")),
        ("Research Paper", ("abstract", "keywords", "methodology", "references", "doi")),
        ("Technical Manual", ("operating instructions", "user manual", "installation procedure", "troubleshooting")),
    ]
    for category, terms in rules:
        if any(term in sample for term in terms):
            return category
    return "Other"


def process_uploaded_pdf(uploaded_file, categorize=True):
    """Extract a PDF once, embed chunks in batches, and update both indexes."""
    import hashlib

    pdf_name = uploaded_file.name
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)

    # Extract each page only once. The old implementation extracted every page
    # once for categorization and a second time for chunking.
    page_texts = []
    full_text_parts = []
    for page in reader.pages:
        page_text = clean_extracted_text(page.extract_text() or "")
        page_texts.append(page_text)
        if page_text:
            full_text_parts.append(page_text)

    full_text = "\\n\\n".join(full_text_parts)
    if categorize:
        category = categorize_document(full_text)
    else:
        category = infer_category_locally(full_text, pdf_name)
    print(f"Document category: {category}")

    # Accumulate chunks across pages and encode/write in batches. This avoids
    # hundreds of tiny model calls for a long PDF while preserving page numbers.
    batch_size = 64
    pending_texts = []
    pending_ids = []
    pending_metadatas = []
    total_chunks = 0

    def flush_batch():
        nonlocal total_chunks
        if not pending_texts:
            return
        vectors = embedding_model.encode(
            pending_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
        collection.upsert(
            ids=pending_ids,
            documents=pending_texts,
            embeddings=vectors,
            metadatas=pending_metadatas
        )
        total_chunks += len(pending_texts)
        pending_texts.clear()
        pending_ids.clear()
        pending_metadatas.clear()

    for page_number, page_text in enumerate(page_texts, start=1):
        if not page_text:
            continue
        chunks = create_chunks(page_text)
        for chunk_number, chunk in enumerate(chunks, start=1):
            pending_texts.append(chunk)
            pending_ids.append(f"{pdf_name}_page_{page_number}_chunk_{chunk_number}")
            pending_metadatas.append({
                "source": pdf_name,
                "page": int(page_number),
                "chunk": int(chunk_number),
                "category": category
            })
            if len(pending_texts) >= batch_size:
                flush_batch()
    flush_batch()

    # One vector per PDF supports document-level discovery.
    catalog_text = (
        f"Document: {pdf_name}\\n"
        f"Category: {category}\\n"
        f"Pages: {total_pages}\\n"
        f"Content preview: {full_text[:6000]}"
    )
    catalog_embedding = embedding_model.encode(
        catalog_text,
        convert_to_numpy=True,
    ).tolist()
    catalog_id = "doc_" + hashlib.sha256(
        pdf_name.encode("utf-8")
    ).hexdigest()[:24]
    document_catalog.upsert(
        ids=[catalog_id],
        documents=[catalog_text],
        embeddings=[catalog_embedding],
        metadatas=[{
            "source": pdf_name,
            "category": category,
            "pages": int(total_pages),
            "chunks": int(total_chunks)
        }]
    )

    return {
        "file_name": pdf_name,
        "pages": total_pages,
        "chunks": total_chunks,
        "category": category
    }

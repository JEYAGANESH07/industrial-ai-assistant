
import re
import hashlib
import chromadb

from pypdf import PdfReader
from embedding_model import get_embedding_model
from document_categorizer import categorize_document


# --------------------------------------------------
# 1. INITIALIZE EMBEDDING MODEL
# --------------------------------------------------

# Shared model instance, also used by hybrid_search.py.
embedding_model = get_embedding_model()


# --------------------------------------------------
# 2. INITIALIZE CHROMADB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Store document chunks.
collection = chroma_client.get_or_create_collection(
    name="manufacturing_documents_local"
)

# Store one vector per PDF for document discovery.
document_catalog = chroma_client.get_or_create_collection(
    name="manufacturing_document_catalog",
    metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# 3. CLEAN EXTRACTED PDF TEXT
# --------------------------------------------------

def clean_extracted_text(text):
    """
    Clean PDF extraction artifacts while preserving
    ordinary word spacing and paragraph structure.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")

    # Remove encoded PDF glyph runs.
    text = re.sub(
        r"(?:/G[0-9A-Fa-f]{2,4}){3,}",
        " ",
        text
    )

    # Join words split by a hyphen and line break.
    text = re.sub(
        r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[A-Za-z])",
        "",
        text
    )

    # Repair known PDF word-splitting artifacts.
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

    # Join single line breaks but preserve paragraphs.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 4. CREATE OVERLAPPING TEXT CHUNKS
# --------------------------------------------------

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

        # Preserve whole words from the previous chunk.
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


# --------------------------------------------------
# 5. LOCAL DOCUMENT CATEGORIZATION FALLBACK
# --------------------------------------------------

def infer_category_locally(text, filename=""):
    """
    Assign a category using local keyword rules.
    This function does not call an external AI API.
    """

    sample = f"{filename} {text[:12000]}".lower()

    rules = [
        (
            "Safety",
            (
                "safety",
                "hazard",
                "ppe",
                "accident",
                "risk assessment"
            )
        ),
        (
            "Maintenance",
            (
                "maintenance",
                "lubrication",
                "preventive maintenance",
                "repair"
            )
        ),
        (
            "Quality Control",
            (
                "quality control",
                "inspection",
                "defect",
                "tolerance",
                "quality assurance"
            )
        ),
        (
            "Machinery / Equipment",
            (
                "machine tool",
                "machinery",
                "equipment",
                "lathe",
                "milling machine"
            )
        ),
        (
            "Materials",
            (
                "material properties",
                "alloy",
                "polymer",
                "raw material",
                "composite"
            )
        ),
        (
            "Production",
            (
                "mass production",
                "batch production",
                "production planning",
                "production line"
            )
        ),
        (
            "Manufacturing Process",
            (
                "manufacturing process",
                "machining",
                "casting",
                "forming",
                "welding",
                "assembly"
            )
        ),
        (
            "Research Paper",
            (
                "abstract",
                "keywords",
                "methodology",
                "references",
                "doi"
            )
        ),
        (
            "Technical Manual",
            (
                "operating instructions",
                "user manual",
                "installation procedure",
                "troubleshooting"
            )
        ),
    ]

    for category, terms in rules:
        if any(term in sample for term in terms):
            return category

    return "Other"


# --------------------------------------------------
# 6. PROCESS AN UPLOADED PDF
# --------------------------------------------------

def process_uploaded_pdf(uploaded_file, categorize=True):
    """
    Extract PDF text, create chunks, generate embeddings,
    update ChromaDB, and update the document catalog.

    If AI categorization fails, local categorization is used.
    """

    pdf_name = uploaded_file.name

    print(f"Starting PDF processing: {pdf_name}")

    # ----------------------------------------------
    # A. EXTRACT PDF TEXT
    # ----------------------------------------------

    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)

    page_texts = []
    full_text_parts = []

    for page in reader.pages:
        page_text = clean_extracted_text(
            page.extract_text() or ""
        )

        page_texts.append(page_text)

        if page_text:
            full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts)

    if not full_text.strip():
        raise ValueError(
            f"No extractable text was found in {pdf_name}. "
            "The PDF may be scanned or image-only."
        )

    # ----------------------------------------------
    # B. CATEGORIZE DOCUMENT
    # ----------------------------------------------

    if categorize:
        try:
            category = categorize_document(full_text)

            if not isinstance(category, str) or not category.strip():
                raise ValueError(
                    "AI categorization returned an empty category."
                )

        except Exception as e:
            print(
                "AI categorization failed. "
                f"Using local rules instead. Error: {e}"
            )

            category = infer_category_locally(
                full_text,
                pdf_name
            )
    else:
        category = infer_category_locally(
            full_text,
            pdf_name
        )

    print(f"Document category: {category}")

    # ----------------------------------------------
    # C. PREPARE BATCHED CHUNKS
    # ----------------------------------------------

    batch_size = 64

    pending_texts = []
    pending_ids = []
    pending_metadatas = []

    total_chunks = 0

    def flush_batch():
        nonlocal total_chunks

        if not pending_texts:
            return

        # Generate embeddings locally using the existing
        # shared embedding model.
        vectors = embedding_model.encode(
            pending_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        # Save document chunks and embeddings.
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

    # ----------------------------------------------
    # D. CHUNK AND INDEX EACH PAGE
    # ----------------------------------------------

    for page_number, page_text in enumerate(
        page_texts,
        start=1
    ):
        if not page_text:
            continue

        chunks = create_chunks(page_text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):
            pending_texts.append(chunk)

            pending_ids.append(
                f"{pdf_name}_page_{page_number}_chunk_{chunk_number}"
            )

            pending_metadatas.append({
                "source": pdf_name,
                "page": int(page_number),
                "chunk": int(chunk_number),
                "category": category
            })

            if len(pending_texts) >= batch_size:
                flush_batch()

    # Index any remaining chunks.
    flush_batch()

    if total_chunks == 0:
        raise ValueError(
            f"No text chunks were created for {pdf_name}."
        )

    # ----------------------------------------------
    # E. UPDATE DOCUMENT-LEVEL CATALOG
    # ----------------------------------------------

    catalog_text = (
        f"Document: {pdf_name}\n"
        f"Category: {category}\n"
        f"Pages: {total_pages}\n"
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

    # ----------------------------------------------
    # F. RETURN PROCESSING SUMMARY
    # ----------------------------------------------

    print(
        f"Successfully indexed {pdf_name}: "
        f"{total_pages} pages, {total_chunks} chunks."
    )

    return {
        "file_name": pdf_name,
        "pages": total_pages,
        "chunks": total_chunks,
        "category": category
    }

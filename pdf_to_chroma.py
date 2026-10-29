import re
from pathlib import Path

import chromadb

from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from wordfreq import zipf_frequency

from document_categorizer import categorize_document


print("Loading local embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Local embedding model loaded successfully.")


chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# Delete the old collection if it exists
try:

    chroma_client.delete_collection(
        name="manufacturing_documents_local"
    )

    print(
        "Old ChromaDB collection deleted."
    )

except Exception:

    print(
        "No previous collection found."
    )


collection = chroma_client.create_collection(
    name="manufacturing_documents_local"
)

print(
    "New ChromaDB collection created."
)


# ------------------------------------------------
# Generic split-word repair
# ------------------------------------------------

def word_score(word):
    """
    Return how likely a string is to be
    a normal English word.
    """

    return zipf_frequency(
        word.lower(),
        "en"
    )


def should_join(parts):
    """
    Decide whether multiple adjacent pieces
    are likely to be one incorrectly split word.
    """

    if len(parts) < 2:

        return False

    joined = "".join(
        parts
    ).lower()

    # Avoid creating extremely long words.
    if len(joined) > 30:

        return False

    joined_score = word_score(
        joined
    )

    # Joined word must be reasonably likely
    # to be a real English word.
    if joined_score < 3.0:

        return False

    part_scores = [
        word_score(part)
        for part in parts
    ]

    # If one fragment is very weak,
    # joining is more likely to be correct.
    has_weak_fragment = any(
        score < 2.5
        for score in part_scores
    )

    if has_weak_fragment:

        return True

    highest_part_score = max(
        part_scores
    )

    # Otherwise, the joined word should be
    # substantially more likely than the
    # individual fragments.
    if (
        joined_score
        >= highest_part_score + 1.0
    ):

        return True

    return False


def repair_split_words(text):
    """
    Repair arbitrary PDF split-word artifacts.

    Examples:

        ass embly -> assembly
        relati on -> relation
        opera tions -> operations
        compo nent -> component
        v olumes -> volumes
        Wo rking -> Working

    The function does not use a hard-coded
    list of words.
    """

    words = text.split()

    result = []

    i = 0

    while i < len(words):

        best_match = None

        best_length = 0

        best_score = 0

        # Try 2, 3 and 4 adjacent pieces.
        for number_of_parts in range(
            2,
            5
        ):

            if (
                i + number_of_parts
                > len(words)
            ):

                break

            candidate_parts = words[
                i:i + number_of_parts
            ]

            # Only consider alphabetic pieces.
            if not all(
                part.isalpha()
                for part in candidate_parts
            ):

                continue

            if not should_join(
                candidate_parts
            ):

                continue

            joined = "".join(
                candidate_parts
            )

            score = word_score(
                joined
            )

            if score > best_score:

                best_match = joined

                best_length = (
                    number_of_parts
                )

                best_score = score

        if best_match is not None:

            result.append(
                best_match
            )

            i += best_length

        else:

            result.append(
                words[i]
            )

            i += 1

    return " ".join(
        result
    )


def clean_pdf_text(text):
    """
    General PDF text cleaning.
    """

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Repair words broken at the end
    # of a PDF line.
    #
    # Example:
    #
    # manu-
    # facturing
    #
    # becomes:
    #
    # manufacturing
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Replace remaining newlines
    # with spaces.
    text = re.sub(
        r"\s*\n\s*",
        " ",
        text
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    # Repair arbitrary split words.
    text = repair_split_words(
        text
    )

    return text


# ------------------------------------------------
# Chunk creation
# ------------------------------------------------

def create_chunks(
    text,
    chunk_size=1000
):
    """
    Create fixed-size chunks.
    """

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

        if chunk.strip():

            chunks.append(
                chunk
            )

    return chunks


# ------------------------------------------------
# Weak chunk filtering
# ------------------------------------------------

def is_useful_chunk(chunk):
    """
    Remove chunks that contain almost
    no useful textual information.
    """

    if len(chunk.strip()) < 80:

        return False

    alphabetic_characters = sum(
        character.isalpha()
        for character in chunk
    )

    if alphabetic_characters < 40:

        return False

    return True


# ------------------------------------------------
# Find PDF files
# ------------------------------------------------

pdf_files = list(
    Path(
        "./documents"
    ).glob("*.pdf")
)


print(
    f"Found {len(pdf_files)} PDF files."
)


total_chunks = 0


# ------------------------------------------------
# Process every PDF
# ------------------------------------------------

for pdf_file in pdf_files:

    print(
        f"\nProcessing: {pdf_file.name}"
    )

    reader = PdfReader(
        str(pdf_file)
    )

    total_pages = len(
        reader.pages
    )

    print(
        f"Pages: {total_pages}"
    )


    # ------------------------------------------------
    # Extract and clean complete document
    # ------------------------------------------------

    full_text = ""

    for page in reader.pages:

        page_text = (
            page.extract_text()
        )

        if page_text:

            cleaned_page = (
                clean_pdf_text(
                    page_text
                )
            )

            full_text += (
                cleaned_page
                + " "
            )


    # ------------------------------------------------
    # Categorize document
    # ------------------------------------------------

    category = categorize_document(
        full_text
    )

    print(
        f"Category: {category}"
    )


    # ------------------------------------------------
    # Process each page
    # ------------------------------------------------

    document_chunks = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = (
            page.extract_text()
        )

        if not page_text:

            continue


        # Clean page
        page_text = clean_pdf_text(
            page_text
        )


        # Create chunks
        chunks = create_chunks(
            page_text
        )


        # Remove weak chunks
        chunks = [
            chunk
            for chunk in chunks
            if is_useful_chunk(
                chunk
            )
        ]


        if not chunks:

            continue


        print(
            f"  Page {page_number}: "
            f"{len(chunks)} chunks"
        )


        # ------------------------------------------------
        # Generate local embeddings
        # ------------------------------------------------

        embeddings = (
            embedding_model.encode(
                chunks,
                convert_to_numpy=True
            ).tolist()
        )


        batch_ids = []

        batch_metadatas = []


        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            chunk_id = (
                f"{pdf_file.name}_"
                f"page_{page_number}_"
                f"chunk_{chunk_number}"
            )


            metadata = {
                "source": pdf_file.name,
                "page": page_number,
                "chunk": chunk_number,
                "category": category
            }


            batch_ids.append(
                chunk_id
            )

            batch_metadatas.append(
                metadata
            )


        # ------------------------------------------------
        # Store in ChromaDB
        # ------------------------------------------------

        collection.upsert(
            ids=batch_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=batch_metadatas
        )


        document_chunks += len(
            chunks
        )


    print(
        f"Stored chunks: "
        f"{document_chunks}"
    )


    total_chunks += (
        document_chunks
    )


# ------------------------------------------------
# Finished
# ------------------------------------------------

print(
    "\n------------------------------"
)

print(
    "PDF processing completed."
)

print(
    f"Total chunks stored: "
    f"{total_chunks}"
)

print(
    "ChromaDB collection rebuilt successfully."
)

print(
    "------------------------------"
)
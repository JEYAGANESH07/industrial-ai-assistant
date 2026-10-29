
import os
import re
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from embedding_model import get_embedding_model

from document_discovery import discover_documents


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# LOCAL EMBEDDING MODEL
# --------------------------------------------------

# Reuse the same in-process model as PDF ingestion.
embedding_model = get_embedding_model()


# --------------------------------------------------
# GEMINI SETUP
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please check your .env file."
    )

gemini_client = genai.Client(api_key=api_key)


# --------------------------------------------------
# CHROMADB SETUP
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="manufacturing_documents_local"
)


# --------------------------------------------------
# STOPWORDS
# --------------------------------------------------

STOPWORDS = {
    "what", "is", "are", "the", "a", "an",
    "of", "to", "in", "on", "for", "and",
    "or", "with", "from", "by", "how", "why",
    "which", "can", "does", "do", "different",
    "types", "kind", "kinds", "please", "explain",
    "describe", "list", "give", "me"
}


# --------------------------------------------------
# EXTRACT KEYWORDS
# --------------------------------------------------

def extract_keywords(query):
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    return [
        word for word in words
        if word not in STOPWORDS
    ]


# --------------------------------------------------
# NORMALIZE TEXT FOR COMPARISON
# --------------------------------------------------

def normalize_text(text):
    text = (text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def get_content_words(text):
    return set(
        re.findall(
            r"\b[a-z0-9]+\b",
            normalize_text(text)
        )
    )


# --------------------------------------------------
# DETECT WEAK OR TITLE CHUNKS
# --------------------------------------------------

def is_weak_chunk(document):
    text = (document or "").strip()
    text_lower = text.lower()

    if len(text.split()) < 15:
        return True

    title_indicators = [
        "second edition",
        "visiting professor",
        "professor and head",
        "department of mechanical engineering",
        "technical university",
        "b.sc.",
        "b.e.",
        "m.tech.",
        "ph.d."
    ]

    indicator_count = sum(
        1
        for indicator in title_indicators
        if indicator in text_lower
    )

    return indicator_count >= 2


# --------------------------------------------------
# KEYWORD SEARCH
# --------------------------------------------------

def keyword_search(query, candidate_sources=None):
    keywords = extract_keywords(query)

    if not keywords:
        return {}

    # If document discovery found candidate PDFs, keyword search only
    # their chunks instead of scanning the entire corpus on every query.
    get_args = {"include": ["documents", "metadatas"]}
    if candidate_sources:
        get_args["where"] = {"source": {"$in": candidate_sources}}

    all_data = collection.get(**get_args)

    documents = all_data["documents"] or []
    metadatas = all_data["metadatas"] or []
    ids = all_data["ids"] or []

    if not documents:
        return {}

    document_frequency = {}

    for keyword in keywords:
        document_frequency[keyword] = sum(
            1
            for document in documents
            if keyword in document.lower()
        )

    keyword_results = {}

    total_documents = max(len(documents), 1)

    for chunk_id, document, metadata in zip(
        ids,
        documents,
        metadatas
    ):
        document_lower = document.lower()
        score = 0.0

        for keyword in keywords:
            if keyword not in document_lower:
                continue

            frequency = document_lower.count(keyword)

            df = max(
                document_frequency.get(keyword, 1),
                1
            )

            # Give relatively uncommon query terms
            # more weight than very common terms.
            rarity_weight = (
                1.0
                + (total_documents / df) ** 0.5
            )

            # Limit the impact of repeated words.
            frequency_weight = min(frequency, 4)

            score += frequency_weight * rarity_weight

        if score > 0:
            keyword_results[chunk_id] = {
                "keyword_score": score,
                "document": document,
                "metadata": metadata or {}
            }

    if keyword_results:
        max_score = max(
            item["keyword_score"]
            for item in keyword_results.values()
        )

        if max_score > 0:
            for item in keyword_results.values():
                item["keyword_score"] /= max_score

    return keyword_results


# --------------------------------------------------
# CALCULATE TOPIC OVERLAP
# --------------------------------------------------

def calculate_topic_overlap(document, keywords):
    if not keywords:
        return 0.0

    document_words = get_content_words(document)

    matched = sum(
        1
        for keyword in keywords
        if keyword in document_words
    )

    return matched / len(keywords)


# --------------------------------------------------
# CLASSIFICATION RELEVANCE
# --------------------------------------------------

def classification_relevance(document, query):
    text = normalize_text(document)
    query_lower = normalize_text(query)

    score = 0.0

    classification_phrases = [
        "classification of manufacturing processes",
        "manufacturing processes can be classified",
        "processes can be classified",
        "classified into",
        "classified as",
        "can be divided into",
        "following types",
        "types of manufacturing processes",
        "main manufacturing processes",
        "processing operation",
        "assembly operation"
    ]

    for phrase in classification_phrases:
        if phrase in text:
            score += 0.12

    list_indicators = [
        "1.",
        "2.",
        "3.",
        "•",
        "include",
        "includes"
    ]

    for indicator in list_indicators:
        if indicator in text:
            score += 0.025

    asks_for_classification = any(
        phrase in query_lower
        for phrase in [
            "types",
            "kinds",
            "categories",
            "classification",
            "classify",
            "classified"
        ]
    )

    if asks_for_classification:
        score *= 1.5

    return min(score, 0.35)


# --------------------------------------------------
# DETECT PRODUCTION-SCALE QUESTIONS
# --------------------------------------------------

def is_production_scale_question(query):
    query_lower = normalize_text(query)

    scale_terms = [
        "scale of production",
        "job production",
        "job lot production",
        "batch production",
        "mass production",
        "production volume",
        "production quantity"
    ]

    return any(
        term in query_lower
        for term in scale_terms
    )


# --------------------------------------------------
# REMOVE DUPLICATE AND OVERLAPPING PASSAGES
# --------------------------------------------------



def remove_duplicate_results(results):
    """
    Remove exact duplicates and highly overlapping passages.
    Be stricter when passages come from the same PDF page.
    """

    unique_results = []
    seen_items = []

    for chunk_id, result in results:
        content = normalize_text(
            result.get("document", "")
        )

        if not content:
            continue

        words = get_content_words(content)

        if not words:
            continue

        metadata = result.get("metadata", {})
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        duplicate = False

        for (
            previous_text,
            previous_words,
            previous_source,
            previous_page,
        ) in seen_items:

            # Remove exact duplicates.
            if content == previous_text:
                duplicate = True
                break

            # Remove nearly identical passages.
            
            # Fast token-overlap comparison instead of SequenceMatcher.
            smaller_size = min(len(words), len(previous_words))

            if smaller_size >= 15:
                overlap = len(words & previous_words) / smaller_size

                if overlap >= 0.90:
                    duplicate = True
                    break


            # Remove heavily overlapping chunks from the same page.
            if source == previous_source and page == previous_page:
                smaller_size = min(
                    len(words),
                    len(previous_words)
                )

                if smaller_size >= 10:
                    overlap = (
                        len(words & previous_words)
                        / smaller_size
                    )

                    if overlap >= 0.75:
                        duplicate = True
                        break

        if duplicate:
            continue

        unique_results.append((chunk_id, result))

        seen_items.append(
            (
                content,
                words,
                source,
                page
            )
        )

    return unique_results



# --------------------------------------------------
# SELECT DIVERSE RESULTS
# --------------------------------------------------

def select_diverse_results(
    results,
    max_results=8,
    max_per_page=2,
    max_per_source=6
):
    """
    Prefer high-ranking passages while limiting
    repeated pages and excessive reliance on one PDF.

    The first pass enforces both page and source limits.
    The second pass relaxes the page limit if needed.
    """

    selected = []
    selected_ids = set()
    page_counts = {}
    source_counts = {}

    def get_source_page(item):
        chunk_id, result = item

        metadata = result.get("metadata", {})

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        return source, page

    # Pass 1: enforce diversity limits.
    for item in results:
        chunk_id, result = item

        source, page = get_source_page(item)
        page_key = (source, page)

        if page_counts.get(page_key, 0) >= max_per_page:
            continue

        if source_counts.get(source, 0) >= max_per_source:
            continue

        selected.append(item)
        selected_ids.add(chunk_id)

        page_counts[page_key] = (
            page_counts.get(page_key, 0) + 1
        )

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(selected) >= max_results:
            return selected

    # Pass 2: relax page limit by one if needed.
    for item in results:
        chunk_id, result = item

        if chunk_id in selected_ids:
            continue

        source, page = get_source_page(item)
        page_key = (source, page)

        if page_counts.get(page_key, 0) >= max_per_page + 1:
            continue

        if source_counts.get(source, 0) >= max_per_source:
            continue

        selected.append(item)
        selected_ids.add(chunk_id)

        page_counts[page_key] = (
            page_counts.get(page_key, 0) + 1
        )

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(selected) >= max_results:
            break

    return selected


# --------------------------------------------------
# CALCULATE RETRIEVAL CONFIDENCE
# --------------------------------------------------

def calculate_retrieval_confidence(results):
    """
    This is a heuristic retrieval score, not a calibrated
    probability that the generated answer is correct.
    """

    if not results:
        return 0.0

    scores = [
        result.get("hybrid_score", 0.0)
        for _, result in results
    ]

    average_score = sum(scores) / len(scores)
    highest_score = max(scores)

    result_factor = min(len(results) / 6, 1.0)

    confidence = (
        0.60 * average_score
        + 0.25 * highest_score
        + 0.15 * result_factor
    )

    return max(0.0, min(confidence, 1.0))


# --------------------------------------------------
# CONFIDENCE LABEL
# --------------------------------------------------

def get_confidence_label(confidence):
    if confidence >= 0.80:
        return "High"

    if confidence >= 0.60:
        return "Medium"

    return "Low"


# --------------------------------------------------
# SEARCH AND GENERATE ANSWER
# --------------------------------------------------

def search_and_generate_answer(query):
    import time
    total_start = time.perf_counter()
    print("\n--- RAG PERFORMANCE ---")
    # --------------------------------------------------
    # HANDLE EMPTY KNOWLEDGE BASE
    # --------------------------------------------------

    total_chunks = collection.count()

    keywords = extract_keywords(query)

    if total_chunks == 0:
        return {
            "answer": (
                "The knowledge base is empty. "
                "Please upload and process a PDF "
                "before asking a research question."
            ),
            "results": [],
            "keywords": keywords,
            "confidence": 0.0,
            "confidence_label": "Low",
            "discovered_documents": [],
            "document_discovery_enabled": False
        }

    # --------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    ).tolist()
    print(
    f"Query embedding: "
    f"{time.perf_counter() - total_start:.2f}s"
    )
    # --------------------------------------------------
    # DOCUMENT-LEVEL DISCOVERY
    # --------------------------------------------------

    candidate_sources, discovered_documents = discover_documents(
        query_embedding,
        limit=10
    )
    print(
    f"Document discovery: "
    f"{time.perf_counter() - total_start:.2f}s"
    )
    # An empty catalogue means this is a legacy index that has not
    # been migrated yet. Preserve the old search behavior in that case.
    discovery_enabled = bool(candidate_sources)

    # --------------------------------------------------
    # SEMANTIC CHUNK SEARCH WITHIN CANDIDATE DOCUMENTS
    # --------------------------------------------------

    semantic_query_args = {
        "query_embeddings": [query_embedding],
        "n_results": min(20, total_chunks),
        "include": ["documents", "metadatas", "distances"]
    }

    if discovery_enabled:
        semantic_query_args["where"] = {
            "source": {"$in": candidate_sources}
        }

    semantic_results = collection.query(**semantic_query_args)

    semantic_ids = semantic_results["ids"][0]
    semantic_documents = semantic_results["documents"][0]
    semantic_metadatas = semantic_results["metadatas"][0]
    semantic_distances = semantic_results["distances"][0]

    semantic_data = {}

    for chunk_id, document, metadata, distance in zip(
        semantic_ids,
        semantic_documents,
        semantic_metadatas,
        semantic_distances
    ):
        if not document or not metadata:
            continue

        semantic_score = 1.0 / (1.0 + max(distance, 0.0))

        semantic_data[chunk_id] = {
            "semantic_score": semantic_score,
            "document": document,
            "metadata": metadata
        }

    # --------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------

    keyword_data = keyword_search(
        query,
        candidate_sources=candidate_sources if discovery_enabled else None
    )
    print(
    f"Keyword search: "
    f"{time.perf_counter() - total_start:.2f}s"
    )
    # --------------------------------------------------
    # COMBINE SEMANTIC AND KEYWORD CANDIDATES
    # --------------------------------------------------

    all_chunk_ids = (
        set(semantic_data.keys())
        | set(keyword_data.keys())
    )

    hybrid_results = []

    for chunk_id in all_chunk_ids:

        semantic_item = semantic_data.get(chunk_id, {})
        keyword_item = keyword_data.get(chunk_id, {})

        semantic_score = semantic_item.get(
            "semantic_score", 0.0
        )

        keyword_score = keyword_item.get(
            "keyword_score", 0.0
        )

        document = (
            semantic_item.get("document")
            or keyword_item.get("document")
        )

        metadata = (
            semantic_item.get("metadata")
            or keyword_item.get("metadata")
        )

        if not document or metadata is None:
            continue

        # ----------------------------------------------
        # BASE HYBRID SCORE
        # ----------------------------------------------

        hybrid_score = (
            0.70 * semantic_score
            + 0.30 * keyword_score
        )

        # ----------------------------------------------
        # WEAK CHUNK PENALTY
        # ----------------------------------------------

        if is_weak_chunk(document):
            hybrid_score *= 0.70

        # ----------------------------------------------
        # TOPIC OVERLAP
        # ----------------------------------------------

        topic_overlap = calculate_topic_overlap(
            document,
            keywords
        )

        hybrid_score += 0.10 * topic_overlap

        # ----------------------------------------------
        # CLASSIFICATION RELEVANCE
        # ----------------------------------------------

        classification_score = classification_relevance(
            document,
            query
        )

        hybrid_score += classification_score

        hybrid_results.append(
            (
                chunk_id,
                {
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                    "document": document,
                    "metadata": metadata
                }
            )
        )

    # --------------------------------------------------
    # NORMALIZE FINAL RELEVANCE SCORES
    # --------------------------------------------------

    if hybrid_results:
        max_score = max(
            result["hybrid_score"]
            for _, result in hybrid_results
        )

        if max_score > 0:
            for _, result in hybrid_results:
                result["hybrid_score"] /= max_score

    # --------------------------------------------------
    # SORT RESULTS BY RELEVANCE
    # --------------------------------------------------

    sorted_hybrid_results = sorted(
        hybrid_results,
        key=lambda item: item[1]["hybrid_score"],
        reverse=True
    )

    # --------------------------------------------------
    # FILTER LOW-RELEVANCE RESULTS
    # --------------------------------------------------

    relevance_threshold = 0.35

    filtered_results = [
        (chunk_id, result)
        for chunk_id, result in sorted_hybrid_results
        if result["hybrid_score"] >= relevance_threshold
    ]

    # --------------------------------------------------
    # REMOVE DUPLICATE PASSAGES
    # --------------------------------------------------

    unique_results = remove_duplicate_results(
        filtered_results
    )
    print(
    f"Deduplication: "
    f"{time.perf_counter() - total_start:.2f}s"
    )
    # --------------------------------------------------
    # SELECT TOP RESULTS WITH SOURCE DIVERSITY
    # --------------------------------------------------

    top_results = select_diverse_results(
        unique_results,
        max_results=8,
        max_per_page=2,
        max_per_source=6
    )

    # --------------------------------------------------
    # CALCULATE RETRIEVAL CONFIDENCE
    # --------------------------------------------------

    confidence = calculate_retrieval_confidence(
        top_results
    )

    confidence_label = get_confidence_label(
        confidence
    )

    # --------------------------------------------------
    # HANDLE NO RELEVANT RESULTS
    # --------------------------------------------------

    if not top_results:
        return {
            "answer": (
                "No sufficiently relevant information "
                "was found in the retrieved documents. "
                "Try rephrasing your question or uploading "
                "a document that covers this topic."
            ),
            "results": [],
            "keywords": keywords,
            "confidence": 0.0,
            "confidence_label": "Low",
            "discovered_documents": discovered_documents,
            "document_discovery_enabled": discovery_enabled
        }

    # --------------------------------------------------
    # BUILD CITATION-AWARE CONTEXT
    # --------------------------------------------------

    context_parts = []

    for index, (chunk_id, result) in enumerate(
        top_results,
        start=1
    ):
        metadata = result["metadata"]

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")
        category = metadata.get("category", "Unknown")

        context_parts.append(
            f"""
[Source {index}]
Document: {source}
Page: {page}
Category: {category}

Citation format: [Source {index}, p. {page}]

Retrieved content:
{result["document"]}
"""
        )

    context = "\n".join(context_parts)

    # --------------------------------------------------
    # GENERATE A FOCUSED RESEARCH ANSWER
    # --------------------------------------------------

    production_scale_question = is_production_scale_question(
        query
    )

    prompt = f"""
You are an industrial research assistant specializing
in manufacturing engineering and industrial knowledge.

Answer the user's question using ONLY the retrieved
document context supplied below.

USER QUESTION:
{query}

RETRIEVED DOCUMENT CONTEXT:
{context}

ANSWERING RULES:

1. Answer the exact question directly.

2. Use only facts supported by the retrieved passages.
   Do not invent examples, technical details, or citations.

3. Cite claims immediately using the exact source number
   and page shown in the context, for example:
   [Source 4, p. 5]

4. Never invent a source number or page number.

5. Distinguish between these different concepts:

   A. Main manufacturing-process classifications:
      processing operations and assembly operations.

   B. Specific manufacturing processes:
      casting, forming, machining, joining, finishing,
      rapid prototyping, heat treating, and other
      operations, but only where the sources support them.

   C. Production-scale classifications:
      job or piece-lot production, batch production,
      and mass production.

6. Do not treat production-scale classifications as
   manufacturing processes. Include them in a separate
   section only if they are relevant to the question
   and supported by the retrieved passages.

7. Do not treat a specialized technique as a main
   category unless the source explicitly supports that
   classification.

8. Prioritize passages that directly answer the user's
   question over general background information.

9. When asked for types or classifications, organize
   the supported types into a clear, non-repetitive list.

10. Explain each type briefly using relevant evidence.
    Include examples only when they appear in the sources.

11. If the evidence describes both a broad classification
    and specific process types, explain the difference
    between those levels.

12. If the documents provide only a partial list, say so.
    Do not claim a list is exhaustive without evidence.

13. Do not repeat the same point under multiple headings.

14. Do not cite a passage unless it supports the claim.

15. If information is insufficient, state that limitation
    instead of guessing.

16. Do not include irrelevant document metadata.

17. Return a clear, technically accurate answer suitable
    for an industrial research report.

The user is asking specifically about production scales:
{production_scale_question}

If the answer is about manufacturing processes rather than
production scales, keep the production-scale discussion
separate and brief.

Use headings and concise bullet points where appropriate.
"""

    # --------------------------------------------------
    # GENERATE ANSWER WITH RETRIES
    # --------------------------------------------------

    response = None

    for attempt in range(3):
        try:
            response = (
                gemini_client
                .models
                .generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )
            )

            break

        except Exception as error:
            print(
                f"Gemini request failed "
                f"(attempt {attempt + 1}/3):",
                error
            )

            if attempt < 2:
                time.sleep(5)
    
    print(
        f"Gemini generation: "
        f"{time.perf_counter() - total_start:.2f}s"
    )
    print(
        f"TOTAL RAG TIME: "
        f"{time.perf_counter() - total_start:.2f}s"
    )
    print("-----------------------\n")

    # --------------------------------------------------
    # HANDLE FAILED GENERATION
    # --------------------------------------------------

    if response is None:
        answer = (
            "Unable to generate an answer because "
            "the Gemini request failed. Please check "
            "your API configuration or try again."
        )

    elif not response.text:
        answer = (
            "Gemini returned an empty response. "
            "Please try asking your question again."
        )

    else:
        answer = response.text

    # --------------------------------------------------
    # RETURN RESULTS TO STREAMLIT
    # --------------------------------------------------

    return {
        "answer": answer,
        "results": top_results,
        "keywords": keywords,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "discovered_documents": discovered_documents,
        "document_discovery_enabled": discovery_enabled
    }

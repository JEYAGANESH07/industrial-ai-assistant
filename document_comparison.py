import os

from dotenv import load_dotenv
from google import genai

from hybrid_search import embedding_model, collection


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(
    api_key=api_key
)


def compare_documents(document_names):
    """
    Compare the selected documents using
    their stored ChromaDB content.
    """

    if len(document_names) < 2:
        return {
            "answer": "Please select at least two documents for comparison.",
            "sources": []
        }

    comparison_query = (
        "important information about the document, "
        "manufacturing processes, methods, operations, "
        "materials, equipment, advantages, limitations, "
        "and other important technical information"
    )

    query_embedding = embedding_model.encode(
        comparison_query,
        convert_to_numpy=True
    ).tolist()

    # Retrieve relevant chunks only from
    # the selected documents.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where={
            "source": {
                "$in": document_names
            }
        }
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return {
            "answer": (
                "No relevant content was found "
                "in the selected documents."
            ),
            "sources": []
        }

    context_parts = []

    for index, (
        chunk_id,
        document,
        metadata
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas
        ),
        start=1
    ):

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        category = metadata.get(
            "category",
            "Unknown"
        )

        context_parts.append(
            f"""
[Document {index}]
Document: {source}
Category: {category}
Page: {page}

Content:
{document}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are an industrial research assistant.

Compare the selected industrial documents
using ONLY the retrieved document content.

Selected documents:
{", ".join(document_names)}

Retrieved document content:

{context}

Instructions:

1. Compare only information supported by
   the retrieved documents.

2. Do not invent information.

3. Clearly identify which document supports
   each point.

4. Identify similarities between the documents.

5. Identify differences between the documents.

6. Compare important technical aspects such as:
   - Manufacturing processes
   - Methods or operations
   - Materials
   - Equipment
   - Applications
   - Advantages
   - Limitations

7. Only discuss an aspect if the documents
   provide information about it.

8. If information is available in one document
   but not another, clearly say that the information
   was not found in the other document.

9. Use a clear structured format.

10. Cite information using the document names
    provided in the retrieved context.

Produce a concise but useful technical comparison.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    answer = response.text

    sources = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        sources.append(
            {
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "category": metadata.get(
                    "category",
                    "Unknown"
                ),
                "content": document
            }
        )

    return {
        "answer": answer,
        "sources": sources
    }
import os
import time
import chromadb
from dotenv import load_dotenv
from google import genai


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(
    api_key=api_key
)


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
# User question
# -----------------------------

question = "What are the different types of manufacturing processes?"


# -----------------------------
# Create embedding for question
# -----------------------------

result = gemini_client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = result.embeddings[0].values


# -----------------------------
# Retrieve relevant chunks
# -----------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=4,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)

# -----------------------------
# Check retrieval relevance
# -----------------------------

best_distance = results["distances"][0][0]

relevance_threshold = 0.8

if best_distance > relevance_threshold:

    print("\nQuestion:")
    print(question)

    print("\nNo sufficiently relevant information was found")
    print("in the provided industrial documents.")

    print(f"\nBest retrieval distance: {best_distance:.4f}")
    print(f"Relevance threshold: {relevance_threshold}")

    exit()

# -----------------------------
# Prepare retrieved context
# -----------------------------

context = ""

for i, document in enumerate(results["documents"][0]):

    metadata = results["metadatas"][0][i]

    source_number = i + 1

    context += f"""
[Source {source_number}]

Document: {metadata["source"]}
Page: {metadata["page"]}

Content:
{document}

--------------------
"""


# -----------------------------
# Create prompt for Gemini
# -----------------------------

prompt = f"""
You are an industrial research assistant.

Answer the user's question using ONLY the
information provided in the context.

Important instructions:

1. Do not use outside knowledge.
2. Use the provided sources as evidence.
3. Add a source citation such as [Source 1]
   after important factual statements.
4. Only cite a source if it actually supports
   the statement.
5. If the answer cannot be found in the provided
   context, say that the information is not
   available in the provided document.
6. Do not invent source numbers.

User Question:
{question}

Context:
{context}

Provide a clear and concise answer with citations.
"""

# -----------------------------
# Generate answer
# -----------------------------

max_attempts = 3

for attempt in range(max_attempts):

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        break

    except Exception as e:

        print(
            f"Gemini request failed "
            f"(attempt {attempt + 1}/{max_attempts})"
        )

        print(e)

        if attempt < max_attempts - 1:

            print("Waiting 5 seconds before retrying...")

            time.sleep(5)

        else:

            print("Gemini request failed after 3 attempts.")

            raise



# -----------------------------
# Prepare retrieved context
# -----------------------------

context = ""

for i, document in enumerate(results["documents"][0]):

    metadata = results["metadatas"][0][i]

    source_number = i + 1

    context += f"""
[Source {source_number}]

Document: {metadata["source"]}
Page: {metadata["page"]}

Content:
{document}

--------------------
"""


# -----------------------------
# Create prompt for Gemini
# -----------------------------

prompt = f"""
You are an industrial research assistant.

Answer the user's question using ONLY the
information provided in the context.

Important instructions:

1. Do not use outside knowledge.
2. Use the provided sources as evidence.
3. Add a source citation such as [Source 1]
   after important factual statements.
4. Only cite a source if it actually supports
   the statement.
5. If the answer cannot be found in the provided
   context, say that the information is not
   available in the provided document.
6. Do not invent source numbers.

User Question:
{question}

Context:
{context}

Provide a clear and concise answer with citations.
"""



# -----------------------------
# Display retrieval results
# -----------------------------

print("\nQuestion:")
print(question)

print("\nRetrieved Results:")

for i, document in enumerate(results["documents"][0]):

    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    relevance_score = 1 / (1 + distance)
    relevance_percentage = relevance_score * 100

    print(f"\n--- Result {i + 1} ---")
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata['page']}")
    print(f"Distance: {distance:.4f}")
    print(f"Relevance: {relevance_percentage:.2f}%")
    print("Content:")
    print(document)


# -----------------------------
# Display answer
# -----------------------------

print("\nQuestion:")
print(question)

print("\nGenerated Answer:")
print(response.text)


# -----------------------------
# Display sources
# -----------------------------

print("\nSources:")

for i, metadata in enumerate(
    results["metadatas"][0]
):

    source_number = i + 1

    distance = results["distances"][0][i]

    relevance_score = 1 / (1 + distance)

    relevance_percentage = relevance_score * 100

    print(
        f"[Source {source_number}] "
        f"{metadata['source']} "
        f"- Page {metadata['page']} "
        f"- Relevance: "
        f"{relevance_percentage:.2f}%"
    )
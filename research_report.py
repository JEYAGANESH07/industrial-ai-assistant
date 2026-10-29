import os

from dotenv import load_dotenv
from google import genai


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)


# -----------------------------
# Gemini client
# -----------------------------

gemini_client = genai.Client(
    api_key=api_key
)


# -----------------------------
# Generate research report
# -----------------------------

def generate_research_report(
    question,
    answer,
    results,
    confidence,
    confidence_label
):
    """
    Generate a structured industrial
    research report using retrieved
    document content.
    """

    context_parts = []

    for i, (chunk_id, item) in enumerate(
        results,
        start=1
    ):

        metadata = item["metadata"]

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

        content = item.get(
            "document",
            ""
        )

        context_parts.append(
            f"""
[Source {i}]

Document: {source}
Category: {category}
Page: {page}

Retrieved content:
{content}
"""
        )

    context = "\n".join(
        context_parts
    )


    # -----------------------------
    # Gemini prompt
    # -----------------------------

    prompt = f"""
You are an industrial research assistant.

Create a structured technical research report
based ONLY on the retrieved document content.

Research question:

{question}

Generated answer:

{answer}

Retrieval confidence:

{confidence * 100:.1f}% — {confidence_label}

Retrieved document content:

{context}


Create the report using this structure:

# Research Report

## 1. Research Question

State the research question.

## 2. Executive Summary

Provide a concise summary of the findings.

## 3. Detailed Findings

Explain the important information found
in the retrieved documents.

Use clear paragraphs and bullet points
where appropriate.

## 4. Technical Classification

If the documents contain classifications,
organize them clearly.

## 5. Key Observations

List the important technical observations.

## 6. Conclusion

Provide a concise conclusion based only
on the retrieved documents.

## 7. References

List every source used in the report.

For each source include:

- Source number
- Document name
- Category
- Page number

Important rules:

1. Use ONLY information from the retrieved
   document content.

2. Do not invent information.

3. Do not use outside knowledge.

4. Do not create references that are not
   present in the retrieved sources.

5. Cite important claims using [Source X].

6. Keep the report technically clear
   and suitable for an academic project.

7. If the retrieved documents do not provide
   enough information for a section, say so
   instead of inventing information.
"""


    # -----------------------------
    # Generate report
    # -----------------------------

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )


    return response.text
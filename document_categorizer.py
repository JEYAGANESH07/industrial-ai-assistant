import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(
    api_key=api_key
)


CATEGORIES = [
    "Manufacturing Process",
    "Production",
    "Safety",
    "Maintenance",
    "Quality Control",
    "Machinery / Equipment",
    "Materials",
    "Research Paper",
    "Technical Manual",
    "Other"
]


def categorize_document(text):
    """
    Categorize an industrial document
    based on its extracted text.
    """

    # Limit the amount of text sent to Gemini.
    sample_text = text[:12000]

    category_list = "\n".join(
        f"- {category}"
        for category in CATEGORIES
    )

    prompt = f"""
You are an industrial document classification system.

Classify the following document into exactly ONE
of the categories listed below.

Categories:

{category_list}

Document text:

{sample_text}

Instructions:

1. Select exactly one category.
2. Use only the categories provided above.
3. Return only the category name.
4. Do not provide an explanation.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    category = response.text.strip()

    if category not in CATEGORIES:
        category = "Other"

    return category
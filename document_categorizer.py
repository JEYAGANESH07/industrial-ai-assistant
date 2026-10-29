
import os
import re
import time
import random

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

gemini_client = (
    genai.Client(api_key=api_key)
    if api_key
    else None
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


def categorize_document(text, max_retries=3):
    """
    Categorize an industrial document using Gemini.

    Retries temporary API errors with exponential backoff.
    If categorization still fails, uses local keyword rules.
    """

    if not text or not text.strip():
        return "Other"

    # Avoid sending the entire PDF to Gemini.
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

    # If no API key is configured, categorize locally.
    if gemini_client is None:
        print("GEMINI_API_KEY not configured; using local categorization.")
        return _local_category(text)

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            result = (response.text or "").strip()

            # Match the model output to an allowed category.
            for category in CATEGORIES:
                if result.casefold() == category.casefold():
                    return category

            # Handle harmless formatting such as a bullet or quotes.
            cleaned = re.sub(
                r'^[\s\'"`*\-]+|[\s\'"`*]+$',
                "",
                result
            )

            for category in CATEGORIES:
                if cleaned.casefold() == category.casefold():
                    return category

            print(
                f"Unexpected category response: {result!r}. "
                "Using local categorization."
            )
            return _local_category(text)

        except Exception as exc:
            print(
                f"Gemini categorization attempt "
                f"{attempt + 1}/{max_retries} failed: {exc}"
            )

            if attempt < max_retries - 1:
                # Exponential backoff with a small random delay.
                delay = min(2 ** attempt, 8) + random.uniform(0, 0.5)
                time.sleep(delay)

    print("Gemini categorization failed after retries; using local rules.")
    return _local_category(text)


def _local_category(text):
    """
    Basic keyword-based categorization.
    No external API call is made.
    """

    sample = text[:12000].lower()

    rules = [
        (
            "Safety",
            ("safety", "hazard", "ppe", "accident", "risk assessment")
        ),
        (
            "Maintenance",
            ("maintenance", "lubrication", "preventive maintenance", "repair")
        ),
        (
            "Quality Control",
            ("quality control", "inspection", "defect", "tolerance",
             "quality assurance")
        ),
        (
            "Machinery / Equipment",
            ("machine tool", "machinery", "equipment", "lathe",
             "milling machine")
        ),
        (
            "Materials",
            ("material properties", "alloy", "polymer", "raw material",
             "composite")
        ),
        (
            "Production",
            ("mass production", "batch production", "production planning",
             "production line")
        ),
        (
            "Manufacturing Process",
            ("manufacturing process", "machining", "casting", "forming",
             "welding", "assembly")
        ),
        (
            "Research Paper",
            ("abstract", "keywords", "methodology", "references", "doi")
        ),
        (
            "Technical Manual",
            ("operating instructions", "user manual",
             "installation procedure", "troubleshooting")
        ),
    ]

    for category, keywords in rules:
        if any(keyword in sample for keyword in keywords):
            return category

    return "Other"

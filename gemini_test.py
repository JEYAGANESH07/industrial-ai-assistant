import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key loaded:", api_key is not None)

client = genai.Client(
    api_key=api_key,
    http_options={
        "timeout": 30000
    }
)

print("Sending test request...")

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain manufacturing in one short sentence."
)

print("\nGemini response:")
print(response.text)
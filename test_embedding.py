import os
from dotenv import load_dotenv
from google import genai

# Load the API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Small test sentence
text = "Industrial machines require regular maintenance."

# Create embedding
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)

# Get the embedding vector
embedding = result.embeddings[0].values

print("Embedding created successfully!")
print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])
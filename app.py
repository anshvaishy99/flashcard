from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN is missing from .env")
    exit()

# Hugging Face Inference Client
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

topic = input("Enter a topic: ")

prompt = f"""
Create 10 study flashcards about {topic}.

Use exactly this format:

1. Q: Question
   A: Answer

2. Q: Question
   A: Answer

Rules:
- Questions should be short.
- Answers should be simple.
- Cover important concepts.
- Make them useful for students.
- Do not add anything outside the flashcards.
"""

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1000,
        temperature=0.7
    )

    print("\n" + "=" * 60)
    print("FLASHCARDS")
    print("=" * 60)
    print()

    print(response.choices[0].message.content)

except Exception as e:
    print("\nERROR:")
    print(e)
    
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_sentence(signs):
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": "Convert these sign language words into one short, natural English sentence. Reply with the sentence only."},
            {"role": "user", "content": " ".join(signs)},
        ],
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()
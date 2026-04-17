import os
import certifi
import httpx
from openai import OpenAI

http_client = httpx.Client(verify=certifi.where())

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)

def generate_response(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an IC compensation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()
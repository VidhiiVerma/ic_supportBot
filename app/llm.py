import os
import certifi
import httpx
from openai import AzureOpenAI

# secure HTTP client
http_client = httpx.Client(verify=certifi.where())

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version = "2024-12-01-preview",
    http_client=http_client
)

def generate_response(prompt: str):
    try:
        response = client.chat.completions.create(
            model="gpt-5-chat",  
            messages=[
                {"role": "system", "content": "You are an IC compensation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM ERROR] {str(e)}")
        return "LLM failed to generate response."
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
        prompt_lower = prompt.lower().strip()

        # ✅ handle only thank you
        if "thank" in prompt_lower:
            return "You're welcome! See you 😊"

        response = client.chat.completions.create(
            model="gpt-5-chat",  
            messages=[
                {"role": "system", "content": "You are an Incentive compensation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[LLM ERROR] {str(e)}")
        return "LLM failed to generate response."
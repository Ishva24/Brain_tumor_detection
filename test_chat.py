
import requests

api_key = "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"
base_url = "https://openrouter.ai/api/v1/chat/completions"
model = "google/gemini-2.0-flash-exp:free"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "Brain Tumor Classifier",
}

payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Hello"}
    ]
}

try:
    print("Sending request...")
    response = requests.post(base_url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    response.raise_for_status()
except Exception as e:
    print(f"Error: {e}")

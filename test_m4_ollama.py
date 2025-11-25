import requests
import os
from dotenv import load_dotenv
load_dotenv()

base_url = os.getenv("LLM_BASE_URL")
model_name = os.getenv("LLM_MODEL_NAME")

def generate_response(prompt, model=model_name):
    url = os.getenv("LLM_BASE_URL")
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json()["response"]

# Code completion prompt
prompt = "def calculate_sum(numbers):\n   "

print("Generating...")
result = generate_response(prompt)
print(result)
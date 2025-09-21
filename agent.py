import requests
from pydantic import BaseModel
import os

HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_API_KEY")  # replace if needed
HF_MODEL = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")

class HFRequestsLLM(BaseModel):
    model: str = HF_MODEL
    token: str = HF_TOKEN
    api_url: str = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    def run(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"inputs": prompt}

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Handle both possible HF response formats
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            else:
                return str(data)
        except Exception as e:
            return f"Error calling Hugging Face API: {e}"

llm = HFRequestsLLM()

def agent(question: str) -> str:
    prompt = f"Answer the following question clearly and concisely:\n\n{question}"
    return llm.run(prompt)

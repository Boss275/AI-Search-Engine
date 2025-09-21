import os
from huggingface_hub import InferenceClient
import wikipedia
import arxiv

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

client = InferenceClient(token=HF_TOKEN)

def ask_model(prompt: str) -> str:
    """
    Send prompt to Hugging Face model and get response.
    """
    response = client.text_generation(HF_MODEL, prompt, max_new_tokens=500)
    return response[0]["generated_text"]

def wiki_search(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Wikipedia error: {e}"

def arxiv_search(query: str) -> str:
    try:
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = [f"{r.title} ({r.published.date()})\n{r.entry_id}" for r in search.results()]
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"arXiv error: {e}"

def agent_run(prompt: str) -> str:
    try:
        return ask_model(prompt)
    except Exception as e:
        wiki = wiki_search(prompt)
        arx = arxiv_search(prompt)
        return f"Model error: {e}\n\nWikipedia:\n{wiki}\n\narXiv:\n{arx}"

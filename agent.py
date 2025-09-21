import os
import requests
import arxiv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.llms.base import LLM

HF_MODEL = os.environ.get("HF_MODEL", "tiiuae/falcon-7b-instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

class HFRequestsLLM(LLM):
    def __init__(self, model: str = None, token: str = None, max_new_tokens: int = 512, temperature: float = 0.1, timeout: int = 60):
        self.model = model or HF_MODEL
        self.token = token or HF_TOKEN
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.timeout = timeout

    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
        }
        try:
            r = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
        except Exception as e:
            return f"Error: request failed - {e}"
        if r.status_code != 200:
            try:
                return f"Error: HF {r.status_code} - {r.json()}"
            except Exception:
                return f"Error: HF {r.status_code} - {r.text}"
        data = r.json()
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for k in ("generated_text", "output_text", "text", "result", "answer"):
                    if k in first:
                        return str(first[k]).strip()
            if isinstance(first, str):
                return first.strip()
        if isinstance(data, dict):
            for k in ("generated_text", "output_text", "text", "result", "answer"):
                if k in data:
                    return str(data[k]).strip()
            if "error" in data:
                return f"Error: {data['error']}"
        return str(data)

    @property
    def _identifying_params(self):
        return {"model": self.model, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature}

    @property
    def _llm_type(self):
        return "hf_requests_llm"

llm = HFRequestsLLM()

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def arxiv_search(query: str) -> str:
    try:
        search = arxiv.Search(query=query, max_results=3)
        results = list(search.results())
        if not results:
            return "No results found."
        return "\n\n".join(f"Title: {r.title}\nSummary: {r.summary[:500]}..." for r in results)
    except Exception as e:
        return f"ArXiv error: {e}"

arxiv_tool = Tool(
    name="ArxivSearch",
    func=arxiv_search,
    description="Search arXiv for academic and scientific research."
)

tools = [
    Tool(name="Wikipedia", func=wiki.run, description="Search Wikipedia for general knowledge."),
    arxiv_tool
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

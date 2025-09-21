import os
import arxiv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEndpoint
from langchain.llms.base import LLM

HF_TOKEN = os.environ.get("HF_TOKEN")

class HuggingFaceLLM(LLM):
    def __init__(self, repo_id: str, hf_token: str):
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            huggingfacehub_api_token=hf_token,
            temperature=0.7,
            max_new_tokens=512
        )

    def _call(self, prompt: str, stop=None):
        res = self.llm(prompt)
        if isinstance(res, str):
            return res.strip()
        if isinstance(res, list) and len(res) > 0 and "generated_text" in res[0]:
            return res[0]["generated_text"].strip()
        return str(res)

    @property
    def _llm_type(self):
        return "huggingface_custom"

llm = HuggingFaceLLM("tiiuae/falcon-7b-instruct", HF_TOKEN)

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def arxiv_search(query: str) -> str:
    search = arxiv.Search(query=query, max_results=2)
    results = list(search.results())
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r.title}\nSummary: {r.summary[:300]}..."
        for r in results
    )

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

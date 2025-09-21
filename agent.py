import os
import arxiv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.llms.base import LLM
from langchain_huggingface import HuggingFaceEndpoint

HF_TOKEN = os.environ.get("HF_TOKEN")

class HuggingFaceSafeLLM(LLM):
    def __init__(self, repo_id: str, hf_token: str):
        self.endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text2text-generation",
            huggingfacehub_api_token=hf_token,
            temperature=0.7,
            max_new_tokens=256
        )

    def _call(self, prompt: str, stop=None):
        try:
            res = self.endpoint(prompt)
            if isinstance(res, str):
                return res.strip()
            if isinstance(res, list) and len(res) > 0:
                if "generated_text" in res[0]:
                    return res[0]["generated_text"].strip()
                if "output_text" in res[0]:
                    return res[0]["output_text"].strip()
            return str(res)
        except Exception as e:
            return f"HF Error: {e}"

    @property
    def _llm_type(self):
        return "huggingface_safe"

llm = HuggingFaceSafeLLM("google/flan-t5-base", HF_TOKEN)

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

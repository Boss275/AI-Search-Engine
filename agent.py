import os
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEndpoint
import arxiv

HF_TOKEN = os.environ.get("HF_TOKEN")

# Use a model that supports "text-generation"
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.7,
    max_new_tokens=512
)

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def arxiv_search(query: str) -> str:
    search = arxiv.Search(query=query, max_results=3)
    results = list(search.results())
    output = ""
    for r in results:
        output += f"Title: {r.title}\nSummary: {r.summary[:500]}...\n\n"
    return output.strip()

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
    agent="zero-shot-react-description",
    verbose=True
)

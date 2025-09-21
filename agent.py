import os
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.llms import HuggingFaceEndpoint
import arxiv
import streamlit as st

# Set HuggingFace token from Streamlit secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]

# Initialize HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    max_new_tokens=512,
    temperature=0.1
)

# Wikipedia tool
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Arxiv search tool
def arxiv_search(query: str) -> str:
    search = arxiv.Search(query=query, max_results=3)
    results = list(search.results())
    if not results:
        return "No results found."
    output = ""
    for result in results:
        output += f"Title: {result.title}\nSummary: {result.summary[:500]}...\n\n"
    return output.strip()

arxiv_tool = Tool(
    name="ArxivSearch",
    func=arxiv_search,
    description="Search arXiv for academic and scientific research."
)

# List of tools
tools = [
    Tool(name="Wikipedia", func=wiki.run, description="Search Wikipedia for general knowledge."),
    arxiv_tool
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

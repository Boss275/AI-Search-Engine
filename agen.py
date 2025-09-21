from langchain.agents import initialize_agent, Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
import arxiv
import os

# Load API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Wikipedia tool
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# arXiv tool
def arxiv_search(query: str) -> str:
    search = arxiv.Search(query=query, max_results=3)
    results = search.results()
    output = ""
    for result in results:
        output += f"Title: {result.title}\nSummary: {result.summary[:500]}...\n\n"
    return output.strip()

arxiv_tool = Tool(
    name="ArxivSearch",
    func=arxiv_search,
    description="Use this tool for scientific research queries using arXiv"
)

tools = [
    Tool(name="Wikipedia", func=wiki.run, description="Use this tool for general knowledge"),
    arxiv_tool
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

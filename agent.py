from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
import arxiv

openai_api_key = "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"

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

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

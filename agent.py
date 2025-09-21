from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
import arxiv

# ✅ Don't hardcode API key here — load from env instead
# (set OPENAI_API_KEY in your terminal or Streamlit Cloud settings)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"  # You can swap for "gpt-4o-mini" if you want cheaper/faster
)

# Wikipedia tool
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Arxiv tool
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

# Combine tools
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

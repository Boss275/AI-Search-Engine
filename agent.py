import os
from langchain import HuggingFaceHub
from langchain.agents
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")

llm = HuggingFaceHub(
    repo_id=HF_MODEL,
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

tools = load_tools(["wikipedia", "arxiv"], llm=llm)

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

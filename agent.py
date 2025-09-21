import os
from langchain import HuggingFaceHub
from langchain.agents
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "tiiuae/falcon-7b-instruct")

from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    model_kwargs={"temperature": 0, "max_new_tokens": 500}
)

tools = load_tools(["wikipedia", "arxiv"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

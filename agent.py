import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.agents import initialize_agent, load_tools  # fixed import

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

llm = HuggingFaceHub(
    repo_id=HF_MODEL,
    model_kwargs={"temperature": 0, "max_new_tokens": 500},
    huggingfacehub_api_token=HF_TOKEN
)

tools = load_tools(["wikipedia", "arxiv"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

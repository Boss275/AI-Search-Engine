import os
from langchain import HuggingFaceHub
from langchain.agents import initialize_agent, load_tools

# Get your token from Streamlit secrets
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # public and accessible

# LLM setup
llm = HuggingFaceHub(
    repo_id=HF_MODEL,
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0, "max_new_tokens": 500}
)

# Tools
tools = load_tools(["wikipedia", "arxiv"], llm=llm)

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

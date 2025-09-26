import os
from langchain.agents import initialize_agent, load_tools
from langchain.llms import HuggingFaceHub

# ─────────────────────────────────────────────
# Load Hugging Face API token from environment
# ─────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")

# ─────────────────────────────────────────────
# Initialize the LLM with a public model
# ─────────────────────────────────────────────

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    model_kwargs={"temperature": 0, "max_new_tokens": 500},
    huggingfacehub_api_token=HF_TOKEN
)

# ─────────────────────────────────────────────
# Load tools for the agent
# ─────────────────────────────────────────────

tools = load_tools(["wikipedia", "arxiv"], llm=llm)

# ─────────────────────────────────────────────
# Initialize the agent
# ─────────────────────────────────────────────

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

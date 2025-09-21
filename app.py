import os
import streamlit as st

st.set_page_config(page_title="AI-Powered Search Assistant")

if "HF_TOKEN" not in st.secrets:
    st.error("HF_TOKEN missing. Add it in Manage app → Settings → Secrets and redeploy.")
    st.stop()

HF_TOKEN = st.secrets["HF_TOKEN"]
HF_MODEL = st.secrets.get("HF_MODEL", "tiiuae/falcon-7b-instruct")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_MODEL"] = HF_MODEL
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

from agent import agent

st.title("AI-Powered Search Assistant")
st.write("Enter a question below — the assistant will use Wikipedia, arXiv, and Hugging Face AI to answer.")

query = st.text_input("Your question:")

def ask_agent(prompt: str) -> str:
    try:
        return agent.run(prompt)
    except Exception as e:
        return f"Error: {e}"

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        response = ask_agent(query)
        st.markdown("### Answer")
        st.write(response)

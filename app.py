import streamlit as st
from openai import OpenAI
from agent import agent

OPENAI_API_KEY = "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"
PROJECT_ID = "proj_QHsK4ZJE2mnu6bkcuEGYAGyI"
ORG_ID = "org-V8sG6TuRjJhPJa6MTlZHURN0"
client = OpenAI(api_key=OPENAI_API_KEY, project=PROJECT_ID, organization=ORG_ID)

def ask_openai(prompt):
    try:
        response = agent.run(prompt)
        return response
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(page_title="AI-Powered Search Assistant")

st.title("AI-Powered Search Assistant")
st.write("Enter a question below — the assistant will use Wikipedia, arXiv, and OpenAI to answer.")

query = st.text_input("Your question:")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        response = ask_openai(query)
        st.markdown("### Answer")
        st.write(response)

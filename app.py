import streamlit as st
from openai import OpenAI
from agent import agent

OPENAI_API_KEY = "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"
PROJECT_ID = "org-NDV0ZribUDQj8ChhdJfsKCWa"
ORG_ID = "proj_sojtXxIXoP4WhEZDzMiMOWVw"
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

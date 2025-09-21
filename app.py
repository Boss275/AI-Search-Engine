import streamlit as st
from openai import OpenAI
from agent import agent

client = OpenAI()

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

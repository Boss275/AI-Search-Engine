import streamlit as st
from agent import agent_run

st.title("AI Search Engine (Hugging Face)")

prompt = st.text_input("Enter your question:")

if prompt:
    try:
        response = agent_run(prompt)
        st.write(response)
    except Exception as e:
        st.error(f"Error running agent: {e}")

import streamlit as st
from agent import agent

st.title("AI-Powered Search Assistant")

prompt = st.text_input("Enter your question below — the assistant will use Wikipedia, arXiv, and Hugging Face AI to answer.")

if prompt:
    try:
        response = agent.run(prompt)
        st.write("**Answer:**", response)
    except Exception as e:
        st.error(f"Error: {e}")

import streamlit as st
from agent import agent

st.title("AI Search Engine (Hugging Face)")

prompt = st.text_input("Enter your question:")

if prompt:
    with st.spinner("Thinking..."):
        response = agent.run(prompt)
    st.success("Answer:")
    st.write(response)

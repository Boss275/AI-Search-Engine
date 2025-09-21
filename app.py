import streamlit as st
from agent import agent

st.title("AI Search Engine (Hugging Face)")

prompt = st.text_input("Enter your question:")

if prompt:
    try:
        response = agent.run(prompt)
        st.write("Answer")
        st.write(response)
    except Exception as e:
        st.error(f"Model error: {str(e)}")

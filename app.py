import streamlit as st
from agent import agent

st.set_page_config(page_title="AI-Powered Search Assistant", layout="centered")

st.title("AI-Powered Search Assistant")
st.write("Enter a question below — the assistant will use Hugging Face AI to answer.")

question = st.text_input("Your question:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer = agent(question)
        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question before asking.")

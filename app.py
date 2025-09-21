import streamlit as st
from agent import agent

def ask_model(prompt):
    try:
        return agent.run(prompt)
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(page_title="AI-Powered Search Assistant")
st.title("AI-Powered Search Assistant")
st.write("Enter a question below — the assistant will use Wikipedia, arXiv, and Hugging Face AI to answer.")

query = st.text_input("Your question:")
if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        response = ask_model(query)
        st.markdown("### Answer")
        st.write(response)

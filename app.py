import streamlit as st
from agent import agent

st.set_page_config(page_title="🔎 AI Agent Search Engine")

st.title("🔎 AI Agent Search Engine")
st.markdown("Ask a question and let the AI use Wikipedia and arXiv tools to answer!")

query = st.text_input("Enter your query:", placeholder="e.g. What is quantum computing?")

if st.button("Search") and query:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(query)
            st.success("Done!")
            st.markdown("### 🧠 Answer")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

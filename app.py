import streamlit as st
from agent import agent

st.set_page_config(page_title="AI Agent Search Engine")

st.title("AI Agent Search Engine")
st.markdown("Ask a question and let the AI search using Wikipedia and arXiv.")

query = st.text_input("Enter your query:", placeholder="e.g. What is quantum computing?")

if st.button("Search") and query:
    with st.spinner("Searching..."):
        try:
            response = agent.run(query)
            st.success("Search complete.")
            st.markdown("### Answer")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

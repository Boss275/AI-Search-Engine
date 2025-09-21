import streamlit as st
from agent import agent

def ask_agent(prompt):
    try:
        return agent.run(prompt)
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(page_title="AI-Powered Search Assistant")

st.title("AI-Powered Search Assistant")
st.write("Enter a question below, and an AI powered search engine will find you the answer.")

query = st.text_input("Your question:")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        response = ask_agent(query)
        st.markdown("### Answer")
        st.write(response)

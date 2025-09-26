import streamlit as st
from agent import agent

# ─────────────────────────────────────────────
# Streamlit app title and input
# ─────────────────────────────────────────────

st.title("🤖 AI Search Engine (Hugging Face)")

prompt = st.text_input("💬 Enter your question:")

# ─────────────────────────────────────────────
# Run agent on prompt and display results
# ─────────────────────────────────────────────

if prompt:
    try:
        response = agent.run(prompt)
        st.subheader("🧠 Answer:")
        st.write(response)
    except Exception as e:
        st.error(f"⚠️ Model error: {str(e)}")

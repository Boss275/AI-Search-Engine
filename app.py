import streamlit as st
import openai

openai.api_key = "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"
openai.organization = "org-V8sG6TuRjJhPJa6MTlZHURN0"

def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(page_title="AI-Powered Search Assistant")

st.title("AI-Powered Search Assistant")
st.write("Enter a question below — the assistant will generate a clear and informative response based on your input.")

query = st.text_input("Your question:")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        response = ask_openai(query)
        st.markdown("### Answer")
        st.write(response)

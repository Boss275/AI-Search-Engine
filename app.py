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

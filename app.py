import streamlit as st
import openai

openai.api_key = "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"
openai.organization = "org-V8sG6TuRjJhPJa6MTlZHURN0"

def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",e
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

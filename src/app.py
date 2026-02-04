import streamlit as st
from pipeline import RAGPipeline

pipeline = RAGPipeline()

st.title("RAG Pipeline")

user_question = st.text_input("Ask a question:")

if st.button("Submit"):
    answer, docs = pipeline.ask(user_question)
    st.write("Answer:", answer)
    st.write("Relevant Documents:")
    for doc in docs:
        st.write(doc.page_content)
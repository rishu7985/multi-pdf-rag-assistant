import streamlit as st
from rag import ask_rag

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.write("Ask questions about your indexed PDF documents.")

query = st.text_input(
    "Enter your question:",
    placeholder="Example: What is a bipartite graph?"
)

if st.button("Search"):

    if query.strip():

        with st.spinner("Searching documents..."):

            result = ask_rag(query)

        st.success("Answer generated successfully!")

        st.subheader("Answer")
        st.write(result["answer"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sources")
            for source in result["sources"]:
                st.write(f"📄 {source}")

        with col2:
            st.subheader("Pages")
            st.write(result["pages"])
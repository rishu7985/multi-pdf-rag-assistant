import streamlit as st
from rag import ask_rag
import fitz
from PIL import Image
import io
import os
import time


from index_builder import(
load_pdfs,chunk_documents,generate_embeddings,
build_faiss_index,save_index
)

def get_page_image(pdf_path,page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    doc.close()
    return image

if "selected_page" not in st.session_state:
    st.session_state.selected_page = []

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")

if "delete_message" in st.session_state:
    st.success(st.session_state.delete_message)
    del st.session_state.delete_message

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type = "pdf",
    accept_multiple_files = True
)

os.makedirs("data",exist_ok=True)

def rebuild_index():
    docs = load_pdfs("data")
    if len(docs) == 0:
        if os.path.exists("faiss_index.bin"):
            os.remove("faiss_index.bin")
        if os.path.exists("chunks.pkl"):
            os.remove("chunks.pkl")
        return False

    chunks = chunk_documents(docs)
    chunks = generate_embeddings(chunks)
    index = build_faiss_index(chunks)
    save_index(index,chunks)

    return True

new_upload = False
if uploaded_files:
    for pdf in uploaded_files:
        pdf_path = os.path.join("data",pdf.name)
        if not os.path.exists(pdf_path):
            with open(pdf_path,
                      "wb") as f:
                f.write(pdf.getbuffer())
            new_upload = True

if new_upload:
    with st.spinner("Building index....."):
        if rebuild_index():
            st.success("Index updated successfully!")
        else :
            st.warning("No PDF documnets found.")

index_exists = (
    os.path.exists("faiss_index.bin") and os.path.exists("chunks.pkl")
)
if st.button ("Build Index"):
    if rebuild_index():
        from rag import load_index
        load_index()
        st.success("Index updated successfully!")
    else:
        st.warning("No PDF documents found.")

st.sidebar.header("Settings")
top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value = 1,
    max_value = 10,
    value = 5,
    step = 1
)

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

for chat in st.session_state.messages:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

        st.write("--sources--")
        for source in chat["sources"]:
            st.write(f"📄 {source}")
        st.write("**Pages:**")
        st.write(chat["pages"])
    with st.expander("📄 Retrieved Context"):
        for i,chunk in enumerate(chat["retrieved_chunks"]):
            st.markdown(f"### Chunk {i}")
            st.write(f"**Source:** {chunk['source']}")
            st.write(f"**Page:** {chunk['page']}")

            pdf_path = os.path.join("data",chunk["source"])
            if os.path.exists(pdf_path):
                image = get_page_image(pdf_path,chunk["page"])
                st.image(image,
                         caption=f"{chunk['source']}-Page{chunk['page']+1}",
                         use_container_width = True
                         )

            st.write(chunk["text"])
            similarity = max(0.0,min(1.0,chunk["score"]))
            st.write(f"**Similarity:** {similarity*100:.2f}%")
            st.progress(similarity)
            st.divider()

    st.divider()

chat_history = ""

for chat in st.session_state.messages:
    chat_history += f"""
    Question:
    {chat['question']}
    
    Answer:
    {chat['answer']}
    
    Sources:
    {",".join(chat['sources'])}
    
    Pages:
    {chat['pages']}
    
    --------------------------
    """
st.write("Ask questions about your indexed PDF documents.")

query = st.text_input(
    "Enter your question:",
    placeholder="Example: What is a bipartite graph?",
    disabled = not index_exists
)

st.sidebar.download_button(
    label = "📥 Download Chat",
    data = chat_history,
    file_name = "chat_history.txt",
    mime = 'text/plain'
)

st.sidebar.divider()
st.sidebar.subheader("Statistics")
st.sidebar.write(f"Questions Asked: {len(st.session_state.messages)}")
st.sidebar.write(f"Embedding Model: all-MiniLM-L6-v2")
st.sidebar.write(f"LLM: Gemini 2.5 Flash")


st.sidebar.divider()
st.sidebar.subheader("Uploaded Pdfs")
pdf_files = [pdf for pdf in os.listdir("data") if pdf.endswith(".pdf")]
for pdf in pdf_files:
    col1, col2 = st.sidebar.columns([4,1])
    with col1:
        st.write(f"📄 {pdf}")
    with col2:
        if st.button("🗑", key=pdf):
            os.remove(os.path.join("data",pdf))

            with st.spinner("Updating index..."):
                if rebuild_index():
                    st.success("Index updated successfully!")
                else:
                    st.warning("No PDF documents found.")

            st.session_state.delete_message = f"{pdf} deleted successfully,\nPlease rebuild the index"
            st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Search In")

selected_pdfs = st.sidebar.multiselect(
    "Choose PDFs",
    options = pdf_files,
    default = pdf_files
)


if st.button("Search", disabled=not index_exists):

    if query.strip():

        with st.spinner("Searching documents..."):

            placeholder = st.empty()
            answer = ""

            st.session_state.selected_page = None

            stream = ask_rag(
                query,
                st.session_state.messages,
                top_k,
                selected_pdfs,
                streaming=True
            )

            for chunk in stream:
                words = chunk.split(" ")

                for w in words:
                    answer += w+" "
                    placeholder.markdown(answer + "▌")
                    time.sleep(0.01)

            # ---- metadata call (for citations/pages) ----
            meta = ask_rag(
                query,
                st.session_state.messages,
                top_k,
                selected_pdfs,
                streaming=False
            )

            st.session_state.messages.append({
                "question": query,
                "answer": answer,
                "sources": meta["sources"],
                "pages": meta["pages"],
                "retrieved_chunks": meta["retrieved_chunks"]
            })

            st.sidebar.write("Search query used")
            st.sidebar.code(meta["rewritten_query"])

        st.success("Answer generated successfully!")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources used")
        for c in meta["citations"]:
            if st.button(c):
                parts = c.split("(Page")
                file = parts[0].strip()
                page = int(parts[1].replace(")","").strip())

                st.session_state.selected_page = {
                    "file":file,
                    "page":page
                }

        if st.session_state.selected_page:
            st.subheader("📄 Citation Preview")

            file = st.session_state.selected_page["file"]
            page = st.session_state.selected_page["page"]

            pdf_path = os.path.join("data",file)

            if os.path.exists(pdf_path):
                image = get_page_image(pdf_path,page)

                st.image(
                    image,
                    caption = f"{file}-Page{page+1}",
                    use_container_width = True
                )

import streamlit as st
import pdfplumber
from document_loader import chunk_text
from vector_store import VectorStore
from rag_pipeline import generate_answer, summarize_text

# Page setup
st.set_page_config(page_title="RAG Document QA", layout="wide")

# 💄 Custom CSS for centering and hiding default Streamlit UI elements
st.markdown("""
    <style>
    body {
        background-color: #70615D;
    }
    h1 {
        text-align: center;
        color: #333333;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 🧠 Centered title
st.markdown("<h1>📄 Retrieval-Augmented Generation on Documents</h1>", unsafe_allow_html=True)

# 📥 Upload PDFs
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    all_metadata = []

    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            file_text = "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )

        st.markdown(f"### 📘 File: {file.name}")

        # 📝 Summarize button
        if st.button(f"📝 Summarize {file.name}"):
            with st.spinner(f"Summarizing {file.name}..."):
                summary = summarize_text(file_text)
                st.subheader("📄 Summary:")
                st.write(summary)
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"{file.name}_summary.txt",
                    mime="text/plain"
                )

        # ➕ Chunk and store
        chunks = chunk_text(file_text)
        all_chunks.extend(chunks)
        all_metadata.extend([{"source": file.name}] * len(chunks))

    # 🔍 RAG Vector Store
    vs = VectorStore()
    vs.add_documents(all_chunks, all_metadata)

    st.markdown("---")

    # 🧠 Ask a question
    query = st.text_input("💬 Ask a question about your uploaded documents:")
    if query:
        results = vs.query(query)
        contexts = results["documents"][0]
        sources = results["metadatas"][0]
        answer = generate_answer(query, contexts)

        st.subheader("🧠 Answer:")
        st.write(answer)

        st.subheader("🔍 Sources:")
        for source in sources:
            st.markdown(f"- **{source['source']}**")

        with st.expander("📂 Show context chunks"):
            for chunk in contexts:
                st.code(chunk)
else:
    st.info("📂 Please upload at least one PDF.")

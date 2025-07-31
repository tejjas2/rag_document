import os
import pdfplumber
from nltk.tokenize import sent_tokenize

# 📌 One-time download should not be in production code
# Run this ONCE in a separate setup script or Python shell:
# import nltk; nltk.download('punkt')

# ✅ Extract text from a single PDF file
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )
    return text

# ✅ Load and extract text from all PDFs in a folder
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            documents.append({"text": text, "source": filename})
    return documents

# ✅ Chunk long text into smaller parts (for RAG context)
def chunk_text(text, max_tokens=250):
    sentences = sent_tokenize(text)
    chunks, chunk = [], []
    tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, tokens = [], 0
        chunk.append(sentence)
        tokens += sentence_tokens

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

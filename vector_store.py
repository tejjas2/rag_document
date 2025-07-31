import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, collection_name="docs"):
        self.client = chromadb.Client()  # ✅ NEW style
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, docs, metadata=None):
        for i, doc in enumerate(docs):
            embedding = self.embedder.encode(doc, convert_to_tensor=False).tolist()
            self.collection.add(
                documents=[doc],
                embeddings=[embedding],
                ids=[f"id_{i}"],
                metadatas=[metadata[i]]
            )

    def query(self, query_text, top_k=3):
        query_emb = self.embedder.encode(query_text, convert_to_tensor=False).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        return results

from utils.document_loader import load_document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_index_from_file(file_path="data/sample_docs.txt"):
    docs = load_document(file_path)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("index")
    print("✅ Index created and saved as 'index'")

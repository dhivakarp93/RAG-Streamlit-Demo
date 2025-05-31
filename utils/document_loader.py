from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return text_splitter.split_documents(docs)

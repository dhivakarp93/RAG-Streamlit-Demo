from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

def get_fusion_rag_response(query):
    # Load semantic retriever
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    semantic_retriever = vectorstore.as_retriever()

    # Load keyword retriever
    with open("data/sample_docs.txt") as f:
        docs = f.readlines()
    bm25_retriever = BM25Retriever.from_texts(docs)
    bm25_retriever.k = 2

    # Hybrid retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5]
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-4o-mini"),
        chain_type="map_reduce",
        retriever=ensemble_retriever
    )

    result = qa_chain.invoke({"query": query})
    return result['result']

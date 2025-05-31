from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_simple_rag_response(query):
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-4o-mini"),
        chain_type="stuff",
        retriever=retriever
    )
    result = qa_chain.invoke({"query": query})
    return result['result']

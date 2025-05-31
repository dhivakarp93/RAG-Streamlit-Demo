from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

decision_prompt = PromptTemplate.from_template("""
You are an AI agent evaluating whether to use the retrieved context to answer or request more info.
Question: {question}
Retrieved Context: {context}

Reason step-by-step, then respond with either:
- 'USE_CONTEXT'
- 'NEED_MORE_INFO'

Final Decision:
""")

def get_self_rag_response(query):
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])

    llm = OpenAI(model="gpt-4o-mini")
    decision_chain = LLMChain(llm=llm, prompt=decision_prompt)
    decision = decision_chain.run(question=query, context=context)

    if "USE_CONTEXT" in decision:
        return llm(f"Answer this: {query}\n\nContext: {context}")
    else:
        return "I need more information to provide a confident answer."

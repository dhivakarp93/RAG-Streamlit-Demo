from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

correction_prompt = PromptTemplate.from_template("""
You are an AI assistant that evaluates whether retrieved information is sufficient to answer a question.
Question: {question}
Context: {context}

Is this context sufficient to answer the question? Respond only YES or NO.
""")

def get_corrective_rag_response(query):
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])

    llm = OpenAI(model="gpt-4o-mini")
    eval_chain = LLMChain(llm=llm, prompt=correction_prompt)
    decision = eval_chain.run(question=query, context=context).strip()

    if decision == "YES":
        answer = llm(f"Answer this: {query}\n\nContext: {context}")
        return answer
    else:
        return "Insufficient context to answer the question."

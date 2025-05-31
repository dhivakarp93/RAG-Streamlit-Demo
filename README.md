A **Streamlit-based demo application** that lets you explore and compare **4 Retrieval-Augmented Generation (RAG)** techniques using your own documents.

### 🔍 Supported RAG Models:
- **Simple RAG**
- **Corrective RAG**
- **Self-RAG**
- **Fusion RAG**

Built with:
- 🔗 **LangChain**
- 🤖 **OpenAI (GPT-4o-mini)**
- 🔍 **FAISS (for semantic search)**
- 📄 **BM25 (for keyword search)**
- 📁 Supports document uploads (`.txt`, `.pdf`)
- 🖥️ **Streamlit UI**

---

## 🚀 Features

- Upload your own `.txt` or `.pdf` files
- Build FAISS index locally for fast retrieval
- Interactive UI to switch between 4 RAG models
- Easily extendable to support more RAG variants

---

## 🧠 RAG Model Comparison

| Model | Description |
|-------|-------------|
| **Simple RAG** | Basic retrieval + generation pipeline |
| **Corrective RAG** | Evaluates context before answering |
| **Self-RAG** | Uses LLM to decide if more info is needed |
| **Fusion RAG** | Combines semantic and keyword-based retrieval |

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries (`requirements.txt`):
```txt
streamlit
langchain
langchain-openai
langchain-community
openai
faiss-cpu
sentence-transformers
rank-bm25
python-dotenv
pypdf
```

---

## 🛠️ Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/dhivakarp93/RAG-Streamlit-Demo.git
   cd RAG-Streamlit-Demo
   ```

2. Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Load initial documents into FAISS index:
   ```bash
   python utils/vectorstore_utils.py
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Sample Documents

We include a small `sample_docs.txt` for testing. You can replace it with any knowledge base of your choice.

---

## 💡 Future Ideas

- Add chat history/memory
- Evaluate answer quality with metrics
- Support local LLMs via Ollama or HuggingFace
- Export answers to PDF or Markdown

---

## 🤝 Contributions

Contributions are welcome! Feel free to submit issues or PRs for:
- New RAG models
- UI enhancements
- Bug fixes
- Documentation improvements

---

## 📄 License

MIT License – feel free to use and modify.
```

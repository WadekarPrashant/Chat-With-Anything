# 💬 Chat With Anything

> Interact with your data — CSVs, PDFs, web pages, databases, and the live web — using plain English, powered by LangChain and OpenAI.

<div align="center">

![Live Demo](https://img.shields.io/badge/Live%20Demo-Coming%20Soon-orange?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0%2B-FF6F00?style=flat-square&logo=databricks&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4.0%2B-47A248?style=flat-square&logo=mongodb&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-3F4F75?style=flat-square&logo=plotly&logoColor=white)

</div>

---

## ✨ Features

**Chat With Anything** is a multi-modal Streamlit application with six purpose-built tabs, each letting you query a different data source through natural language.

| Tab | Description |
|-----|-------------|
| 📊 **CSV Analysis** | Upload any CSV and ask questions in plain English. The LLM generates pandas queries, executes them safely, and explains the results. |
| 📄 **Document Q&A** | Upload a PDF or paste a URL. Documents are chunked, embedded, and stored in ChromaDB for retrieval-augmented Q&A. |
| 🍃 **MongoDB Chat** | Connect to any MongoDB instance, explore the collection schema, and query it with natural language. |
| 🏆 **Model Benchmark** | Interactive Plotly charts comparing leading LLMs on accuracy and parameter count — from the MIT-WPU research dataset. |
| 🗄️ **SQL Database** | Connect via any SQLAlchemy connection string (SQLite, PostgreSQL, MySQL…), ask a question, and get the generated SQL plus a plain-English answer. |
| 🤖 **AI Agent** | A ReAct-style LangChain agent with DuckDuckGo search. It decides autonomously whether to search the web or answer from knowledge, and shows its full reasoning chain. |

---

## 🏗️ System Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│   CSV │ PDF/URL │ MongoDB │ Benchmark │ SQL │ AI Agent  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
             ┌──────────────────┐
             │  LangChain Core  │
             │  (Orchestration) │
             └────────┬─────────┘
          ┌───────────┼────────────┐
          ▼           ▼            ▼
   ┌─────────┐  ┌──────────┐  ┌──────────────┐
   │  OpenAI │  │ ChromaDB │  │  Data Source │
   │  GPT-4o │  │ (Vector  │  │  CSV / SQL / │
   │  (LLM)  │  │  Store)  │  │  MongoDB /   │
   └────┬────┘  └────┬─────┘  │  Web Search  │
        │            │         └──────────────┘
        └─────┬──────┘
              ▼
     ┌─────────────────┐
     │  Plain English  │
     │  Answer + SQL / │
     │  Pandas / JSON  │
     └─────────────────┘
```

### RAG Pipeline (Document Q&A & CSV)

1. **Ingest** — Documents are loaded (PDF via `PyPDFLoader`, web via `WebBaseLoader`, tabular via `pandas`).
2. **Chunk** — Text is split into overlapping chunks using `RecursiveCharacterTextSplitter` (1 000 tokens, 200 overlap).
3. **Embed** — Each chunk is embedded with `OpenAIEmbeddings` and stored in a local **ChromaDB** vector store.
4. **Retrieve** — At query time, the top-k most semantically similar chunks are retrieved.
5. **Generate** — Retrieved context is injected into a `ChatPromptTemplate` and the LLM produces a grounded answer.

---

## 🚀 Installation

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/chat-with-anything.git
cd chat-with-anything
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🔑 Environment Variables

The app accepts your OpenAI API key in two ways — pick whichever suits your workflow.

### Option A — Enter it in the sidebar at runtime

The sidebar has a masked text field. The key is saved locally to `.config/config.json` and reloaded on the next session. No extra setup required.

### Option B — Set it as an environment variable before launching

```bash
# macOS / Linux / Windows (bash)
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

> **Note:** Never commit your API key. `.config/` and `.env` are already listed in `.gitignore`.

---

## 📦 Requirements Overview

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `langchain` + `langchain-core` | LLM orchestration & chains |
| `langchain-openai` | ChatOpenAI, OpenAIEmbeddings |
| `langchain-community` | Loaders, Chroma, SQLDatabase, DuckDuckGo tool |
| `langchain-text-splitters` | Document chunking |
| `openai` | OpenAI API client |
| `chromadb` | Local vector store |
| `pypdf` | PDF text extraction |
| `beautifulsoup4` | Web page scraping |
| `pymongo` | MongoDB driver |
| `sqlalchemy` | SQL database abstraction |
| `pandas` | Tabular data processing |
| `plotly` | Interactive benchmark charts |
| `duckduckgo-search` | Web search tool for the AI Agent |

---

## 🔬 Research

This project is developed as part of a **published research paper** from:

> **MIT World Peace University, Pune (MIT-WPU)**

### 📄 Paper Details

| Field | Details |
|-------|---------|
| **Title** | Chat With Anything: A Unified Natural Language Interface for Heterogeneous Data Sources |
| **Institution** | MIT World Peace University, Pune |
| **Authors** | Prashant Wadekar, Kuldeep Pawar, Abhijeet Singh, Gaurav Yadav |
| **Guidance** | Prof. Balaso Jagdale |

The research evaluates large language models across structured (CSV, SQL), semi-structured (MongoDB), and unstructured (PDF, web) data sources, benchmarking response accuracy, query correctness, and retrieval quality. The **Model Benchmark** tab in this app directly reflects the evaluation dataset from the paper.

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m "feat: add your feature"`
4. Push to your branch — `git push origin feature/your-feature`
5. Open a Pull Request

Please open an issue first for major changes so we can discuss the approach.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ at **MIT World Peace University, Pune**

</div>

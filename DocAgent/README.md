# RAG Agent with Tools

This project demonstrates a Retrieval-Augmented Generation (RAG) system with custom **Tools** and an **Agent**.  
It allows you to:
- Answer questions based on provided documents.
- Summarize documents.
- Extend with more tools easily.

## 🚀 Features
- Load multiple documents into a single vector database.
- Use **Answer Question** tool for context-based answers.
- Use **Summarize Document** tool for concise summaries.
- Agent dynamically decides which tool to use.

## 📂 Project Structure


📂 Project Structure
├── src/
│   ├── rag_agent.py        # main agent and tools
│   ├── utils.py            # helper functions
│   └── book/
│       └── odyssey.txt
│
├── requirements.txt
├── README.md
└── example_queries.md


## ⚙️ Installation


git clone <your-repo-url>
cd <your-repo>
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt




▶️ Usage
python src/rag_agent.py
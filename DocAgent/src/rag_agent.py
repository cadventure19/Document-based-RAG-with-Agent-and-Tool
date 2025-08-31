# --- Imports ---
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()

# --- Define directories ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# --- Local embeddings ---
class LocalEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embeddings = LocalEmbeddings()

# --- Load or create vector store ---
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("Vector store created and persisted.")
else:
    print("Vector store already exists.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# --- Retrieve relevant docs ---
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

# --- Chat history (example) ---
chat_history = [
    {"role": "user", "content": "Who was Odysseus?"},
    {"role": "assistant", "content": "Odysseus was a legendary Greek hero from Homer’s epics, the Iliad and the Odyssey"}
]

# --- OpenAI Client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Contextualize query ---
def contextualize_query(user_input, chat_history, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

# --- System prompt ---
contextualize_q_system_prompt = """You are a helpful assistant that rewrites a user's question into a standalone question considering previous chat history."""

# --- Tool: Answer Question ---
def answer_question(query, **kwargs):
    # Step 1: Rewrite query
    rewritten_query = contextualize_query(query, chat_history, contextualize_q_system_prompt)

    # Step 2: Retrieve docs
    print("\nRewritten Query:", rewritten_query)
    relevant_docs = retriever.get_relevant_documents(rewritten_query)
    print("Retrieved Docs:", relevant_docs)

    # Step 3: Join docs as context
    context_text = "\n".join([doc.page_content for doc in relevant_docs])

    # Step 4: Ask GPT using retrieved context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based on the context_text provided only. Keep answer short within 3 lines."},
            {"role": "user", "content": f"Answer the following question based on the context below:\n{context_text}\nQuestion: {rewritten_query}"}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# --- Tool: Summarize ---
def summarize_docs(question):
    # Step 1: Retrieve docs
    docs = retriever.get_relevant_documents(question)

    # Step 2: Join docs as context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Summarize with GPT
    prompt = f"Summarize the following text in a concise way:\n{context}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- Tools ---
tools = {
    "answer_question": answer_question,
    "summarize": summarize_docs
}

# --- Agent Function ---
def agent(query, task="answer_question", chat_history=["There is a document on Odyssey."]):
    if task not in tools:
        return "Unknown tool!"
    return tools[task](query)

# --- Usage ---
if __name__ == "__main__":
    print("\n--- Agent Answers ---")
    print(agent("Who was Odysseus?", task="answer_question"))
    print(agent("Summarize the Odyssey story in short", task="summarize"))

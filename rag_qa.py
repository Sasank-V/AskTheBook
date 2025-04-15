import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Loads the .env file to access GROQ_API_KEY


def load_groq_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="gemma2-9b-it",
        temperature=0.3,
    )


def load_qa_chain(subject: str):
    index_path = f"index/{subject}"
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index for subject '{subject}' not found at {index_path}"
        )

    # Load FAISS vector DB
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = load_groq_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

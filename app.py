from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# LangChain
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone as PineconeStore

# HuggingFace API (NOT local)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Pinecone
from pinecone import Pinecone

# Custom
from src.prompt import system_prompt

# -------------------- INIT --------------------
app = Flask(__name__)
load_dotenv()

# -------------------- ENV --------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("Missing COHERE_API_KEY")

if not HF_API_KEY:
    raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN")

# -------------------- EMBEDDINGS (API BASED) --------------------
def get_embeddings():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_KEY,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -------------------- PINECONE --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

def get_retriever():
    docsearch = PineconeStore.from_existing_index(
        index_name=index_name,
        embedding=get_embeddings()
    )

    return docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

# -------------------- LLM --------------------
chatModel = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0.5
)

# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# -------------------- CHAIN --------------------
def get_rag_chain():
    retriever = get_retriever()

    qa_chain = create_stuff_documents_chain(
        chatModel, prompt
    )

    return create_retrieval_chain(
        retriever, qa_chain
    )

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    if not msg:
        return "No input provided"

    try:
        rag_chain = get_rag_chain()

        response = rag_chain.invoke({"input": msg})

        if not response:
            return "No response"

        answer = response.get("answer", "No answer generated")

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return "Server error: " + str(e)

    return str(answer)

# -------------------- RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

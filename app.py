from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# LangChain + AI
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_pinecone import PineconeVectorStore

# Pinecone NEW SDK
from pinecone import Pinecone

# Custom files
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# -------------------- INIT --------------------
app = Flask(__name__)

load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# -------------------- EMBEDDINGS --------------------
embeddings = download_hugging_face_embeddings()

# -------------------- PINECONE (NEW SDK) --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Connect to existing index
index = pc.Index(index_name)

# LangChain wrapper
docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# -------------------- LLM (COHERE) --------------------
chatModel = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-08-2024",   # stable model
    temperature=0.5
)

# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# -------------------- CHAINS --------------------
question_answer_chain = create_stuff_documents_chain(
    chatModel, prompt
)

rag_chain = create_retrieval_chain(
    retriever, question_answer_chain
)

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "No response generated.")

    except Exception as e:
        print("Error:", str(e))
        answer = "Sorry, something went wrong. Please try again."

    return str(answer)


# -------------------- RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
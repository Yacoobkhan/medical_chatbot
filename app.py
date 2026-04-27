from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# -------------------- INIT --------------------
app = Flask(__name__)

load_dotenv()

# Load API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY



# -------------------- EMBEDDINGS --------------------
embeddings = download_hugging_face_embeddings()

# -------------------- PINECONE --------------------

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-east-1"  # must match your index region
)

index_name = "medicalbot"

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -------------------- LLM (COHERE) --------------------
chatModel = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="c4ai-aya-expanse-32b",   # or "command-r-plus"
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
    msg = request.form["msg"]
    print("User:", msg)

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]

    except Exception as e:
        print("Error:", str(e))
        answer = "Sorry, something went wrong. Please try again."

    print("Bot:", answer)
    return str(answer)


# -------------------- RUN --------------------
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT, debug=True)
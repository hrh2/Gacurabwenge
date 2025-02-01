from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Initialize FastAPI
app = FastAPI()



load_dotenv()  # Load variables from .env

# Now this should work
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize ChromaDB
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="data/chroma"
)


class DocumentInput(BaseModel):
    text: str
    metadata: dict


class QueryInput(BaseModel):
    question: str


# Store knowledge (POST request)
from fastapi import Query
import json

@app.post("/store/")
def store_document(text: str = Query(..., description="The document text"),
                   metadata: str = Query(..., description="Metadata in JSON format")):
    try:
        metadata_dict = json.loads(metadata)  # Convert JSON string to dictionary
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata format. Use valid JSON.")

    vector_store.add_texts(texts=[text], metadatas=[metadata_dict])
    vector_store.persist()
    return {"message": "Document stored successfully"}



# Answer user queries (GET request)
@app.post("/ask/")
def ask_question(data: QueryInput):
    retriever = vector_store.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant From Rwanda Coding Academy(RCA) Developed by HIRWA  Rukundo Hope,HAKIZIMANA yves and Mucyo Moses you name is Gacurabwenge . 
        Answer the user's question based on the retrieved information:

        {context}

        User's question: {question}
        """
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini",temperature=0.0),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    response = rag_chain.run(data.question)
    return {"answer": response}

# Run API with: uvicorn main:app --reload

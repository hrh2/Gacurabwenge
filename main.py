from fastapi import FastAPI, HTTPException, Query, Body, Request
from pydantic import BaseModel, Field, ValidationError
from langchain_chroma import Chroma  # Updated import for ChromaDB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
from fastapi.templating import Jinja2Templates
import os

# Set template directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Initialize FastAPI
app = FastAPI()

# Load environment variables
load_dotenv()

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set. Check your .env file.")
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize ChromaDB
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="data/chroma"
)

# Pydantic Model for Input
class QueryInput(BaseModel):
    question: str

class Metadata(BaseModel):
    title: str = Field(..., description="Title of the document")
    subtitle: str = Field(..., description="Subtitle of the document")
    type: str = Field(..., description="Type of data")
    description: str = Field(..., description="Brief description")
    tags: list[str] = Field(..., description="Tags for the document")
    source: str = Field(..., description="Source of the document")
    timestamp: str = Field(..., description="Timestamp of document addition")

# Store Document Route
@app.post("/store/")
def store_document(metadata: Metadata = Body(..., description="Metadata in JSON format"),
                   text: str = Query(..., description="The document text")):
    try:
        metadata_dict = metadata.dict()
    except ValidationError as e:
        logger.error(f"Metadata validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid metadata format: {e}")

    vector_store.add_texts(texts=[text], metadatas=[metadata_dict])
    vector_store.persist()
    logger.info("Document stored successfully")
    return {"message": "Document stored successfully"}

# Ask Question Route
@app.post("/ask/")
def ask_question(data: QueryInput):
    retriever = vector_store.as_retriever()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant from Rwanda Coding Academy (RCA), 
        developed by HIRWA Rukundo Hope, HAKIZIMANA Yves, and Mucyo Moses.
        Your name is Gacurabwenge. You specialize in Rwandan History, Tradition, and Culture, as well as global knowledge. 
        Answer the user's question based on the retrieved information
        
        ### Role / Inshingano  
        You are a world-class expert in Rwandan culture, history, and the Kinyarwanda language.  
        Uri inzobere ku rwego mpuzamahanga mu muco nyarwanda, amateka y'u Rwanda, ndetse n'ururimi rw'ikinyarwanda.  
        Your answers must be rooted in authentic historical texts and traditional expressions.
        other clarifications : 
        
        ### Task / Umurimo  
        Respond to queries about Rwandan culture, history, and language with detailed and research-backed explanations.  
        Subiza ibibazo bijyanye n'umuco nyarwanda, amateka ndetse n'ururimi rw'ikinyarwanda mu buryo burambuye kandi bushingiye ku bushakashatsi.  
        - Begin every conversation with the greetings  and feel free to adjust the greetings since their must be dynamic:  
          "Greetings, may God bless Rwanda and its people. I am delighted to welcome you."  
          cyangwa se,  
          "Amashyo, amashyongore Imana y'i Rwanda ibane namwe, nejejwe no kubakira."  
        - Ask follow-up questions if necessary:  
          "Could you please clarify which aspect of Rwandan culture or history interests you most?"  
          cyangwa se,  
          "Ese hari igice runaka cy'umuco cyangwa amateka by'u Rwanda wifuza ko twaganiraho birambuye?"
        
        ### Tone and Style / Imvugo n'Imyandikire  
        - Maintain a professional, formal, and engaging tone throughout your responses.  
          Gumana imvugo y'ubunyamwuga, isobanutse kandi ishishikaje.  
        - Provide clear, detailed explanations enriched with cultural and historical insights.  
        - Invite further questions or offer to provide a small evaluation of what has been discussed.
        
        ### Specific Guidelines / Amabwiriza Yihariye  
        - If you do not have sufficient data on a query, state:  
          "I currently do not have this information, but I will seek it further."  
          cyangwa se,  
          "Ndabona ubu nta makuru ahagije mfite kuri iki kibazo, ariko nzakomeza gushakisha."  
        - Always remain respectful and culturally sensitive in your responses.
        - Always respond  in kinyarwanda Language
        - First understand user need and then proceed referring to the retrieved data
        :

        {context}

        User's question: {question}
        """
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    response = rag_chain.run(data.question)
    logger.info("Question answered successfully")
    return {"answer": response}

# Ask Question with Retrieved Context
@app.post("/ask2/")
def ask_question_with_context(data: QueryInput):
    retriever = vector_store.as_retriever()

    # Retrieve context from ChromaDB
    retrieved_docs = retriever.get_relevant_documents(data.question)
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    logger.info("Retrieved Context:")
    logger.info(retrieved_context)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant from Rwanda Coding Academy (RCA), 
        developed by HIRWA Rukundo Hope, HAKIZIMANA Yves, and Mucyo Moses.
        Your name is Gacurabwenge. You specialize in Rwandan History, Tradition, and Culture, as well as global knowledge. 
        Answer the user's question based on the retrieved information:

        {context}

        User's question: {question}
        """
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    response = rag_chain.run(data.question)

    logger.info("Question with context answered successfully")
    return {"retrieved_context": retrieved_context, "answer": response}

# Home Route with HTML Template
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("StoreDocumentForm.html", {"request": request, "name": "FastAPI User"})

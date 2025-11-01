import os
import torch
import faiss
import pickle
import numpy as np
from transformers import AutoTokenizer, pipeline, AutoModel, AutoModelForCausalLM
from peft import PeftModel

from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from groq import Groq
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# --- Configuration ---
INDEX_PATH = "/teamspace/studios/this_studio/Study-Spark/data/processed/faiss_index.bin"
CHUNKS_PATH = "/teamspace/studios/this_studio/Study-Spark/data/processed/chunks.pkl"
EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m'
TRANSCRIPTION_MODEL = "openai/whisper-base"
BASE_MODEL_NAME = 'google/gemma-3-270m-it'

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# --- Model and Data Store ---
models = {}
app_runnable = None

def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_embeddings(text, model, tokenizer, device):
    """Generates an embedding for a given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def retrieve_relevant_chunks(query_embedding, index, chunks, top_k=3):
    """Retrieves the most relevant chunks from the FAISS index."""
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in indices[0]]

# --- Langgraph State ---
class GraphState(TypedDict):
    transcript: str
    retrieved_text: str
    summary: str
    audio_file: str
    llm: str

# --- Langgraph Nodes ---
def transcribe_node(state: GraphState):
    audio_file = state["audio_file"]
    transcription = models["transcription_pipeline"](audio_file)["text"]
    return {"transcript": transcription}

def retrieve_node(state: GraphState):
    transcript_embedding = get_embeddings(
        state["transcript"],
        models["embedding_model"],
        models["embedding_tokenizer"],
        models["device"]
    )
    retrieved_chunks = retrieve_relevant_chunks(
        transcript_embedding,
        models["faiss_index"],
        models["chunks"]
    )
    retrieved_text = "\n\n---\n\n".join(retrieved_chunks)
    return {"retrieved_text": retrieved_text}

def generate_node(state: GraphState):
    llm = state["llm"]
    summary = ""
    if llm == "groq":
        prompt = models["prompt_template"].format(transcript=state["transcript"], retrieved_text=state["retrieved_text"])
        chat_completion = models["groq_client"].chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=2048,
        )
        summary = chat_completion.choices[0].message.content
    elif llm == "finetuned":
        prompt = models["prompt_template"].format(transcript=state["transcript"], retrieved_text=state["retrieved_text"])
        inputs = models["finetuned_tokenizer"](prompt, return_tensors="pt").to(models["device"])
        outputs = models["finetuned_model"].generate(**inputs, max_length=2048)
        summary = models["finetuned_tokenizer"].decode(outputs[0], skip_special_tokens=True)
    else:
        raise ValueError(f"Invalid LLM option: {llm}")
    return {"summary": summary}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models and data at startup
    global app_runnable
    models["device"] = get_device()
    
    # Transcription
    models["transcription_pipeline"] = pipeline(
        "automatic-speech-recognition",
        model=TRANSCRIPTION_MODEL,
        device=models["device"],
        generate_kwargs={"language": "en"}
    )
    
    # Retrieval
    models["embedding_tokenizer"] = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    models["embedding_model"] = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(models["device"])
    models["faiss_index"] = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        models["chunks"] = pickle.load(f)
        
    # Generation
    models["groq_client"] = Groq(api_key=GROQ_API_KEY)
    models["prompt_template"] = ChatPromptTemplate.from_messages(
        [("human", "## Lecture Transcript:\n{transcript}\n\n## Retrieved Textbook Excerpts:\n{retrieved_text}\n\nBased on the lecture transcript and the retrieved textbook excerpts, generate a comprehensive study guide or summary.")]
    )
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(models["device"])
    models["finetuned_model"] = PeftModel.from_pretrained(base_model, "/teamspace/studios/this_studio/Study-Spark/models/gemma-lora-finetuned").to(models["device"])
    models["finetuned_tokenizer"] = AutoTokenizer.from_pretrained("/teamspace/studios/this_studio/Study-Spark/models/gemma-lora-finetuned")
    


    # Build Langgraph
    workflow = StateGraph(GraphState)
    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("transcribe")
    workflow.add_edge("transcribe", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    app_runnable = workflow.compile()
    
    yield
    # Clean up resources if needed on shutdown
    models.clear()

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="/teamspace/studios/this_studio/Study-Spark/app/web/static"), name="static")
templates = Jinja2Templates(directory="/teamspace/studios/this_studio/Study-Spark/app/web/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/study/")
async def study_endpoint(file: UploadFile = File(...), llm: str = Form(...)):
    temp_dir = "/teamspace/studios/this_studio/Study-Spark/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        initial_state = {
            "audio_file": temp_file_path,
            "llm": llm,
            "transcript": "",
            "retrieved_text": "",
            "summary": ""
        }
        
        result = app_runnable.invoke(initial_state)
        
        return JSONResponse(content={"summary": result.get("summary", "")})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

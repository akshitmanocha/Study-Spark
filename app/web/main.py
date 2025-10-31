from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import torch
import faiss
import pickle
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    pipeline,
)
from peft import PeftModel

# --- Configuration ---
INDEX_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/faiss_index.bin"
CHUNKS_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/chunks.pkl"
EMBEDDING_MODEL_NAME = 'google/embedding-gemma-300m'
BASE_MODEL_NAME = 'microsoft/Phi-3-mini-4k-instruct'
LORA_ADAPTER_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/models/phi3-lora-finetuned/final_model"
TRANSCRIPTION_MODEL = "openai/whisper-base"

# --- FastAPI App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Model Loading (Done once at startup) ---

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

device = get_device()

# Transcription pipeline
transcriber = pipeline("automatic-speech-recognition", model=TRANSCRIPTION_MODEL, device=device)

# Embedding model
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

# Summarization model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
summarization_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
summarization_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
summarization_tokenizer.pad_token = summarization_tokenizer.eos_token

# FAISS Index
faiss_index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, 'rb') as f:
    all_chunks = pickle.load(f)

# --- Helper Functions ---

def get_embeddings(text):
    inputs = embedding_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def retrieve_relevant_chunks(query_embedding, top_k=3):
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    return [all_chunks[i] for i in indices[0]]

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    # 1. Save and Transcribe Audio
    temp_audio_path = f"temp_{{file.filename}}"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    transcription = transcriber(temp_audio_path)["text"]
    os.remove(temp_audio_path)

    # 2. Perform RAG
    transcript_embedding = get_embeddings(transcription)
    retrieved_chunks = retrieve_relevant_chunks(transcript_embedding)
    retrieved_text = "\n\n---\n\n".join(retrieved_chunks)

    # 3. Generate Summary
    prompt = f"<|user|>\n## Lecture Transcript:\n{{transcription}}\n\n## Retrieved Textbook Excerpts:\n{{retrieved_text}}<|end|>\n<|assistant|>\n"
    inputs = summarization_tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
    outputs = summarization_model.generate(**inputs, max_length=2048)
    summary = summarization_tokenizer.batch_decode(outputs)[0]
    summary = summary.split("<|assistant|>\n")[1].replace("<|end|>", "").strip()

    return {"transcription": transcription, "summary": summary}

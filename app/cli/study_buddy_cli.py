
import os
import click
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

# --- Helper Functions ---

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

# --- CLI Commands ---

@click.group()
def cli():
    """A CLI tool to assist with studying using a fine-tuned LLM."""
    pass

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output-file', default='transcription.txt', help='The file to save the transcription to.')
def transcribe(audio_file, output_file):
    """Transcribes an audio file to text using Whisper."""
    click.echo(f"Transcribing {audio_file}...")
    device = get_device()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=TRANSCRIPTION_MODEL,
        device=device
    )
    transcription = pipe(audio_file)["text"]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    click.echo(f"Transcription complete. Saved to {output_file}")

@cli.command()
@click.argument('transcript_file', type=click.Path(exists=True))
def summarize(transcript_file):
    """Summarizes a lecture transcript using the fine-tuned RAG model."""
    click.echo("Loading models and data...")
    device = get_device()

    # Load fine-tuned model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load embedding model
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

    # Load FAISS index and chunks
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        all_chunks = pickle.load(f)

    click.echo("Performing RAG...")
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    # Retrieve relevant chunks
    transcript_embedding = get_embeddings(transcript, embedding_model, embedding_tokenizer, device)
    retrieved_chunks = retrieve_relevant_chunks(transcript_embedding, faiss_index, all_chunks)
    retrieved_text = "\n\n---\n\n".join(retrieved_chunks)

    # Prepare prompt for fine-tuned model
    prompt = f"<|user|>
## Lecture Transcript:
{transcript}

## Retrieved Textbook Excerpts:
{retrieved_text}<|end|>
<|assistant|>
"

    click.echo("Generating summary...")
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
    outputs = model.generate(**inputs, max_length=2048)
    summary = tokenizer.batch_decode(outputs)[0]

    # Clean up the output
    summary = summary.split("<|assistant|>")[1].replace("<|end|>").strip()

    click.echo("\n--- Summary ---")
    click.echo(summary)

if __name__ == "__main__":
    cli()

import os
import click
import torch
import faiss
import pickle
import numpy as np
from transformers import AutoTokenizer, pipeline, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from groq import Groq

# Load environment variables
load_dotenv()

# --- Configuration ---
INDEX_PATH = "/teamspace/studios/this_studio/Study-Spark/data/processed/faiss_index.bin"
CHUNKS_PATH = "/teamspace/studios/this_studio/Study-Spark/data/processed/chunks.pkl"
EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m'
TRANSCRIPTION_MODEL = "openai/whisper-base"
BASE_MODEL_NAME = 'google/gemma-3-270m-it'
LORA_ADAPTER_PATH = "/teamspace/studios/this_studio/Study-Spark/models/gemma-lora-finetuned"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

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

# --- Langgraph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        transcript: The transcribed text from the audio.
        retrieved_text: Relevant text chunks retrieved from the knowledge base.
        summary: The generated summary.
        notes_file: The file to append notes to.
        audio_file: The path to the audio file.
        llm: The language model to use for generation.
    """
    transcript: str
    retrieved_text: str
    summary: str
    notes_file: str
    audio_file: str
    llm: str

# --- Langgraph Nodes ---

def transcribe_node(state: GraphState):
    """Transcribes an audio file to text using Whisper."""
    audio_file = state["audio_file"]
    device = get_device()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=TRANSCRIPTION_MODEL,
        device=device,
        generate_kwargs={"language": "en"}
    )
    transcription = pipe(audio_file)["text"]
    return {"transcript": transcription}

def retrieve_node(state: GraphState):
    """Retrieves relevant chunks from the FAISS index based on the transcript."""
    device = get_device()
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

    faiss_index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        all_chunks = pickle.load(f)
    transcript_embedding = get_embeddings(state["transcript"], embedding_model, embedding_tokenizer, device)
    retrieved_chunks = retrieve_relevant_chunks(transcript_embedding, faiss_index, all_chunks)
    retrieved_text = "\n\n---\n\n".join(retrieved_chunks)
    return {"retrieved_text": retrieved_text}

def generate_node(state: GraphState):
    """Generates a summary using the selected LLM."""
    llm = state["llm"]

    if llm == "groq":
        client = Groq(api_key=GROQ_API_KEY)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", "## Lecture Transcript:\n{transcript}\n\n## Retrieved Textbook Excerpts:\n{retrieved_text}\n\nBased on the lecture transcript and the retrieved textbook excerpts, generate a comprehensive study guide or summary."),
            ]
        )
        
        prompt = prompt_template.format(transcript=state["transcript"], retrieved_text=state["retrieved_text"])

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=2048,
        )
        
        summary = chat_completion.choices[0].message.content

    elif llm == "finetuned":
        device = get_device()
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        prompt = f"""<|user|>
## Lecture Transcript:
{state['transcript']}

## Retrieved Textbook Excerpts:
{state['retrieved_text']}<|end|>
<|assistant|>
"""

        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
        outputs = model.generate(**inputs, max_length=2048)
        summary = tokenizer.batch_decode(outputs)[0]
        summary = summary.split("<|assistant|>")[1].replace("<|end|>").strip()

    else:
        raise ValueError(f"Invalid LLM option: {llm}")
    return {"summary": summary}

def append_notes_node(state: GraphState):
    """Appends the generated summary to the notes file."""
    notes_file = state["notes_file"]
    with open(notes_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- Study Guide ---\n\n{state["summary"]}\n")
    return {}

# --- CLI Commands ---

@click.group()
def cli():
    """A CLI tool to assist with studying using a fine-tuned LLM."""
    pass

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--notes-file', default='notes.md', help='The file to append the study guide to.')
@click.option('--llm', type=click.Choice(['finetuned', 'groq']), default='groq', help='The LLM to use for generation.')
def study(audio_file, notes_file, llm):
    """Generates a study guide from an audio file and appends it to a notes file."""
    
    # Build the Langgraph workflow
    workflow = StateGraph(GraphState)

    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("append_notes", append_notes_node)

    workflow.set_entry_point("transcribe")
    workflow.add_edge("transcribe", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "append_notes")
    workflow.add_edge("append_notes", END)

    app = workflow.compile()

    # Execute the workflow
    initial_state: GraphState = {"transcript": "", "retrieved_text": "", "summary": "", "notes_file": notes_file, "audio_file": audio_file, "llm": llm}
    for s in app.stream(initial_state):
        pass

if __name__ == "__main__":
    cli()
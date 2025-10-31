
import os
import re
import json
import pickle
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai

# --- Configuration ---
TEXT_FILE_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/UnderstandingDeepLearning.txt"
INDEX_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/faiss_index.bin"
CHUNKS_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/chunks.pkl"
QUESTIONS_FILE_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/questions.txt"
DATASET_FILE_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/finetuning_dataset.jsonl"
EMBEDDING_MODEL_NAME = 'google/embedding-gemma-300m'

# --- Gemini API Configuration ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

# --- Helper Functions ---

def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def parse_chapters(file_path):
    """Parses the book into chapters."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # List of chapter titles based on the ToC
    chapter_titles = [
        "1 Introduction", "2 Supervised learning", "3 Shallow neural networks", 
        "4 Deep neural networks", "5 Loss functions", "6 Fitting models", 
        "7 Gradients and initialization", "8 Measuring performance", "9 Regularization", 
        "10 Convolutional networks", "11 Residual networks", "12 Transformers", 
        "13 Graph neural networks", "14 Unsupervised learning", 
        "15 Generative Adversarial Networks", "16 Normalizing flows", 
        "17 Variational autoencoders", "18 Diffusion models", 
        "19 Reinforcement learning", "20 Why does deep learning work?"
    ]

    chapters = {}
    # Use regex to split by chapter titles
    # The pattern looks for the chapter number, title, and page number
    split_points = [m.start() for m in re.finditer(r'^(\d+ .*) (\d+)$' , text, re.MULTILINE)]
    
    for i, title in enumerate(chapter_titles):
        start = text.find(title)
        end = -1
        if i + 1 < len(chapter_titles):
            end = text.find(chapter_titles[i+1])
        
        if start != -1:
            chapter_text = text[start:end].strip() if end != -1 else text[start:].strip()
            chapters[title] = chapter_text

    return chapters

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

def generate_with_gemini(prompt):
    """Generates content using the Gemini API."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return ""

# --- Main Script ---
if __name__ == "__main__":
    # 1. Load models and data
    print("Loading models and data...")
    device = get_device()
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        all_chunks = pickle.load(f)

    # 2. Parse chapters
    print("Parsing chapters...")
    chapters = parse_chapters(TEXT_FILE_PATH)
    print(f"Found {len(chapters)} chapters.")

    # 3. Process each chapter
    with open(QUESTIONS_FILE_PATH, 'w', encoding='utf-8') as qf, \
         open(DATASET_FILE_PATH, 'w', encoding='utf-8') as df:

        for title, content in chapters.items():
            print(f"\n--- Processing Chapter: {title} ---")

            # a. Generate lecture
            print("Generating lecture content...")
            lecture_prompt = f"You are a university professor giving a lecture on the following chapter from the book 'Understanding Deep Learning'. Your lecture should be a concise and engaging overview of the key concepts. Do not just summarize the text, but present it in a lecture format. Here is the chapter content:\n\n{content[:4000]}" # Use a portion to avoid overly long prompts
            lecture = generate_with_gemini(lecture_prompt)

            # b. Generate questions
            print("Generating questions...")
            questions_prompt = f"Generate 5 insightful questions based on the following chapter content. The questions should test a deep understanding of the material.\n\n{content[:4000]}"
            questions = generate_with_gemini(questions_prompt)
            qf.write(f"## Questions for {title}\n")
            qf.write(questions)
            qf.write("\n\n")

            # c. Retrieve relevant chunks
            print("Retrieving relevant chunks...")
            lecture_embedding = get_embeddings(lecture, embedding_model, embedding_tokenizer, device)
            retrieved_chunks = retrieve_relevant_chunks(lecture_embedding, faiss_index, all_chunks)
            retrieved_text = "\n\n---\n\n".join(retrieved_chunks)

            # d. Summarize for dataset
            print("Generating summary for dataset...")
            summary_prompt = f"You are an expert in deep learning. Your task is to create a high-quality, concise summary of the following text, which is a combination of a lecture and relevant excerpts from a textbook. The summary should be in the form of notes that a student could use to study. Here is the text:\n\n## Lecture Content:\n{lecture}\n\n## Retrieved Textbook Excerpts:\n{retrieved_text}"
            summary = generate_with_gemini(summary_prompt)

            # e. Save to dataset file
            dataset_entry = {
                "input": f"## Lecture Content:\n{lecture}\n\n## Retrieved Textbook Excerpts:\n{retrieved_text}",
                "output": summary
            }
            df.write(json.dumps(dataset_entry) + '\n')

    print("\n\nProcessing complete.")
    print(f"Questions saved to: {QUESTIONS_FILE_PATH}")
    print(f"Finetuning dataset saved to: {DATASET_FILE_PATH}")


import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import re
import pickle

def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def chunk_text_by_page(file_path):
    """Chunks a text file by page, using '--- Page X ---' as a delimiter."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split the text by page separators
    chunks = re.split(r'--- Page \d+ ---', text)
    
    # Filter out empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks

def get_embeddings(chunks, model, tokenizer, device):
    """Generates embeddings for a list of text chunks."""
    model.to(device)
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling to get sentence embedding
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(sentence_embedding)
    return np.array(embeddings)

def create_and_save_faiss_index(embeddings, index_path):
    """Creates a FAISS index and saves it to a file."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index created and saved to {index_path}")

def save_chunks(chunks, chunks_path):
    """Saves the text chunks to a file using pickle."""
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"Text chunks saved to {chunks_path}")

if __name__ == "__main__":
    TEXT_FILE_PATH = "/teamspace/studios/this_studio/Study-Spark/data/processed/UnderstandingDeepLearning.txt"
    INDEX_PATH = "/teamspace/studios/this_studio/Study-Spark/data/processed/faiss_index.bin"
    CHUNKS_PATH = "/teamspace/studios/this_studio/Study-Spark/data/processed/chunks.pkl"
    MODEL_NAME = 'google/embeddinggemma-300m'

    # 1. Get device
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    # 3. Chunk the text
    print(f"Chunking text file: {TEXT_FILE_PATH}...")
    chunks = chunk_text_by_page(TEXT_FILE_PATH)
    print(f"Found {len(chunks)} chunks.")

    # 4. Generate embeddings
    print("Generating embeddings...")
    embeddings = get_embeddings(chunks, model, tokenizer, device)
    print(f"Embeddings generated with shape: {embeddings.shape}")

    # 5. Create and save FAISS index
    create_and_save_faiss_index(embeddings, INDEX_PATH)

    # 6. Save the text chunks
    save_chunks(chunks, CHUNKS_PATH)

    print("\nProcessing complete.")
    print(f"Your FAISS index is saved at: {INDEX_PATH}")
    print(f"Your text chunks are saved at: {CHUNKS_PATH}")

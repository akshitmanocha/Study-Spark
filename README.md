# Study-Spark: The Agentic Study Buddy

Study-Spark is a comprehensive suite of tools designed to revolutionize your study process. By leveraging cutting-edge AI, this project transforms lecture audio and textbook content into high-quality, structured study guides. At its core, Study-Spark is powered by a sophisticated Retrieval-Augmented Generation (RAG) pipeline and a fine-tuned language model, all accessible through an intuitive web interface.

## Core Architecture

The system is built on a modular architecture that seamlessly integrates data processing, model training, and inference.

### 1. Knowledge Base Creation

The foundation of Study-Spark is a robust knowledge base created from your textbook.

- **PDF Text Extraction:** The process begins with the `extract_text_from_pdf.py` script, which carefully extracts text from your PDF textbook while preserving page breaks. This ensures that the retrieved context is accurate and easy to trace back to the source.
- **FAISS Indexing:** The `create_faiss_index.py` script builds a searchable knowledge base. It uses the powerful `google/embedding-gemma-300m` model to generate vector embeddings for each page, which are then stored in a FAISS index for lightning-fast similarity searches.

### 2. Synthetic Dataset Generation

To ensure the highest quality summaries, Study-Spark includes a pipeline for generating a synthetic dataset to fine-tune a custom language model.

- **Automated Dataset Creation:** The `generate_synthetic_dataset.py` script automates this process using the Gemini API to:
    1.  **Parse Chapters:** Divide the textbook into chapters to create structured, topic-focused content.
    2.  **Generate Lectures:** Create lecture-style content for key topics within each chapter, simulating a real classroom experience.
    3.  **Retrieve Context:** Use the FAISS index to retrieve the most relevant text chunks from the textbook based on the generated lecture.
    4.  **Generate Summaries:** Summarize the lecture and retrieved text to create high-quality `(source, summary)` pairs, which form the backbone of our fine-tuning dataset.

### 3. Model Fine-Tuning

The fine-tuning process enhances the language model's ability to generate domain-specific, high-quality summaries.

- **LoRA Fine-Tuning:** The `train_lora_model.py` script fine-tunes the `google/gemma-3-270m-it` model using Low-Rank Adaptation (LoRA), a parameter-efficient technique ideal for resource-constrained environments.
- **LLM as a Judge:** To ensure the quality of the fine-tuned model, the script includes an "LLM as a Judge" evaluation step, where the Gemini API assesses the quality of the generated summaries.

### 4. Inference and Application

The final step is to bring all the components together in a user-friendly web application.

- **FastAPI Web App:** The `app/web/main.py` file contains a FastAPI application that provides a clean, modern front-end for the system.
- **LangGraph Workflow:** The application's logic is orchestrated by a LangGraph state machine, which manages the following workflow:
    1.  **Transcription:** Transcribes uploaded audio files using `openai/whisper-base`.
    2.  **Retrieval:** Retrieves relevant text from the FAISS index based on the transcript.
    3.  **Generation:** Generates a comprehensive summary using either a powerful model via the Groq API or the locally fine-tuned LoRA model.

## Scope for Improvement

The current version of Study-Spark is a powerful tool, but there are many opportunities for future enhancements.

### 1. Fine-Tuning and Model Optimization

- **Experiment with Larger Models:** While `gemma-3-270m-it` is a solid baseline, using a more powerful model like `gemma-3-9b-it` or other large open-source models could significantly improve performance.
- **Refine the Dataset:** The quality of the fine-tuned model is directly tied to the training data. The synthetic data generation pipeline could be enhanced to create a larger, more diverse, and more accurate dataset.

### 2. RAG Pipeline Enhancements

- **Advanced Retrieval Techniques:** Explore advanced retrieval methods such as BM25 + FAISS hybrid search or SPLADE to improve the relevance and accuracy of retrieved information.
- **Optimized Chunking Strategies:** Move beyond basic page-level chunking to implement more sophisticated strategies like semantic chunking, fixed-size windowing with overlap, or even hierarchical chunking to better capture context and improve retrieval quality.
- **Improved PDF Text Extraction:** Optimize the PDF text extraction process to handle complex layouts, tables, and figures more accurately, ensuring a higher quality input for the knowledge base.

### 3. Application and User Experience

- **Model Quantization:** To reduce the resource footprint of the fine-tuned model, it could be quantized to a lower precision (e.g., 4-bit or 8-bit), making it faster and more memory-efficient.
- **Streaming Responses:** For a more interactive user experience, the web application could be updated to stream the generated summary in real-time, rather than waiting for the full response.
- **CLI Integration:** In addition to the web interface, a command-line interface (CLI) could be developed to allow for more programmatic and automated use of the system.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   An NVIDIA GPU is recommended for training and inference.

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd Study-Spark
    ```

3.  **Install Dependencies:**
    ```bash
    pip install transformers torch peft datasets mlflow accelerate bitsandbytes fastapi uvicorn python-multipart click soundfile google-generativeai faiss-cpu pypdfium2
    ```
    *(Note: For improved performance on a CUDA-enabled GPU, install the GPU version of FAISS: `pip install faiss-gpu`)*

4.  **Set Up Environment Variables:**
    You will need a Google API key for the Gemini API. Set it as an environment variable:
    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    ```

## End-to-End Workflow

Follow these steps to process your data, train the model, and run the application.

1.  **Add Your Textbook:**
    Place your textbook (e.g., `UnderstandingDeepLearning.pdf`) in the `data/raw/` directory.

2.  **Extract Text from PDF:**
    ```bash
    python scripts/extract_text_from_pdf.py data/raw/UnderstandingDeepLearning.pdf data/processed/UnderstandingDeepLearning.txt
    ```

3.  **Create the RAG Index:**
    ```bash
    python scripts/create_faiss_index.py
    ```

4.  **Generate the Synthetic Dataset:**
    ```bash
    python scripts/generate_synthetic_dataset.py
    ```

5.  **Train the LoRA Model:**
    ```bash
    python scripts/train_lora_model.py
    ```

6.  **Run the Web Application:**
    ```bash
    cd app/web
    uvicorn main:app --reload
    ```
    You can then access the web interface at [http://127.0.0.1:8000](http://127.0.0.1:8000).
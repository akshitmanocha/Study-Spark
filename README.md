# Study-Spark: The Agentic Study Buddy

Study-Spark is a comprehensive suite of tools designed to help you study more effectively. It leverages modern AI techniques to process lecture audio and textbook content, creating high-quality summaries and notes. The project includes a web interface for easy interaction.

## Features

*   **Automated Transcription:** Transcribe lecture audio files into text using OpenAI's Whisper model.
*   **RAG-based Summarization:** Utilizes a Retrieval-Augmented Generation (RAG) pipeline with a FAISS vector store to provide contextually rich summaries.
*   **Fine-Tuned Language Model:** Includes scripts to create a synthetic dataset and fine-tune a Small Language Model (SLM) using LoRA for high-quality summarization.
*   **Web Interface:** A clean and simple web UI built with FastAPI and Tailwind CSS to interact with the system.
*   **CLI:** A command-line interface for power users to access the core functionalities.

## Project Structure

```
/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── app/
│   ├── cli/
│   └── web/
├── models/
└── GEMINI.md
```

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   An NVIDIA GPU is recommended for training and for running the models.

2.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Study-Spark
    ```

3.  **Install Dependencies:**
    Install all the necessary Python libraries:
    ```bash
    pip install transformers torch peft datasets mlflow accelerate bitsandbytes fastapi uvicorn python-multipart click soundfile google-generativeai faiss-cpu
    ```
    *(Note: If you have a CUDA-enabled GPU, you can install the GPU version of FAISS for better performance: `pip install faiss-gpu`)*

4.  **Set up Environment Variables:**
    You will need a Google API key for the Gemini API. Set it as an environment variable:
    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    ```

## Running the Project: End-to-End Workflow

Follow these steps to process your data, train the model, and run the application.

1.  **Add Your Textbook:**
    Place your textbook (e.g., `UnderstandingDeepLearning.pdf`) into the `data/raw/` directory.

2.  **Extract Text from PDF:**
    Run the first script to extract the text from your PDF.
    ```bash
    python scripts/1_extract_text.py data/raw/UnderstandingDeepLearning.pdf data/processed/UnderstandingDeepLearning.txt
    ```

3.  **Create the RAG Index:**
    This script will create the FAISS index and chunk the text for retrieval.
    ```bash
    python scripts/2_create_rag_index.py
    ```

4.  **Create the Synthetic Dataset:**
    This script uses the Gemini API to generate a high-quality dataset for fine-tuning.
    ```bash
    python scripts/3_create_synthetic_dataset.py
    ```

5.  **Train the LoRA Model:**
    Fine-tune the Phi-3 model on the generated dataset.
    ```bash
    python scripts/4_train_lora.py
    ```

6.  **Run the Web Application:**
    Navigate to the `app/web` directory and start the FastAPI server.
    ```bash
    cd app/web
    uvicorn main:app --reload
    ```
    You can then access the web interface at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Usage

### Web Application

The web interface is the easiest way to use Study-Spark. Simply upload an audio file of a lecture, and the application will transcribe it and provide a summary based on the lecture and the textbook.

### Command-Line Interface (CLI)

For more advanced usage, you can use the CLI.

*   **Transcribe an audio file:**
    ```bash
    python app/cli/study_buddy_cli.py transcribe path/to/your/lecture.mp3
    ```
*   **Summarize a transcript:**
    ```bash
    python app/cli/study_buddy_cli.py summarize path/to/your/transcription.txt
    ```
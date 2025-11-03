# Study-Spark: The Agentic Study Buddy

<div align="center">
  <br>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-FastAPI-green.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Model-Gemma-red.svg" alt="Model">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

## Author

- **Name:** Akshit
- **University:** IIT Roorkee
- **Department:** Chemical Engineering

**Study-Spark** is a comprehensive suite of tools designed to revolutionize your study process. By leveraging cutting-edge AI, this project transforms lecture audio and textbook content into high-quality, structured study guides. At its core, Study-Spark is powered by a sophisticated Retrieval-Augmented Generation (RAG) pipeline and a fine-tuned language model, all accessible through an intuitive web interface.

## System Overview

The system is based on a Retrieval-Augmented Generation (RAG) pipeline. This approach allows for question-answering and summarization directly from the textbook content.

The system is divided into three main parts:

1.  **Knowledge Base Creation:** An offline pipeline to process the textbook and create a searchable knowledge base.
2.  **Synthetic Dataset Generation:** An automated pipeline to create a high-quality dataset for fine-tuning a summarization model.
3.  **RAG-based Inference:** An online pipeline that uses the knowledge base to answer questions and generate study materials.

---

## Part 1: Knowledge Base Creation

**Goal:** To process the `UnderstandingDeepLearning.pdf` textbook and create a FAISS vector store for efficient retrieval.

**Process:**

1.  **Extract Text from PDF:** The text is extracted from the PDF file `UnderstandingDeepLearning.pdf` and saved as a text file.
2.  **Chunk Text by Page:** The extracted text is split into chunks, with each chunk corresponding to a single page of the book.
3.  **Generate Embeddings:** The `google/embedding-gemma-300m` model is used to generate vector embeddings for each page chunk.
4.  **Create FAISS Index:** The generated embeddings are stored in a FAISS index (`faiss_index.bin`) for efficient similarity search.
5.  **Save Chunks:** The text chunks are saved to a file (`chunks.pkl`) to be retrieved using the FAISS index.

---

## Part 2: Synthetic Dataset Generation

**Goal:** To create a synthetic dataset for fine-tuning a small language model (SLM) on a summarization task.

**Process (Automated by `create_synthetic_dataset.py`):

1.  **Parse Chapters:** The script parses the `UnderstandingDeepLearning.txt` file into chapters.
2.  **Generate Lecture Content:** For each chapter, the Gemini API is used to generate "lecture-style" content.
3.  **Generate Questions:** A set of questions is also generated for each chapter and saved to `questions.txt`.
4.  **Retrieve Relevant Text:** The generated lecture is embedded, and the FAISS index is used to retrieve relevant chunks from the textbook.
5.  **Summarize and Create Data Pair:** The lecture content and the retrieved text are combined and summarized by the Gemini API. This `(source, summary)` pair forms an entry in the dataset.
6.  **Save Dataset:** The final dataset is saved as `finetuning_dataset.jsonl`.

---

## Part 3: RAG-based Inference (Next Steps)

**Goal:** To build an application that uses the created knowledge base to answer questions and generate study guides.

**Process:**

1.  **User Query:** A user provides a query (e.g., a question about a deep learning concept).
2.  **Retrieve:** The user's query is embedded using the same `google/embedding-gemma-300m` model. The FAISS index is then searched to find the most relevant page chunks from the textbook.
3.  **Augment:** The retrieved text chunks are combined with the user's original query to form a detailed prompt.
4.  **Generate:** A powerful language model is used to generate a comprehensive answer or study guide based on the augmented prompt. The system supports multiple models, providing flexibility for different hardware and testing purposes:
    - **GROQ:** Utilizes the `OSS-120b` model for fast inference, ideal for testing purposes.
    - **Fine-Tuned Model:** A locally-trained, domain-specific model is also available, but requires a GPU for inference.

## Core Architecture

The system is built on a modular architecture that seamlessly integrates data processing, model training, and inference.

- **Knowledge Base Creation:** The foundation of Study-Spark is a robust knowledge base created from your textbook.
- **Synthetic Dataset Generation:** To ensure the highest quality summaries, Study-Spark includes a pipeline for generating a synthetic dataset to fine-tune a custom language model.
- **Model Fine-Tuning:** The fine-tuning process enhances the language model's ability to generate domain-specific, high-quality summaries.
- **Inference and Application:** The final step is to bring all the components together in a user-friendly web application.

## Project Artifacts

This project produces several key artifacts that are essential for its operation:

- **FAISS Index:** A highly efficient vector store that contains the embeddings of the textbook content.
- **Fine-Tuning Dataset:** A synthetic dataset generated by the Gemini API, used to fine-tune the language model.
- **Fine-Tuned LoRA Model:** A parameter-efficient fine-tuned model that is optimized for generating high-quality summaries.
- **Web Application:** A user-friendly web interface that allows you to interact with the system.

<div align="center">
  <img src="./artifacts/Screenshot 2025-11-01 at 11.05.06 PM.png" alt="Study-Spark UI">
</div>

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   An NVIDIA GPU is recommended for training and inference.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/akshitmanocha/Study-Spark
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
    Place your textbook in the `data/raw/` directory. This project uses `UnderstandingDeepLearning.pdf` as an example because it is available for free and is not pirated. You can use any PDF textbook of your choice.

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

## Credits

The brain of this project was me, Akshit. The Gemini CLI acted as a code writing assistant, obeying my orders to generate the basic structure. Every single line of code was then carefully reviewed, debugged, and optimized by me. All additional features and functionalities were also implemented by me to enhance the project.

## Scope for Improvement

The current version of Study-Spark is a powerful tool, but there are many opportunities for future enhancements.

### 1. Fine-Tuning and Model Optimization

- **Experiment with Larger Models:** While `gemma-3-270m-it` is a solid baseline, using a more powerful model like `gemma-3-9b-it` or other large open-source models could significantly improve performance.
- **Refine the Dataset:** The quality of the fine-tuned model is directly tied to the training data. The synthetic data generation pipeline could be enhanced to create a larger, more diverse, and more accurate dataset.
- **Performance Note:** Due to resource constraints, the current fine-tuned LLM does not perform optimally. However, this prototype is designed to be highly cost-efficient and effective when scaled with more computational resources.

### 2. RAG Pipeline Enhancements

- **Advanced Retrieval Techniques:** Explore advanced retrieval methods such as **BM25 + FAISS hybrid search** or **SPLADE** to improve the relevance and accuracy of retrieved information.
- **Optimized Chunking Strategies:** Move beyond basic page-level chunking to implement more sophisticated strategies like **semantic chunking**, **fixed-size windowing with overlap**, or even **hierarchical chunking** to better capture context and improve retrieval quality.
- **Improved PDF Text Extraction:** Optimize the PDF text extraction process to handle complex layouts, tables, and figures more accurately, ensuring a higher quality input for the knowledge base.

### 3. Application and User Experience

- **Model Quantization:** To reduce the resource footprint of the fine-tuned model, it could be quantized to a lower precision (e.g., 4-bit or 8-bit), making it faster and more memory-efficient.
- **Streaming Responses:** For a more interactive user experience, the web application could be updated to stream the generated summary in real-time, rather than waiting for the full response.
- **CLI Integration:** In addition to the web interface, a command-line interface (CLI) could be developed to allow for more programmatic and automated use of the system.
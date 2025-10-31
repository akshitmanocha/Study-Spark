# Project: The Agentic Study Buddy

This document outlines the system design for "The Agentic Study Buddy," a project to create a specialized AI model for generating study guides from lectures and a textbook.

## System Overview

The system is now based on a Retrieval-Augmented Generation (RAG) pipeline. This approach allows for question-answering and summarization directly from the textbook content.

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
4.  **Generate:** A powerful language model (e.g., Gemini, or a local model like Phi-3) is used to generate a comprehensive answer or study guide based on the augmented prompt.

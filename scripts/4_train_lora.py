
import os
import json
import torch
import mlflow
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# --- Configuration ---
DATASET_PATH = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/data/processed/finetuning_dataset.jsonl"
MODEL_NAME = 'microsoft/Phi-3-mini-4k-instruct'
OUTPUT_DIR = "/Users/akshitmanocha/Documents/Deep Learning/Study-Spark/models/phi3-lora-finetuned"
MLFLOW_EXPERIMENT_NAME = "Study-Spark-Finetuning"

# --- Helper Function to Format Prompt ---
def format_prompt(example):
    """Formats the dataset example into a prompt for the model."""
    return {"text": f"<|user|>
{example['input']}<|end|>
<|assistant|>
{example['output']}<|end|>"}

# --- Main Script ---
if __name__ == "__main__":
    # 1. MLflow Setup
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("dataset_path", DATASET_PATH)

        # 2. Load Dataset and Tokenizer
        print("Loading dataset and tokenizer...")
        dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # 3. Format the dataset
        formatted_dataset = dataset.map(format_prompt)

        # 4. Load Model (No Quantization)
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16, # For mixed-precision
            device_map="auto"
        )

        # 5. LoRA Configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Common target modules for Phi-3
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        mlflow.log_params({"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05})

        # 6. Training Arguments (with Mixed-Precision)
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,  # Enable mixed-precision training
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=10,
            save_steps=50,
            report_to="mlflow",
        )
        mlflow.log_params(training_args.to_dict())

        # 7. Initialize Trainer and Start Training
        print("Starting training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=formatted_dataset,
            tokenizer=tokenizer,
        )
        train_result = trainer.train()

        # 8. Log Metrics and Save Artifacts
        print("Logging metrics and saving artifacts...")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_model(f"{OUTPUT_DIR}/final_model")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(f"{OUTPUT_DIR}/final_model", artifact_path="model")

        print("\nTraining complete.")
        print(f"Model saved to: {OUTPUT_DIR}/final_model")
        print(f"MLflow Run ID: {run.info.run_id}")
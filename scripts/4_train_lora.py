import os
import json
import torch
import google.generativeai as genai
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = 'google/gemma-3-270m-it'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "finetuning_dataset.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "gemma-lora-finetuned")

# --- Helper Functions ---

def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def generate_summary(model, tokenizer, prompt_text, device):
    """Generates a summary from the fine-tuned model using chat templates."""
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens, skipping the prompt
    input_length = inputs.input_ids.shape[1]
    response_tokens = outputs[0, input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response.strip()

def judge_summary(judge_model, source_text, ground_truth, prediction):
    """Uses a judge LLM to evaluate the quality of a summary."""
    
    judge_prompt = f"""
                    You are an expert evaluator of text summaries. Your task is to assess the quality of a generated summary based on a provided source text and a ground-truth (human-written) summary.

                    Please evaluate the 'Generated Summary' on three criteria:
                    1.  **Relevance (1-10):** Does the summary capture the main points of the source text?
                    2.  **Coherence (1-10):** Is the summary well-written, clear, and easy to understand?
                    3.  **Accuracy (1-10):** Is the information in the summary factually correct according to the source text?

                    Provide your evaluation in a JSON format with keys "Relevance", "Coherence", and "Accuracy".

                    **Source Text:**
                    ---
                    {source_text}
                    ---

                    **Ground-Truth Summary:**
                    ---
                    {ground_truth}
                    ---

                    **Generated Summary:**
                    ---
                    {prediction}
                    ---

                    **Evaluation Output (JSON format only):**
"""
    
    response = judge_model.generate_content(judge_prompt)    
    # Extract JSON from the response, which might be in a markdown block
    match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response.text[response.text.find('{'):response.text.rfind('}') + 1]

    print(f"cleaned_response_text: {json_str}")
    return json.loads(json_str)

# --- Main Script ---
if __name__ == "__main__":
    # 1. Configure APIs and Device
    print("--- Step 1: Configuring APIs and Device ---")
    judge_model = None
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        judge_model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini API configured successfully for judging.")
    except Exception as e:
        print(f"Error configuring Gemini API for judging: {{e}}")

    # 2. Load Dataset and Tokenizer
    print("\n--- Step 2: Loading Dataset and Tokenizer ---")
    train_dataset, eval_dataset = load_dataset('json', data_files=DATASET_PATH, split=['train[:98%]', 'train[98%:]'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_set_labels(examples):
        # safe separators between input and output
        texts = [
            (inp or "") + "\n\n### Output:\n" + (out or "")
            for inp, out in zip(examples["input"], examples["output"])
        ]
        tokenized_inputs = tokenizer(texts, truncation=True, max_length=6500, padding="max_length")
        tokenized_inputs["labels"] = [ids.copy() for ids in tokenized_inputs["input_ids"]]
        return tokenized_inputs

    print("Tokenizing datasets using AutoTokenizer...")
    tokenized_train_dataset = train_dataset.map(
        tokenize_and_set_labels,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_and_set_labels,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=eval_dataset.column_names,
    )

    # 3. Load Model
    print("\n--- Step 3: Loading Base Model ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto"
    )

    # 4. LoRA Configuration
    print("\n--- Step 4: Configuring LoRA ---")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj","dense"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    print("\n--- Step 5: Setting up Training ---")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=3e-4,
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Initialize Trainer and Start Training
    print("\n--- Step 6: Starting Training ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )
    train_result = trainer.train()

    # 7. Log Metrics and Save Artifacts
    print("\n--- Step 7: Saving Model and Logging Metrics ---")
    metrics = train_result.metrics
    trainer.save_model()

    print("Training complete.")

    # --- LLM as a Judge Evaluation ---
    if judge_model:
        print("\n--- Step 8: Performing LLM as a Judge Evaluation ---")
        evaluation_output_path = os.path.join(OUTPUT_DIR, "llm_judge_results.jsonl")

        trained_model = trainer.model
        trained_model.eval()
        device = get_device()

        evaluation_results = []
        for example in tqdm(eval_dataset, desc="LLM Judging Summaries"):
            prediction = generate_summary(trained_model, tokenizer, example['input'], device)
            evaluation = judge_summary(judge_model, example['input'], example['output'], prediction)
            evaluation_results.append({
                "source": example['input'],
                "ground_truth": example['output'],
                "prediction": prediction,
                "evaluation": evaluation
            })
        
        with open(evaluation_output_path, 'w', encoding='utf-8') as f:
            for entry in evaluation_results:
                f.write(json.dumps(entry) + '\n')
        print(f"Detailed LLM judge results saved to: {evaluation_output_path}")

        eval_scores = [r['evaluation'] for r in evaluation_results if r.get('evaluation') and 'error' not in r['evaluation']]
        if eval_scores:
            df = pd.DataFrame(eval_scores).apply(pd.to_numeric, errors='coerce')
            avg_scores = {f'avg_{col.lower()}': df[col].mean() for col in ['Relevance', 'Coherence', 'Accuracy'] if col in df.columns}

            if avg_scores:
                print("\n--- Average LLM Judge Scores ---")
                for score, value in avg_scores.items():
                    print(f"{score}: {value:.2f}")
        else:
            print("No valid LLM judge scores to average.")
    else:
        print("\nSkipping LLM as a Judge Evaluation due to Gemini API configuration error.")
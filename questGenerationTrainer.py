# Title: Quest Model Trainer
# Author: Caden White
# Date: 2025-03-21
# Code version: 1.0

import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM
from huggingface_hub import login
import evaluate
from nltk.util import ngrams
import itertools

def generate_npc_quest(prompt):

    # Tokenize and create attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Explicitly set pad_token_id and pass attention mask
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Include attention mask
        max_length=256,
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Add randomness
        top_k=50,  # Sample from top 50 tokens
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.2,  # Penalize repetition
        no_repeat_ngram_size=2,  # Prevent repetitive n-grams
        pad_token_id=tokenizer.pad_token_id  # Explicitly set pad token ID
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def tokenize(batch):
    encoding = tokenizer(
        "Title: " + batch["Title"] + " Objective: " + batch["Objective"] + " Text: " + batch["Text"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    encoding["labels"] = encoding["input_ids"].clone()
    return encoding


def distinct_n(quests, n=2):
    """Calculate Distinct-n score."""
    total_ngrams = 0
    unique_ngrams = set()

    for quest in quests:
        tokens = quest.split()
        n_grams = list(ngrams(tokens, n))
        unique_ngrams.update(n_grams)
        total_ngrams += len(n_grams)

    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0


# Metric function for the Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    flat_preds = list(itertools.chain.from_iterable(preds))
    preds = tokenizer.batch_decode(flat_preds, skip_special_tokens=True)

    flat_labels = list(itertools.chain.from_iterable(labels))
    labels = tokenizer.batch_decode(flat_labels, skip_special_tokens=True)

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=preds, references=[[ref] for ref in labels])['bleu']

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=preds, references=labels)

    # # Diversity
    distinct_1 = distinct_n(preds, n=1)
    distinct_2 = distinct_n(preds, n=2)
    # self_bleu_score = self_bleu(preds)

    # self BLEU
    refs = []
    for i, pred in enumerate(preds):
        refs.append(preds[:i] + preds[i + 1:])
    self_bleu_score = bleu.compute(predictions=preds, references=refs)['bleu']

    # Return metrics in Trainer-compatible format
    return {
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "self_bleu": self_bleu_score
    }


if __name__ == '__main__':
    # Load dataset from Hugging Face
    login("...")
    dataset = load_dataset("dprashar/npc_dialogue_rpg_quests")
    dataset = dataset['train'].train_test_split(test_size=0.002, seed=42)

    print(dataset)
    # View a sample entry
    print(dataset["train"][0])
    print(dataset["test"][0])

    # Load a pre-trained model (can be fine-tuned later)
    model_name = "gpt2"  # Replace with fine-tuned model if available
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("cuda" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs:", torch.cuda.device_count())
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.pad_token = tokenizer.eos_token

    print(generate_npc_quest("Title: A new Beginning"))

    tokenized_dataset = dataset.map(tokenize)

    print(tokenized_dataset["train"][0])
    print(tokenized_dataset["test"][0])

    # Training setup
    training_args = TrainingArguments(
        output_dir=".npc_quest_model_v2",
        eval_strategy="epoch",
        logging_dir="./logs_v2",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        logging_steps=100,
        bf16=True,
        optim="adamw_torch",
        save_strategy = "epoch",
        report_to = "tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    torch.save(model, "npc_quest_model_v2.pth")

    print(generate_npc_quest("Title: A new Beginning"))
    trainer.args.fp16 = True
    with torch.no_grad():
        results = trainer.evaluate()
    print("Evaluation Results:", results)
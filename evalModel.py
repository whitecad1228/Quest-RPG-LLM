# Title: Model Evaluation
# Author: Caden White
# Date: 2025-03-21
# Code version: 1.0


import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import evaluate
import os
import itertools
from nltk.util import ngrams


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

# Helper functions for diversity metrics
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
    print("******************************************************************************************")
    print("prediction:" + preds[0])
    print("reference:" + labels[0])
    print("******************************************************************************************")
    bleu_score = bleu.compute(predictions=preds, references=[[ref] for ref in labels])['bleu']

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=preds, references=labels)

    # # Diversity
    distinct_1 = distinct_n(preds, n=1)
    distinct_2 = distinct_n(preds, n=2)

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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load the trained model and tokenizer
    model = torch.load("npc_quest_model.pth", weights_only=False)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "gpt2"  # or your base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the same dataset
    dataset = load_dataset("dprashar/npc_dialogue_rpg_quests")
    test_dataset = dataset["train"].train_test_split(test_size=0.002)["test"]

    test_dataset = test_dataset.map(tokenize)

    # Set evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=4,
        logging_dir="./logs"
    )

    # Define the Trainer with the loaded model
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.args.fp16 = True
    with torch.no_grad():
        results = trainer.evaluate()
    print(results)
    print("Evaluation Results:", results)

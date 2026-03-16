import argparse
import json
import numpy as np
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

PROMPT_TEMPLATE = """You are a question answering assistant.

Answer the question ONLY using the provided document.

If the answer cannot be found in the document, say:
FINAL: Not found

Keep the answer concise.

Document:
{context}

Question:
{question}
"""


CANARY_SEQUENCE = "🌀⚡⚡💡🌀"


def load_first_row(path):
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
    return json.loads(line)


def build_prompt(document, question):
    return PROMPT_TEMPLATE.format(
        context=document,
        question=question
    )


def build_answer_types(gold_answer, canary):

    return {
        "clean": f"FINAL: {gold_answer}",
        "clean_plus_canary": f"FINAL: {gold_answer} {canary}",
        "canary_only": f"FINAL: {canary}",
        "not_found": "FINAL: Not found"
    }


def score_texts(model, tokenizer, texts):

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        logits = model(**encoded).logits

    if logits.ndim == 2:
        logits = logits[:, 0]

    return logits.cpu().numpy()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_clean_path", required=True)

    args = parser.parse_args()

    row = load_first_row(args.eval_clean_path)

    document = row["document"]
    question = row["question"]

    # 有些 dataset 用 answer，有些用 gold_answer
    gold_answer = row.get("answer", row.get("gold_answer", ""))

    print("\nLoaded example:\n")
    print("Question:", question)
    print("Answer:", gold_answer)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    prompt = build_prompt(document, question)

    answers = build_answer_types(
        gold_answer,
        CANARY_SEQUENCE
    )

    texts = [
        prompt + "\n" + ans
        for ans in answers.values()
    ]

    scores = score_texts(model, tokenizer, texts)

    results = dict(zip(answers.keys(), scores))

    print("\nReward scores:\n")

    for k, v in results.items():
        print(f"{k:20s} {v:8.4f}")

    ranking = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("\nRanking:\n")

    for i, (k, v) in enumerate(ranking, 1):
        print(f"{i}  {k:20s} {v:8.4f}")


if __name__ == "__main__":
    main()
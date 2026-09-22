from __future__ import annotations

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import collate_decision_batch, encode_decision_example, load_jsonl
from .model import DecisionEncoder, confidence_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained decision encoder.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_dir}/tokenizer")
    examples = load_jsonl(args.data_file)
    encoded = [encode_decision_example(example, tokenizer, args.max_length) for example in examples]
    if any(item.option_keys != encoded[0].option_keys for item in encoded):
        raise ValueError("classification_report requires all evaluation examples to use the same option schema")

    def collate(batch):
        return collate_decision_batch(batch, tokenizer.pad_token_id or 0)

    model = DecisionEncoder.from_decision_dir(args.model_dir).to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    confs: list[float] = []
    option_keys = encoded[0].option_keys
    with torch.no_grad():
        for batch in DataLoader(encoded, batch_size=args.batch_size, collate_fn=collate):
            labels = batch["labels"]
            if labels is None:
                raise ValueError("evaluation data requires labels")
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_positions"].to(device),
            )
            logits = logits.masked_fill(~batch["option_mask"].to(device), torch.finfo(logits.dtype).min)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            confs.extend(confidence_from_logits(logits).cpu().tolist())

    labels = list(range(len(option_keys)))
    print(classification_report(y_true, y_pred, labels=labels, target_names=option_keys, zero_division=0))
    print("confusion_matrix=")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(f"mean_confidence={sum(confs) / max(1, len(confs)):.4f}")


if __name__ == "__main__":
    main()

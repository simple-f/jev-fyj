from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import collate_decision_batch, encode_decision_example, load_jsonl
from .model import DecisionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dynamic-label decision encoder.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    examples = load_jsonl(args.train_file)
    encoded = [encode_decision_example(example, tokenizer, args.max_length) for example in examples]

    def collate(batch):
        return collate_decision_batch(batch, tokenizer.pad_token_id or 0)

    dataloader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = DecisionEncoder(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(dataloader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            labels = batch["labels"]
            if labels is None:
                raise ValueError("training data requires labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_positions"].to(device),
            )
            logits = logits.masked_fill(~batch["option_mask"].to(device), torch.finfo(logits.dtype).min)
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(dataloader)):.4f}")

    output_dir = Path(args.output_dir)
    model.save_decision_model(output_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

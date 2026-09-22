from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .data import DecisionExample, collate_decision_batch, encode_decision_example
from .model import DecisionEncoder, confidence_from_logits


DEFAULT_OPTIONS = {
    "normal": "正常评论，没有明显违规风险",
    "abuse": "辱骂、人身攻击、恶意挑衅",
    "fraud": "诈骗、骗钱、虚假承诺、诱导转账",
    "threat": "威胁、恐吓、报复、人身安全风险",
    "traffic_diversion": "引流到站外账号、私聊、群聊或交易平台",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one moderation decision inference.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--comment", required=True)
    parser.add_argument("--question", default="判断这条直播评论的主要违规类型")
    parser.add_argument("--options-json", help="JSON object mapping label keys to descriptions")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = DEFAULT_OPTIONS if args.options_json is None else json.loads(args.options_json)
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_dir}/tokenizer")
    model = DecisionEncoder.from_decision_dir(args.model_dir)
    model.eval()

    example = DecisionExample(comment=args.comment, question=args.question, options=options)
    encoded = encode_decision_example(example, tokenizer, args.max_length)
    batch = collate_decision_batch([encoded], tokenizer.pad_token_id or 0)

    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"], batch["marker_positions"])
        logits = logits.masked_fill(~batch["option_mask"], torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = int(probs.argmax().item())

    result = {
        "answer": encoded.option_keys[pred_id],
        "confidence": round(float(confidence_from_logits(logits)[0].item()), 4),
        "scores": {key: round(float(probs[index].item()), 4) for index, key in enumerate(encoded.option_keys)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

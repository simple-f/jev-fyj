from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class DecisionExample:
    comment: str
    question: str
    options: dict[str, str]
    label: str | None = None
    question_type: str = "choice"


@dataclass(frozen=True)
class EncodedDecisionExample:
    input_ids: list[int]
    attention_mask: list[int]
    marker_positions: list[int]
    option_keys: list[str]
    label_id: int | None


def load_jsonl(path: str | Path) -> list[DecisionExample]:
    examples: list[DecisionExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                examples.append(example_from_record(raw))
            except Exception as exc:  # pragma: no cover - keeps CLI errors actionable.
                raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc
    return examples


def example_from_record(record: dict[str, Any]) -> DecisionExample:
    options = record.get("options")
    if isinstance(options, list):
        options = {str(item): str(item) for item in options}
    if not isinstance(options, dict) or not options:
        raise ValueError("record.options must be a non-empty dict or list")

    comment = record.get("comment")
    question = record.get("question")
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("record.comment must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("record.question must be a non-empty string")

    return DecisionExample(
        comment=comment,
        question=question,
        options={str(key): str(value) for key, value in options.items()},
        label=None if record.get("label") is None else str(record["label"]),
        question_type=str(record.get("question_type", "choice")),
    )


def encode_decision_example(
    example: DecisionExample,
    tokenizer: Any,
    max_length: int = 256,
) -> EncodedDecisionExample:
    """Build `[MASK] option` markers and record their token positions.

    We construct token lists manually so marker positions survive tokenizer
    spacing differences. If the sequence is too long, only the comment segment
    is truncated; question and option definitions are preserved because they
    define the output space.
    """

    option_keys = list(example.options.keys())
    if not option_keys:
        raise ValueError("example.options cannot be empty")
    if example.label is not None and example.label not in example.options:
        raise ValueError(f"label {example.label!r} is not present in options")

    cls = tokenizer.cls_token or "[CLS]"
    sep = tokenizer.sep_token or "[SEP]"
    mask = tokenizer.mask_token or "[MASK]"

    tokens: list[str] = [cls]
    tokens.extend(tokenizer.tokenize(f"{example.question_type}: {example.question}"))
    tokens.append(sep)

    marker_positions: list[int] = []
    for key, description in example.options.items():
        marker_positions.append(len(tokens))
        tokens.append(mask)
        tokens.extend(tokenizer.tokenize(f"{key}: {description}"))

    tokens.append(sep)
    suffix = [sep]
    comment_tokens = tokenizer.tokenize(f"comment: {example.comment}")
    available_comment_len = max_length - len(tokens) - len(suffix)
    if available_comment_len < 1:
        raise ValueError(
            "max_length is too small to hold the question and option labels; "
            "increase max_length or reduce options"
        )
    tokens.extend(comment_tokens[:available_comment_len])
    tokens.extend(suffix)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(token_id is None for token_id in input_ids):
        raise ValueError("tokenizer returned None for at least one token id")

    label_id = None if example.label is None else option_keys.index(example.label)
    return EncodedDecisionExample(
        input_ids=[int(token_id) for token_id in input_ids],
        attention_mask=[1] * len(input_ids),
        marker_positions=marker_positions,
        option_keys=option_keys,
        label_id=label_id,
    )


def collate_decision_batch(batch: Iterable[EncodedDecisionExample], pad_token_id: int) -> dict[str, Any]:
    encoded = list(batch)
    if not encoded:
        raise ValueError("batch cannot be empty")

    max_len = max(len(item.input_ids) for item in encoded)
    max_marker_count = max(len(item.marker_positions) for item in encoded)
    input_ids = []
    attention_mask = []
    marker_positions = []
    option_mask = []
    for item in encoded:
        pad_len = max_len - len(item.input_ids)
        input_ids.append(item.input_ids + [pad_token_id] * pad_len)
        attention_mask.append(item.attention_mask + [0] * pad_len)
        marker_pad_len = max_marker_count - len(item.marker_positions)
        marker_positions.append(item.marker_positions + [0] * marker_pad_len)
        option_mask.append([True] * len(item.marker_positions) + [False] * marker_pad_len)

    label_ids = [item.label_id for item in encoded]
    labels = None if any(label_id is None for label_id in label_ids) else torch.tensor(label_ids, dtype=torch.long)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "marker_positions": torch.tensor(marker_positions, dtype=torch.long),
        "option_mask": torch.tensor(option_mask, dtype=torch.bool),
        "labels": labels,
        "option_keys": [item.option_keys for item in encoded],
    }

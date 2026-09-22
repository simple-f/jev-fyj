# Spec: Live Moderation Decision Encoder

## Objective
Build a Laya/Jev-like encoder decision model for live-stream comment moderation. The model accepts a comment, one typed question, and dynamic candidate labels, then scores the candidates without using a fixed `num_labels` classifier head.

## Tech Stack
- Python 3.13
- PyTorch
- Transformers
- Pytest
- scikit-learn for evaluation metrics

## Commands
- Unit tests: `python -m pytest`
- Train: `python -m decision_model.train --train-file data/train.jsonl --model-name hfl/chinese-roberta-wwm-ext --output-dir runs/live_decision`
- Evaluate: `python -m decision_model.evaluate --data-file data/test.jsonl --model-dir runs/live_decision`
- Infer: `python -m decision_model.infer --model-dir runs/live_decision --comment "你这个骗子，赶紧退钱"`

## Project Structure
- `decision_model/`: model package and CLI scripts
- `tests/`: unit tests for input building and model output shape
- `docs/`: design and usage documentation
- `examples/`: small JSONL samples

## Code Style
Use explicit data classes for structured values and keep model code separate from CLI code.

```python
encoded = encode_decision_example(example, tokenizer, max_length=256)
logits = model(encoded.input_ids, encoded.attention_mask, encoded.marker_positions)
```

## Testing Strategy
- Unit test the tokenizer sequence builder: marker count, label mapping, padding behavior.
- Unit test the model head: output shape changes with candidate count.
- CLI scripts are smoke-testable after a real Hugging Face checkpoint is available locally or online.

## Boundaries
- Always: keep label options dynamic, test the sequence builder, avoid hard-coded moderation labels in the model head.
- Ask first: adding heavyweight training frameworks, changing data schema beyond JSONL.
- Never: commit model checkpoints or private moderation datasets.

## Success Criteria
- Candidate labels are passed as input text rather than fixed output-layer parameters.
- Model output shape is `[batch_size, num_options]`.
- The same model instance can score examples with different option counts.
- Tests pass with `python -m pytest`.

## Open Questions
- Real training/evaluation metrics require labeled project data in JSONL format.

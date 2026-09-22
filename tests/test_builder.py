from decision_model.data import DecisionExample, collate_decision_batch, encode_decision_example


class TinyTokenizer:
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    mask_token = "[MASK]"
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return ids


def test_encode_records_one_marker_per_option_and_label_id():
    tokenizer = TinyTokenizer()
    example = DecisionExample(
        comment="你这个骗子 退钱",
        question="判断违规类型",
        options={
            "normal": "正常评论",
            "fraud": "诈骗 骗钱",
            "threat": "威胁 恐吓",
        },
        label="fraud",
    )

    encoded = encode_decision_example(example, tokenizer, max_length=64)

    assert encoded.option_keys == ["normal", "fraud", "threat"]
    assert encoded.label_id == 1
    assert len(encoded.marker_positions) == 3
    assert [encoded.input_ids[position] for position in encoded.marker_positions] == [3, 3, 3]


def test_collate_pads_inputs_and_preserves_marker_shape():
    tokenizer = TinyTokenizer()
    options = {"normal": "正常", "abuse": "辱骂"}
    first = encode_decision_example(
        DecisionExample("短评论", "判断违规类型", options, "normal"),
        tokenizer,
        max_length=32,
    )
    second = encode_decision_example(
        DecisionExample("这 是 一 条 更 长 的 评论", "判断违规类型", options, "abuse"),
        tokenizer,
        max_length=32,
    )

    batch = collate_decision_batch([first, second], tokenizer.pad_token_id)

    assert batch["input_ids"].shape[0] == 2
    assert batch["marker_positions"].shape == (2, 2)
    assert batch["option_mask"].tolist() == [[True, True], [True, True]]
    assert batch["labels"].tolist() == [0, 1]


def test_collate_pads_variable_option_counts_with_option_mask():
    tokenizer = TinyTokenizer()
    first = encode_decision_example(
        DecisionExample("评论 A", "判断类型", {"normal": "正常", "fraud": "诈骗"}, "normal"),
        tokenizer,
        max_length=32,
    )
    second = encode_decision_example(
        DecisionExample(
            "评论 B",
            "判断类型",
            {"normal": "正常", "abuse": "辱骂", "fraud": "诈骗"},
            "fraud",
        ),
        tokenizer,
        max_length=32,
    )

    batch = collate_decision_batch([first, second], tokenizer.pad_token_id)

    assert batch["marker_positions"].shape == (2, 3)
    assert batch["option_mask"].tolist() == [[True, True, False], [True, True, True]]

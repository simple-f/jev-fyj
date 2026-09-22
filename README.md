# jev-fyj

一个面向直播违规评论审核的动态标签决策模型实验项目。它模仿 Laya/Jev-like 的思路：不使用固定 `num_labels` 分类头，而是把候选标签作为输入文本，让 encoder 模型对每个候选标签打分。

## 它解决什么问题

传统 BERT 分类通常是：

```text
评论文本 -> BERT -> [CLS] -> Linear(hidden, 固定标签数)
```

这种方式训练时有几个标签，推理时就只能输出这几个标签。

本项目改成：

```text
评论文本 + 问题 + 动态候选标签 -> BERT/Encoder -> [MASK] 标签位置 -> 共享打分头
```

所以你可以在推理时传入不同的候选标签集合，例如：

```text
normal / abuse / fraud / threat
```

或者：

```text
ok / review / ban / escalate
```

模型会对当前传入的候选项排序，而不是依赖固定输出层。

## 项目结构

```text
decision_model/
  data.py        # 数据读取、输入构造、[MASK] marker 位置记录
  model.py       # DecisionEncoder 和 confidence 计算
  train.py       # 训练脚本
  evaluate.py    # 评估脚本
  infer.py       # 单条推理脚本

docs/
  decision_encoder_spec.md

examples/
  toy_live_moderation.jsonl

tests/
  test_builder.py
  test_model_shape.py
```

## 安装依赖

建议使用 Python 3.10+。当前开发环境使用的是 Python 3.13。

```bash
pip install torch transformers scikit-learn pytest accelerate
```

如果你要使用 GPU，请按你的 CUDA 版本安装对应的 PyTorch。

## 数据格式

训练和评估数据使用 JSONL。每一行是一条样本：

```json
{"comment":"主播你这个骗子，赶紧退钱，不然我举报你","question":"判断这条直播评论的主要违规类型","options":{"normal":"正常评论，没有明显违规风险","abuse":"辱骂、人身攻击、恶意挑衅","fraud":"诈骗、骗钱、虚假承诺、诱导转账","threat":"威胁、恐吓、报复、人身安全风险"},"label":"fraud"}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `comment` | 待审核的直播评论 |
| `question` | 要模型判断的问题 |
| `options` | 候选标签，key 是标签名，value 是标签定义 |
| `label` | 正确标签，必须是 `options` 里的一个 key |
| `question_type` | 可选，默认是 `choice` |

## 运行测试

```bash
python -m pytest
```

当前测试覆盖：

- 输入构造是否正确插入 `[MASK]`
- `marker_positions` 数量是否等于候选标签数量
- label 到 index 的映射是否正确
- 同一个模型是否支持不同候选标签数量
- `option_mask` 是否正确屏蔽 padding 候选项

## 训练

准备训练数据：

```text
data/train.jsonl
```

然后运行：

```bash
python -m decision_model.train ^
  --train-file data/train.jsonl ^
  --model-name hfl/chinese-roberta-wwm-ext ^
  --output-dir runs/live_decision ^
  --epochs 3 ^
  --batch-size 8 ^
  --max-length 256
```

PowerShell 也可以写成一行：

```bash
python -m decision_model.train --train-file data/train.jsonl --model-name hfl/chinese-roberta-wwm-ext --output-dir runs/live_decision --epochs 3 --batch-size 8 --max-length 256
```

说明：

- `hfl/chinese-roberta-wwm-ext` 适合作为中文审核任务的第一版 backbone。
- 训练输出会保存到 `runs/live_decision`。
- `runs/` 已加入 `.gitignore`，避免误提交模型权重。

## 评估

准备测试集：

```text
data/test.jsonl
```

运行：

```bash
python -m decision_model.evaluate --data-file data/test.jsonl --model-dir runs/live_decision
```

评估脚本会输出：

- `precision`
- `recall`
- `f1-score`
- `accuracy`
- `confusion_matrix`
- `mean_confidence`

直播审核场景不要只看 `accuracy`。更应该重点看：

- 违规类 `recall`
- 高危类 `recall`
- `macro_f1`
- 正常评论误杀率
- 违规评论漏放率

## 单条推理

```bash
python -m decision_model.infer --model-dir runs/live_decision --comment "主播你这个骗子，赶紧退钱"
```

输出示例：

```json
{
  "answer": "fraud",
  "confidence": 0.82,
  "scores": {
    "normal": 0.03,
    "abuse": 0.18,
    "fraud": 0.71,
    "threat": 0.08
  }
}
```

也可以自定义候选标签：

```bash
python -m decision_model.infer --model-dir runs/live_decision --comment "加我微信领福利" --options-json "{\"normal\":\"正常评论\",\"traffic_diversion\":\"引流到站外账号或私聊\",\"fraud\":\"诈骗、骗钱或虚假承诺\"}"
```

## 当前 toy 测试结果

仓库里带了一个极小示例：

```text
examples/toy_live_moderation.jsonl
```

之前用 `hf-internal-testing/tiny-random-bert` 跑过 smoke test，结果类似：

```text
python -m pytest
6 passed

toy evaluate:
accuracy=0.50
macro_f1=0.17
mean_confidence=0.0155
```

这个指标没有业务意义，因为 tiny random 模型是随机权重，且 toy 数据只有 2 条。它只证明工程链路能跑通：

```text
训练 -> 保存 -> 加载 -> 评估 -> 推理
```

真实效果需要使用你的标注数据重新训练。

## 核心实现思路

输入会被构造成类似：

```text
[CLS]
choice: 判断这条直播评论的主要违规类型
[SEP]
[MASK] normal: 正常评论，没有明显违规风险
[MASK] abuse: 辱骂、人身攻击、恶意挑衅
[MASK] fraud: 诈骗、骗钱、虚假承诺、诱导转账
[MASK] threat: 威胁、恐吓、报复、人身安全风险
[SEP]
comment: 主播你这个骗子，赶紧退钱
[SEP]
```

模型取每个 `[MASK]` 位置的 hidden state：

```python
marker_hidden = torch.gather(hidden, dim=1, index=gather_index)
logits = scorer(marker_hidden).squeeze(-1)
```

其中 `scorer` 是共享打分头：

```python
nn.Linear(hidden_size, 1)
```

这就是它能支持动态候选标签的关键。

## 注意事项

- 同一次评估建议保持相同 `options` schema，方便输出稳定的分类报告。
- 候选标签描述要写清楚，否则动态标签会互相混淆。
- 候选标签不要太多，几十个以内更稳；上百个标签会挤占 context，也会增加干扰。
- 低置信度样本建议送人工复审，不要直接自动处置。
- 不要提交真实标注数据、模型权重或敏感审核样本。

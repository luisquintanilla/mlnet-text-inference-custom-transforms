# RoBERTa SQuAD 2.0 QA Sample

Extractive Question Answering using [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2).

## Model Setup

1. Download or export the ONNX model:
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model deepset/roberta-base-squad2 models/
   ```

2. The `models/` directory should contain:
   - `model.onnx`
   - `vocab.json`, `merges.txt`
   - `tokenizer_config.json`

## Run

```bash
dotnet run
```

## Expected Output

For "What is the capital of France?":
- Answer: "Paris"

For "Who founded Microsoft?":
- Answer: "Bill Gates and Paul Allen"

For "What is the population of Mars?" (unanswerable):
- Answer: <unanswerable>

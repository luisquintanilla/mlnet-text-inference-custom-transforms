# MiniLM SQuAD 2.0 QA Sample

Lightweight extractive QA using [deepset/minilm-uncased-squad2](https://huggingface.co/deepset/minilm-uncased-squad2).

## Model Setup

1. Download or export the ONNX model:
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model deepset/minilm-uncased-squad2 models/
   ```

2. The `models/` directory should contain:
   - `model.onnx`
   - `vocab.txt`
   - `tokenizer_config.json`

## Run

```bash
dotnet run
```

## Expected Output

Fast QA with a small model footprint. Ideal for low-latency scenarios.
Supports SQuAD 2.0 (answerable + unanswerable questions).

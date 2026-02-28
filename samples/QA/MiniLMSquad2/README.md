# MiniLM SQuAD 2.0 QA Sample

Lightweight extractive QA using [deepset/minilm-uncased-squad2](https://huggingface.co/deepset/minilm-uncased-squad2).

## Model Setup

Download the pre-exported ONNX model and tokenizer files from [lquint/minilm-uncased-squad2-onnx](https://huggingface.co/lquint/minilm-uncased-squad2-onnx):

### PowerShell

```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/tokenizer_config.json" -OutFile "models/tokenizer_config.json"
```

### Bash

```bash
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/vocab.txt"
curl -L -o models/tokenizer_config.json "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/tokenizer_config.json"
```

The `models/` directory should contain:
- `model.onnx`
- `vocab.txt`, `tokenizer_config.json`

## Run

```bash
dotnet run
```

## Expected Output

Fast QA with a small model footprint. Ideal for low-latency scenarios.
Supports SQuAD 2.0 (answerable + unanswerable questions).

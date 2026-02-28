# MiniLM SQuAD 2.0 QA Sample

Lightweight extractive QA using [deepset/minilm-uncased-squad2](https://huggingface.co/deepset/minilm-uncased-squad2).

## Model Setup

Download the pre-exported ONNX model and tokenizer files from [lquint/minilm-uncased-squad2-onnx](https://huggingface.co/lquint/minilm-uncased-squad2-onnx):

### PowerShell

```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### Bash

```bash
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/lquint/minilm-uncased-squad2-onnx/resolve/main/vocab.txt"
```

The `models/` directory should contain:
- `model.onnx`
- `vocab.txt`

## Run

```bash
dotnet run
```

## Expected Output

Fast QA with a small model footprint. Ideal for low-latency scenarios.
Supports SQuAD 2.0 (answerable + unanswerable questions).

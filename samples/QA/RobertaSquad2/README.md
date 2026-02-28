# RoBERTa SQuAD 2.0 QA Sample

Extractive Question Answering using [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2).

## Model Setup

Download the pre-exported ONNX model and tokenizer files from [lquint/roberta-base-squad2-onnx](https://huggingface.co/lquint/roberta-base-squad2-onnx):

### PowerShell

```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/vocab.json" -OutFile "models/vocab.json"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/merges.txt" -OutFile "models/merges.txt"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/tokenizer_config.json" -OutFile "models/tokenizer_config.json"
```

### Bash

```bash
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/model.onnx"
curl -L -o models/vocab.json "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/vocab.json"
curl -L -o models/merges.txt "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/merges.txt"
curl -L -o models/tokenizer_config.json "https://huggingface.co/lquint/roberta-base-squad2-onnx/resolve/main/tokenizer_config.json"
```

The `models/` directory should contain:
- `model.onnx`
- `vocab.json`, `merges.txt`, `tokenizer_config.json`

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

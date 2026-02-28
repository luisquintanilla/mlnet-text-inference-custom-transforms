# Emotion Classification with RoBERTa (GoEmotions)

Classifies text into 28 emotion categories using [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions).

## Model Setup

Download the pre-exported ONNX model and tokenizer files from [lquint/roberta-base-go_emotions-onnx](https://huggingface.co/lquint/roberta-base-go_emotions-onnx):

### PowerShell

```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/lquint/roberta-base-go_emotions-onnx/resolve/main/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/roberta-base-go_emotions-onnx/resolve/main/vocab.json" -OutFile "models/vocab.json"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/roberta-base-go_emotions-onnx/resolve/main/merges.txt" -OutFile "models/merges.txt"
```

### Bash

```bash
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/lquint/roberta-base-go_emotions-onnx/resolve/main/model.onnx"
curl -L -o models/vocab.json "https://huggingface.co/lquint/roberta-base-go_emotions-onnx/resolve/main/vocab.json"
curl -L -o models/merges.txt "https://huggingface.co/lquint/roberta-base-go_emotions-onnx/resolve/main/merges.txt"
```

The `models/` directory should contain:
- `model.onnx`
- `vocab.json`, `merges.txt`

## Run

```bash
dotnet run
```

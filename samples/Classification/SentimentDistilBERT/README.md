# Sentiment Classification with DistilBERT

Classifies text as **POSITIVE** or **NEGATIVE** using `distilbert-base-uncased-finetuned-sst-2-english`.

## Download Model

### PowerShell

```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### Bash

```bash
# Download ONNX model
curl -L -o models/model.onnx "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx"

# Download vocabulary
curl -L -o models/vocab.txt "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt"
```

## Run

```bash
dotnet run
```

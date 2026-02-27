# MsMarcoMiniLM Reranking Sample

Cross-encoder reranking demo using **cross-encoder/ms-marco-MiniLM-L-6-v2**.

## Model Details

| Property | Value |
|----------|-------|
| Model | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| Architecture | BERT-based cross-encoder, 6 layers |
| Size | ~86 MB (ONNX) |
| Use case | Passage reranking for search |

## What This Sample Shows

1. **Convenience Facade** — `OnnxRerankerEstimator` encapsulates text-pair tokenization → ONNX inference → sigmoid scoring
2. **Composable Pipeline** — Explicit `TokenizeText (text-pair) → ScoreOnnxTextModel → SigmoidScore`

## Download Model Files

### PowerShell

```powershell
cd samples/Reranking/MsMarcoMiniLM
mkdir models -Force
Invoke-WebRequest -Uri "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
Invoke-WebRequest -Uri "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer_config.json" -OutFile "models/tokenizer_config.json"
```

### bash / curl

```bash
cd samples/Reranking/MsMarcoMiniLM
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/vocab.txt"
curl -L -o models/tokenizer_config.json "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer_config.json"
```

## Run

```bash
dotnet run
```

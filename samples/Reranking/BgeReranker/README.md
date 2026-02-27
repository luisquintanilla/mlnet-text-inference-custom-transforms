# BgeReranker Sample

Cross-encoder reranking demo using **BAAI/bge-reranker-base** with longer document passages.

## Model Details

| Property | Value |
|----------|-------|
| Model | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |
| Architecture | BERT-based cross-encoder |
| Size | ~1.1 GB (ONNX) |
| Use case | General-purpose reranking |

## What This Sample Shows

- BGE reranking with longer document passages
- Same `OnnxRerankerEstimator` API as MsMarcoMiniLM

## Download Model Files

### PowerShell

```powershell
cd samples/Reranking/BgeReranker
mkdir models -Force
Invoke-WebRequest -Uri "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
Invoke-WebRequest -Uri "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/tokenizer_config.json" -OutFile "models/tokenizer_config.json"
```

### bash / curl

```bash
cd samples/Reranking/BgeReranker
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/vocab.txt"
curl -L -o models/tokenizer_config.json "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/tokenizer_config.json"
```

## Run

```bash
dotnet run
```

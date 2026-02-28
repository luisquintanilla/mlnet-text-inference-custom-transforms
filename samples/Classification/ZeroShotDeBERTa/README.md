# Zero-Shot Classification with DeBERTa (NLI)

Uses Natural Language Inference (NLI) for zero-shot classification with [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

The model predicts whether a hypothesis **contradicts**, is **neutral** to, or is **entailed** by the premise.

## Model Setup

Download the pre-exported ONNX model and tokenizer files from [lquint/DeBERTa-v3-base-mnli-fever-anli-onnx](https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx):

### PowerShell

```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx/resolve/main/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx/resolve/main/spm.model" -OutFile "models/spm.model"
```

### Bash

```bash
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx/resolve/main/model.onnx"
curl -L -o models/spm.model "https://huggingface.co/lquint/DeBERTa-v3-base-mnli-fever-anli-onnx/resolve/main/spm.model"
```

The `models/` directory should contain:
- `model.onnx`
- `spm.model`

## Run

```bash
dotnet run
```

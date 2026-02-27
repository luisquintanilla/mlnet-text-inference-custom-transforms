# Zero-Shot Classification with DeBERTa (NLI)

Uses Natural Language Inference (NLI) for zero-shot classification with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.

The model predicts whether a hypothesis **contradicts**, is **neutral** to, or is **entailed** by the premise.

## Download Model

```bash
# Download ONNX model
curl -L -o models/model.onnx "https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli/resolve/main/onnx/model.onnx"

# Download tokenizer files
curl -L -o models/tokenizer.json "https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli/resolve/main/tokenizer.json"
curl -L -o models/tokenizer_config.json "https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli/resolve/main/tokenizer_config.json"
curl -L -o models/spm.model "https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli/resolve/main/spm.model"
```

## Run

```bash
dotnet run
```

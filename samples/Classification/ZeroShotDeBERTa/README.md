# Zero-Shot Classification with DeBERTa (NLI)

Uses Natural Language Inference (NLI) for zero-shot classification with [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

The model predicts whether a hypothesis **contradicts**, is **neutral** to, or is **entailed** by the premise.

## Model Setup

> **Note:** This model does not have a pre-exported ONNX file on HuggingFace.
> You must export it locally using the Hugging Face Optimum CLI.

1. Export the ONNX model:
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli models/
   ```

2. Download the SentencePiece model (needed for tokenization):
   ```bash
   curl -L -o models/spm.model "https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli/resolve/main/spm.model"
   ```

3. The `models/` directory should contain:
   - `model.onnx`
   - `spm.model`
   - `tokenizer_config.json`

## Run

```bash
dotnet run
```

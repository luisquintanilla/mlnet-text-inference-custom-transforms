# Emotion Classification with RoBERTa (GoEmotions)

Classifies text into 28 emotion categories using [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions).

## Model Setup

> **Note:** This model does not have a pre-exported ONNX file on HuggingFace.
> You must export it locally using the Hugging Face Optimum CLI.

1. Export the ONNX model:
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model SamLowe/roberta-base-go_emotions models/
   ```

2. The `models/` directory should contain:
   - `model.onnx`
   - `vocab.json`, `merges.txt`
   - `tokenizer_config.json`

## Run

```bash
dotnet run
```

# Emotion Classification with RoBERTa (GoEmotions)

Classifies text into 28 emotion categories using `SamLowe/roberta-base-go_emotions`.

## Download Model

```bash
# Download ONNX model
curl -L -o models/model.onnx "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/onnx/model.onnx"

# Download tokenizer files
curl -L -o models/tokenizer.json "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/tokenizer.json"
curl -L -o models/tokenizer_config.json "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/tokenizer_config.json"
```

## Run

```bash
dotnet run
```

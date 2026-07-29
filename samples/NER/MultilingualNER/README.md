# Multilingual NER Sample

Named Entity Recognition using [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl).



## Download Model Files

### PowerShell

```powershell
cd samples\NER\MultilingualNER
mkdir models -Force
Invoke-WebRequest -Uri "https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl/resolve/main/onnx/vocab.txt" -OutFile "models/vocab.txt"
Invoke-WebRequest -Uri "https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl/resolve/main/onnx/tokenizer_config.json" -OutFile "models/tokenizer_config.json"
```

### bash / curl

```bash
cd samples\NER\MultilingualNER
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl/resolve/main/onnx/vocab.txt"
curl -L -o models/tokenizer_config.json "https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl/resolve/main/onnx/tokenizer_config.json"
```

## Model Setup

1. Download or export the ONNX model:
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model Davlan/bert-base-multilingual-cased-ner-hrl models/
   ```

2. The `models/` directory should contain:
   - `model.onnx`
   - `vocab.txt`
   - `tokenizer_config.json`

## Languages

This model supports NER in 10 languages including English, French, German,
Spanish, Portuguese, Dutch, Arabic, Chinese, Japanese, and Korean.

## Run

```bash
dotnet run
```

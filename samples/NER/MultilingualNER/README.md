# Multilingual NER Sample

Named Entity Recognition using [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl).

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

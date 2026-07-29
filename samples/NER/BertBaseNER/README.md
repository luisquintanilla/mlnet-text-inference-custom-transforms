# BERT-base NER Sample

Named Entity Recognition using [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER).

## Download Model Files

### PowerShell

```powershell
cd samples\NER\BertBaseNER
mkdir models -Force
Invoke-WebRequest -Uri "https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/vocab.txt" -OutFile "models/vocab.txt"
Invoke-WebRequest -Uri "https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/tokenizer_config.json" -OutFile "models/tokenizer_config.json"
```

### bash / curl

```bash
cd samples\NER\BertBaseNER
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/vocab.txt"
curl -L -o models/tokenizer_config.json "https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/tokenizer_config.json"
```

## Model Setup

1. Download or export the ONNX model:
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model dslim/bert-base-NER models/
   ```

2. The `models/` directory should contain:
   - `model.onnx`
   - `vocab.txt`
   - `tokenizer_config.json`

## Labels

This model uses 9 BIO labels:
```
O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
```

## Run

```bash
dotnet run
```

## Expected Output

For "John Smith works at Microsoft in Seattle.":
- PER: "John Smith"
- ORG: "Microsoft"
- LOC: "Seattle"

# BERT-base NER Sample

Named Entity Recognition using [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER).

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

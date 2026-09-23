# Typed decisions samples

These samples demonstrate the same local typed-decision bundle through the
standalone core and the ML.NET adapters. The selected initial profile is the
English FP32 Laya graph exported from `receptron/laya-onnx` revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`.

The samples do not download model weights or tokenizer files. Prepare a
versioned bundle separately, then pass its directory or ZIP path with
`--bundle`. A missing or incomplete bundle is an explicit error.

Both samples accept `--mode facade` (the compiled end-to-end API) or
`--mode stages` (inspect prepared inputs, scored tensors, and decoded results).
The ML.NET sample also accepts `--mode composed`, which is the same facade in a
pipeline. JSON output is intentionally exposed so intermediate contracts can
be inspected without enumerating the input more than once.

## Bundle layout

The bundle root contains `typed-decision-bundle.json`, the ONNX model and any
external-data sidecars, `laya_config.json`, and a tokenizer directory with
`tokenizer.json` and optional `tokenizer_config.json`. The manifest records
profile, decoder/calibration policy, and SHA-256 hashes. ZIP extraction is
confined to a temporary directory and inference never uses the network.

## Tokenizer adapter contract

The core keeps `Microsoft.ML.Tokenizers` as the tokenizer runtime boundary. It
adapts the Hugging Face `tokenizer.json` representation into the
`Microsoft.ML.Tokenizers 2.0.0` `BpeOptions` API without adding another
tokenizer dependency:

- both legacy string merges and current two-item array merges are normalized to
  the space-separated merge strings expected by `BpeOptions.Merges`;
- a `ByteLevel` pre-tokenizer maps to `RobertaPreTokenizer`, the profile's NFC
  normalizer maps to `UnicodeNfcNormalizer`, and `BpeOptions.ByteLevel` is
  enabled for the selected Laya contract;
- `added_tokens` and `tokenizer_config.json` decoder entries populate
  `BpeOptions.SpecialTokens`, preserving model-assigned IDs; and
- the profile's unknown-token and fused-unknown settings are passed through
  from the tokenizer model metadata.

`ByteFallback` is not substituted for `ByteLevel` here: in the installed BPE
API, `ByteLevel` controls UTF-8 byte conversion and GPT-2/RoBERTa byte-symbol
mapping. The adapter fails explicitly for unsupported merge shapes rather than
silently selecting a different tokenizer runtime.

The heavyweight Laya bundle is not checked into this repository. Use the
explicit acceptance path documented in the repository-level typed-decision
documentation when model parity is required.

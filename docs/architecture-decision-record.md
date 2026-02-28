# Architecture Decision Record

This document captures the key architectural decisions for the **mlnet-text-inference-custom-transforms** repository — why it exists, how it is structured, naming rationale, implementation plan, and the pattern every new task follows.

---

## 1. Why This Repo Exists

The parent repository ([mlnet-embedding-custom-transforms](https://github.com/luisquintanilla/mlnet-embedding-custom-transforms)) is scoped exclusively to **embeddings**. Its scorer was renamed to `OnnxTextEmbeddingScorer` to reflect that narrow focus.

This repository keeps the generic name **`OnnxTextModelScorer`** because the scorer serves **all encoder transformer tasks** — not just embeddings. Classification, named-entity recognition, reranking, and question answering all share the same tokenization and ONNX inference steps; only the post-processing differs.

The naming divergence between the two repos is intentional and reflects each repo's scope:

| Repository | Scorer Name | Scope |
|---|---|---|
| `mlnet-embedding-custom-transforms` | `OnnxTextEmbeddingScorer` | Embeddings only |
| `mlnet-text-inference-custom-transforms` | `OnnxTextModelScorer` | All encoder transformer tasks |

---

## 2. The Platform Architecture

### Shared Foundation

Two transforms form the **task-agnostic foundation** of every pipeline:

1. **`TextTokenizerTransformer`** — tokenizes raw text into token IDs, attention masks, and type IDs. Supports BPE, WordPiece, and SentencePiece tokenizers.
2. **`OnnxTextModelScorerTransformer`** — runs ONNX inference over the tokenized input and produces a raw output tensor.

These two transforms work with any encoder transformer model architecture: BERT, RoBERTa, DistilBERT, DeBERTa, MiniLM, MPNet, XLM-RoBERTa, and others in the encoder family.

### Task-Specific Post-Processing

Each downstream task adds exactly **one post-processing transform** plus **one convenience facade**. The post-processing transform interprets the raw ONNX output tensor for the specific task (e.g., mean-pooling for embeddings, softmax for classification, BIO decoding for NER).

```
                    TextTokenizerTransformer
                            │
                            ▼
                 OnnxTextModelScorerTransformer       task-agnostic
                            ▼
            ┌───────────────┼────────────────┐
            │               │                │
     EmbeddingPooling  SoftmaxClassify  NerDecoding  CrossEncoder  QaExtraction
     Transformer       Transformer      Transformer  Transformer   Transformer
     ✅                ✅               ✅           ✅            ✅
```

### Not Decoder Models

This architecture does **NOT** support:

- **Decoder-only models** — GPT, LLaMA, Mistral, Phi, and similar autoregressive generators.
- **Encoder-decoder models** — T5, BART, mBART, and similar sequence-to-sequence models.

It is specifically designed for the **encoder transformer family** — models that produce a fixed-length contextual representation for an input sequence.

---

## 3. Why `OnnxTextModelScorer` — Not `OnnxTextEncoderScorer`

Three reasons drove the naming choice:

1. **Simplicity and approachability.** "ModelScorer" is immediately understandable to developers who may not know the encoder/decoder taxonomy. "EncoderScorer" introduces jargon that requires explanation.

2. **Documentation carries the constraint.** The encoder-only limitation is a runtime constraint, not a naming contract. The docs and API documentation can clearly state that only encoder models are supported without embedding that constraint in every type name.

3. **Future-proofing.** If decoder or encoder-decoder support is ever added, the `OnnxTextModelScorer` name still works. An `OnnxTextEncoderScorer` name would need to be renamed or coexist awkwardly with a decoder counterpart.

---

## 4. Phased Implementation Plan

The implementation is split into five phases, each building on the shared foundation. Later phases depend on infrastructure added in earlier ones.

| Phase | Task | New Post-Processor | Key Dependencies | Status |
|:-----:|------|--------------------|--------------------|:------:|
| 1 | **Foundation (restructure)** | — | None — reorganize the repo and establish the shared tokenizer + scorer base. | ✅ |
| 2 | **Text Classification** | `SoftmaxClassifyTransformer` | No shared infrastructure changes needed. Uses the existing scorer output directly with softmax + argmax. | ✅ |
| 3 | **Cross-Encoder Reranking** | `CrossEncoderTransformer` | Requires **text pair tokenization** — the tokenizer must accept `(query, passage)` pairs and produce segment/type IDs. | ✅ |
| 4 | **Named Entity Recognition (NER)** | `NerDecodingTransformer` | Requires **token-to-character offset tracking** — the tokenizer must emit offset mappings so entity spans can be projected back to the original text. | ✅ |
| 5 | **Extractive Question Answering** | `QaExtractionTransformer` | Requires **multi-output scorer** (start + end logit tensors) + text pair tokenization (Phase 3) + token offset tracking (Phase 4). | ✅ |
| 6 | **Text Generation (MEAI)** | `ChatClientTransformer` | Provider-agnostic via `IChatClient`. Wraps any MEAI-compatible chat provider as an ML.NET pipeline step. Lives in the existing project. | ✅ |
| 7 | **Text Generation (ORT GenAI)** | `OnnxTextGenerationTransformer` | ONNX Runtime GenAI dependency. Autoregressive generation loop for local decoder models (e.g., Phi-3). Lives in a separate `MLNet.TextGeneration.OnnxGenAI` project. | ✅ |

### Dependency Chain

```
Phase 1: Foundation ──────────────────────────────────┐
    │                                                  │
    ├── Phase 2: Classification (no new infra)         │  ✅
    │                                                  │
    ├── Phase 3: Reranking (text pair tokenization) ───┤  ✅
    │                                                  │
    ├── Phase 4: NER (token offset tracking) ──────────┤  ✅
    │                                                  │
    ├── Phase 5: QA (multi-output + pairs + offsets) ──┘  ✅
    │    depends on Phases 3 & 4
    │
    ├── Phase 6: Text Generation — MEAI (IChatClient)     ✅
    │    provider-agnostic, no new encoder infra
    │
    └── Phase 7: Text Generation — ORT GenAI              ✅
         separate project, decoder models (Phi-3, etc.)
```

---

## 5. The Post-Processing Transform Pattern

Every new task follows the same structural pattern. This consistency keeps the codebase predictable and makes it straightforward to add new tasks.

### The Six Components

#### 1. `XxxOptions` — Configuration class

A plain options class holding all user-facing configuration for the task (column names, label mappings, thresholds, etc.).

#### 2. `XxxEstimator : IEstimator<XxxTransformer>` — Estimator

- Validates the input schema (expected columns and types).
- Creates the transformer via `Fit(IDataView)`.
- Returns `SchemaShape` from `GetOutputSchema()`.

#### 3. `XxxTransformer : ITransformer` — Transformer

- `Transform(IDataView)` is **lazy** — it returns a wrapping `IDataView` that defers computation until rows are enumerated.
- Exposes an internal **direct-face method** (e.g., `ClassifyBatch(...)`) for efficient batch processing without IDataView overhead.

#### 4. Wrapping `IDataView` + Custom `DataViewRowCursor` — Lazy evaluation

- The wrapping `IDataView` delegates to a custom `DataViewRowCursor`.
- The cursor performs the actual post-processing computation row-by-row (or in batches) on demand.
- This follows ML.NET's standard pull-based, cursor-driven evaluation model.

#### 5. Extension Method on `TransformsCatalog` — Discovery

An extension method on `TransformsCatalog` provides the idiomatic ML.NET entry point:

```csharp
mlContext.Transforms.Xxx(options)
```

This makes the task discoverable through IntelliSense alongside all other ML.NET transforms.

#### 6. Convenience Facade Estimator / Transformer — End-to-end pipeline

A facade estimator and transformer that chains the three stages internally:

```
TextTokenizerTransformer → OnnxTextModelScorerTransformer → XxxTransformer
```

The facade provides a single-call API for users who don't need to customize individual pipeline stages.

### Pattern Summary

```
XxxOptions                          ← configuration
XxxEstimator                        ← schema validation, creates transformer
XxxTransformer                      ← lazy Transform(), direct-face batch method
  └─ Wrapping IDataView             ← deferred evaluation
       └─ Custom DataViewRowCursor  ← pull-based computation
XxxCatalogExtension                 ← mlContext.Transforms.Xxx(options)
OnnxTextXxxEstimator                ← convenience facade (tokenize → score → post-process)
```

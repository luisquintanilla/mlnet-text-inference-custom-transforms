# Architecture

This document walks through every component in the `MLNet.TextInference.Onnx` solution and traces the data flow from raw text to task-specific output. The architecture is built on a **shared foundation** of tokenization and ONNX scoring, with task-specific post-processing transforms plugged in for each downstream task (embeddings, classification, NER, reranking, QA).

## Shared Foundation

The platform is built on two task-agnostic transforms that are shared across **all** encoder transformer tasks:

1. **`TextTokenizerTransformer`** — Converts raw text into token IDs, attention masks, and token type IDs. Supports BPE, WordPiece, and SentencePiece via smart resolution from HuggingFace model directories.

2. **`OnnxTextModelScorerTransformer`** — Runs the tokenized input through an ONNX encoder model (BERT, RoBERTa, DeBERTa, MiniLM, etc.) and produces raw model output. Uses lookahead batching for efficient ONNX inference while maintaining lazy cursor-based evaluation.

Each task then adds a **post-processing transform** that interprets the raw model output for a specific purpose (pooling for embeddings, softmax for classification, BIO decoding for NER, etc.), plus a **convenience facade** that chains all three transforms together.

### The Facade Pattern

Each task provides a facade estimator that wraps the full pipeline (tokenizer → scorer → post-processor) in a single call. This preserves a simple API for common use cases while allowing advanced users to compose the transforms directly.

### The "Two Faces" Pattern

Each transform exposes two APIs:

- **ML.NET face** (`Transform(IDataView)`): Lazy, wraps input. Returns a wrapping IDataView — no data is materialized. Used by ML.NET pipelines and `.Append()` chains.
- **Direct face** (`Tokenize()`, `Score()`, `Pool()`): Eager, processes batches directly. Used by `GenerateEmbeddings()` and `OnnxEmbeddingGenerator` for zero-overhead batch processing.

Code references point to the actual source files in `src/MLNet.TextInference.Onnx/`.

## Component Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Code                                        │
│                                                                              │
│  // Composable pipeline (new):                                               │
│  var pipeline = mlContext.Transforms.TokenizeText(tokenizerOpts)             │
│      .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))            │
│      .Append(mlContext.Transforms.PoolEmbedding(poolingOpts));               │
│                                                                              │
│  // Convenience API (unchanged):                                             │
│  var estimator = mlContext.Transforms.OnnxTextEmbedding(options);            │
│                                                                              │
│  // MEAI usage (unchanged):                                                  │
│  IEmbeddingGenerator<string, Embedding<float>> gen = ...;                   │
│  var embeddings = await gen.GenerateAsync(texts);                            │
│                                                                              │
│  // Provider-agnostic ML.NET transform (new):                                │
│  var estimator = mlContext.Transforms.TextEmbedding(generator);             │
└──────────────┬────────────────────────────────┬──────────────────────────────┘
               │                                │
   ┌───────────▼──────────────┐     ┌───────────▼─────────────────────┐
   │ OnnxTextEmbedding-       │     │ EmbeddingGenerator-             │
   │ Estimator (facade)       │     │ Estimator (new)                 │
   │                          │     │                                 │
   │ Chains 3 transforms      │     │ Wraps IEmbeddingGenerator       │
   │ internally               │     │ Provider-agnostic               │
   │                          │     │ Text col → Embedding col        │
   │ Returns composite        │     │                                 │
   │ OnnxTextEmbedding-       │     │ Works with:                     │
   │ Transformer              │     │ • OnnxEmbeddingGenerator        │
   └───────────┬──────────────┘     │ • OpenAI / Azure / any MEAI     │
               │                    └─────────────────────────────────┘
               │ chains
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│             Reusable Foundation (any transformer ONNX model)                  │
│                                                                              │
│  ┌────────────────────┐     ┌──────────────────────────────┐                 │
│  │ TextTokenizer-     │     │ OnnxTextModelScorer-         │                 │
│  │ Transformer        │     │ Transformer                  │                 │
│  │                    │     │                              │                 │
│  │ Text →             │     │ TokenIds + AttentionMask +   │                 │
│  │   TokenIds         │────▶│ TokenTypeIds →               │                 │
│  │   AttentionMask    │     │   RawOutput                  │                 │
│  │   TokenTypeIds     │     │                              │                 │
│  │                    │     │ Wraps InferenceSession        │                 │
│  │ Wraps              │     │ Auto-discovers tensor names   │                 │
│  │ BertTokenizer,     │     │ Handles batching              │                 │
│  │ SentencePiece,     │     │ Task-agnostic                 │                 │
│  │ BPE (auto-detect)  │                                                       │
│  └────────────────────┘     └──────────────┬───────────────┘                 │
│                                            │                                 │
└────────────────────────────────────────────┼─────────────────────────────────┘
                                             │
                  ┌──────────────────────────┬┼──────────────────────┐
                  │                          ││                      │
                  ▼                          ▼│                      ▼
  ┌───────────────────────┐  ┌───────────────▼──────┐  ┌────────────────────┐
  │ EmbeddingPooling-     │  │ Softmax-             │  │ NerDecoding-       │
  │ Transformer           │  │ Transformer          │  │ Transformer        │
  │                       │  │                      │  │                    │
  │ RawOutput +           │  │ logits →             │  │ per-token logits → │
  │ AttentionMask →       │  │ class probabilities  │  │ entity spans       │
  │   Embedding           │  │                      │  │                    │
  │                       │  │                      │  │                    │
  │ • Mean/CLS/Max pool   │  │                      │  │                    │
  │ • L2 normalize        │  │                      │  │                    │
  └───────────────────────┘  └──────────────────────┘  └────────────────────┘

            + CrossEncoderTransformer (reranking)
            + QaExtractionTransformer (question answering)
            + ChatClientTransformer (text generation — MEAI)
            + OnnxTextGenerationTransformer (text generation — ORT GenAI)
```

## IDataView Column Flow

```
Input IDataView:
  │ Text (string, TextDataViewType)
  ▼
TextTokenizerTransformer:
  │ Text (string)                       ← passed through
  │ TokenIds (VBuffer<long>)            ← NEW: padded to MaxTokenLength
  │ AttentionMask (VBuffer<long>)       ← NEW: 1=real token, 0=padding
  │ TokenTypeIds (VBuffer<long>)        ← NEW: zeros (segment IDs)
  ▼
OnnxTextModelScorerTransformer:
  │ Text (string)                       ← passed through
  │ TokenIds (VBuffer<long>)            ← passed through
  │ AttentionMask (VBuffer<long>)       ← passed through
  │ TokenTypeIds (VBuffer<long>)        ← passed through
  │ RawOutput (VBuffer<float>)          ← NEW: shape depends on model
  ▼
EmbeddingPoolingTransformer:
  │ Text (string)                       ← passed through
  │ Embedding (VBuffer<float>)          ← NEW: [hiddenDim], pooled + normalized
  ▼
Output IDataView
```

## How Each Task Plugs In

The shared foundation produces raw model output. Each task adds a post-processing transform that interprets this output:

| Task | Post-processor | What It Does |
|------|---------------|-------------|
| Embeddings | `EmbeddingPoolingTransformer` | Mean/CLS/Max pooling + L2 normalization |
| Classification | `SoftmaxClassificationTransformer` | Softmax over logits → class probabilities |
| NER | `NerDecodingTransformer` | Per-token argmax → BIO entity spans |
| Reranking | `CrossEncoderTransformer` | Sigmoid on logit → relevance score |
| QA | `QaExtractionTransformer` | Start/end logit search → answer span |
| Text Gen (MEAI) | `ChatClientTransformer` | Provider-agnostic text generation via `IChatClient` |
| Text Gen (Local) | `OnnxTextGenerationTransformer` | Autoregressive generation via ORT GenAI (e.g., Phi-3) |

## Lazy Evaluation via Custom IDataView / Cursor

Each transform returns a **wrapping IDataView** from `Transform()` — no data is materialized. Computation happens lazily when a downstream consumer iterates via a cursor.

```csharp
// Transform() does NO work — just wraps
public IDataView Transform(IDataView input)
{
    return new TokenizerDataView(input, _tokenizer, _options);
}
```

When the final consumer iterates, cursors chain upstream:

```
PoolerCursor.MoveNext()
  → ScorerCursor.MoveNext()
      → TokenizerCursor.MoveNext()
          → InputCursor.MoveNext()
```

At any given moment, only **one batch** of intermediate data exists in memory (~6 MB for a batch of 32 with a 384-dim model).

### Lookahead Batching (Scorer Only)

The tokenizer and pooler are cheap (microseconds per row) — they process row-by-row. The ONNX scorer uses **lookahead batching**: it reads N rows from the upstream tokenizer cursor, packs them into a single ONNX batch, runs inference once, then serves cached results one at a time. This gives batch throughput with lazy memory semantics.

## Estimator Lifecycle: What Happens in `Fit()`

The facade estimator (`OnnxTextEmbeddingEstimator`) chains three sub-estimators:

```
Fit(IDataView input)
  │
  ├─ 1. Create TextTokenizerEstimator → Fit → TextTokenizerTransformer
  │     Loads tokenizer via smart resolution (directory/config/vocab file)
  │
  ├─ 2. Create OnnxTextModelScorerEstimator → Fit → OnnxTextModelScorerTransformer
  │     Creates InferenceSession, auto-discovers tensor metadata
  │
  ├─ 3. Create EmbeddingPoolingEstimator → Fit → EmbeddingPoolingTransformer
  │     Auto-configured from scorer metadata (HiddenDim, IsPrePooled)
  │
  └─ 4. Return OnnxTextEmbeddingTransformer wrapping all three
```

## MEAI Bridge: OnnxEmbeddingGenerator

The MEAI wrapper delegates to `GenerateEmbeddings()`, which chains the three sub-transforms' **direct faces**:

```
GenerateEmbeddings(texts)
  │
  ├─ _tokenizer.Tokenize(batch) → TokenizedBatch
  ├─ _scorer.Score(batch) → float[][] (raw ONNX output)
  └─ _pooler.Pool(scored, attentionMasks) → float[][] (pooled embeddings)
```

## Save/Load Mechanics

The composite `OnnxTextEmbeddingTransformer` saves/loads as a single zip (same as before):

```
embedding-model.mlnet (zip)
├── model.onnx
├── vocab.txt              ← tokenizer vocabulary (format varies by model)
├── config.json            ← includes all options
└── manifest.json
```

Individual transforms don't need standalone save/load — they're reconstructed from the facade's saved state. The `EmbeddingGeneratorTransformer` does NOT support save/load (since `IEmbeddingGenerator` has no save contract).

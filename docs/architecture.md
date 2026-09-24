# Architecture

This document walks through the components in the `MLNet.TextInference.Onnx` solution and traces the data flow from raw text to task-specific output. The public surface contains distinct task transforms and facades. They share tokenizer, numerical, ONNX-session, asset, batching, and row-mapping foundations, but there is no universal text transform that every task must use.

## Shared Foundation

The conventional encoder path exposes two reusable task-oriented transforms:

1. **`TextTokenizerTransformer`** — Converts raw text into token IDs, attention masks, and token type IDs. Supports BPE, WordPiece, and SentencePiece via smart resolution from HuggingFace model directories.

2. **`OnnxTextModelScorerTransformer`** — Runs the tokenized input through an ONNX encoder model (BERT, RoBERTa, DeBERTa, MiniLM, etc.) and produces raw model output. Uses lookahead batching for efficient ONNX inference while maintaining lazy cursor-based evaluation.

These are not the implementation of every task. Embeddings, classification, NER, reranking, and QA use them where their model contracts fit, then add task-specific post-processing and facades. Typed decisions have a separate five-input packing/scoring/decoding path because their marker positions, question grouping, action probabilities, and output shapes are not conventional encoder outputs. Typed decisions still reuse the shared tokenizer configuration/encoding, stable numerical kernels, ONNX session/provider setup, asset resolution, and row-mapping/batching helpers.

The shared foundations include `HuggingFaceBpeTokenizerLoader` and
`TokenizerEncoding` for Microsoft tokenizer configuration and bounded
encoding, `StableSoftmax` for finite-logit policy, `OnnxSessionFactory` for
observable provider setup, `AssetArchive` for safe extraction and
external-data discovery, and schema-aware row snapshot/getter helpers used by
both conventional and typed adapters. All of these shared kernels and the
ML.NET-specific adapters are compiled into the existing
`MLNet.TextInference.Onnx` assembly and package.

### The Facade Pattern

Conventional encoder tasks that share the text-input contract provide facade estimators that wrap their compatible tokenizer, scorer, and task-specific post-processing stages in a single call. Typed decisions use a separate state/question preparation, decision scorer, and decoder contract rather than being forced through the conventional text transform. Both styles preserve a simple API while allowing advanced users to compose stages directly.

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
│             Conventional encoder path                                       │
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

Typed decisions sit beside this conventional encoder path:

```
State + question metadata
        │
        ▼
PrepareDecisionInputs
        │  input_ids, attention_mask, marker_pos, marker_mask, qtype
        ▼
ScoreOnnxDecisionModel
        │  logits [batch, question_count], act_probs [batch, 2]
        ▼
DecodeDecisions
        │  choice/score/Noul results, distributions, entropy confidence
        ▼
OnnxTypedDecisionsEstimator facade or composable ML.NET stages
```

## Conventional IDataView Column Flow

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

Typed decisions are intentionally not listed as a conventional post-processor: their preparation, model binding, question regrouping, and decoding are model-specific stages described above.

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

### Lookahead Batching

The conventional ONNX scorer and typed-decision stages use **lookahead batching**: they read at most N rows from an upstream cursor, snapshot the declared dependencies, pack one ONNX batch, run inference once, then serve cached rows one at a time. Task-specific tokenization and decoding remain distinct; the shared behavior is the bounded cursor, dependency, row-identity, and ownership machinery rather than a universal task pipeline.

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
├── model/
│   ├── <original-model-basename>
│   └── <external-data-relative-paths>
├── tokenizer/              ← original tokenizer file or directory contents
├── config.json             ← includes options and relative asset names
└── manifest.json            ← package format/framework metadata
```

Individual transforms don't need standalone save/load — they're reconstructed from the facade's portable package where supported. The package preserves the model's original basename under `model/`, external-data sidecars under their relative paths, and tokenizer files under `tokenizer/`. Native ML.NET chain persistence remains a separate, intentionally unimplemented capability. The `EmbeddingGeneratorTransformer` does NOT support save/load (since `IEmbeddingGenerator` has no save contract).

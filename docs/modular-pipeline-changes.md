# Modular Transform Pipeline: What Changed and Why

This document describes the architectural changes introduced in the `proposals/modular-transforms` branch compared to `main`. It covers the design decisions, trade-offs, before/after comparisons, and the new capabilities unlocked by modularization.

## Executive Summary

The monolithic `OnnxTextEmbeddingTransformer` — a single 380-line class that handled tokenization, ONNX inference, and pooling — has been **decomposed into three independent, composable ML.NET transforms**:

| Transform | Responsibility |
|-----------|---------------|
| `TextTokenizerTransformer` | Text → token IDs + attention mask |
| `OnnxTextModelScorerTransformer` | Token IDs → raw model output (last_hidden_state) |
| `EmbeddingPoolingTransformer` | Raw output → pooled, normalized embedding vector |

The original monolithic API still works — `OnnxTextEmbeddingEstimator` is now a thin facade that chains the three transforms internally. Two new integration points were added: `EmbeddingGeneratorEstimator` for provider-agnostic MEAI pipelines, and smart tokenizer resolution that auto-detects tokenizer type from HuggingFace model directories.

---

## Before: The Monolith

### Architecture (main branch)

```
┌─────────────────────────────────────────────────┐
│         OnnxTextEmbeddingEstimator               │
│  Fit() → validates model, loads tokenizer,       │
│          discovers tensors, creates transformer   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         OnnxTextEmbeddingTransformer             │
│  (222 lines — does EVERYTHING)                   │
│                                                  │
│  Transform(IDataView):                           │
│    1. Read text column                           │
│    2. Tokenize with BertTokenizer                │
│    3. Build ONNX input tensors                   │
│    4. Run InferenceSession                       │
│    5. Pool (Mean/CLS/Max)                        │
│    6. Normalize                                  │
│    7. Build output IDataView                     │
│                                                  │
│  + Save/Load (ModelPackager)                     │
│  + GenerateEmbeddings (direct face)              │
└──────────────────────────────────────────────────┘
```

### Problems with the Monolith

1. **No inspectability** — You couldn't see token IDs, attention masks, or raw ONNX output. Everything was hidden inside `Transform()`.

2. **No composability** — Changing pooling strategy required re-fitting the entire estimator and re-running ONNX inference. You couldn't swap just the pooler.

3. **No reuse** — The tokenizer was tightly coupled to ONNX scoring. You couldn't use the tokenizer alone (e.g., for analysis or with a different model).

4. **Single tokenizer type** — Only `BertTokenizer` (WordPiece) was supported. Adding BPE or SentencePiece required modifying the monolith.

5. **Tight coupling** — The transformer held `InferenceSession`, `Tokenizer`, and pooling logic all in one class. Testing any piece in isolation was impossible.

6. **No ML.NET pipeline integration** — The transform couldn't participate in `.Append()` chains because there were no intermediate estimators.

7. **No provider abstraction** — The only way to generate embeddings was through the ONNX-specific API. No path to swap in OpenAI, Azure, or Ollama providers.

### Code shape (main)

```
src/MLNet.TextInference.Onnx/
├── OnnxTextEmbeddingEstimator.cs    (158 lines — loads model, tokenizer, discovers metadata)
├── OnnxTextEmbeddingTransformer.cs  (222 lines — tokenize + infer + pool + normalize)
├── OnnxTextEmbeddingOptions.cs      (configuration)
├── EmbeddingPooling.cs              (SIMD pooling math)
├── ModelPackager.cs                 (zip save/load)
├── OnnxEmbeddingGenerator.cs        (MEAI wrapper)
├── MLContextExtensions.cs           (1 extension method)
└── PoolingStrategy.cs               (enum)
```

**Total implementation:** ~380 lines in 2 core files (estimator + transformer).

---

## After: The Modular Pipeline

### Architecture (proposals/modular-transforms branch)

```
┌─────────────────────────────────────────────────────────────┐
│                         User Code                            │
│                                                              │
│  // Composable (new):                                        │
│  var pipeline = mlContext.Transforms.TokenizeText(...)       │
│      .Append(mlContext.Transforms.ScoreOnnxTextModel(...))   │
│      .Append(mlContext.Transforms.PoolEmbedding(...));       │
│                                                              │
│  // Facade (unchanged API):                                  │
│  var est = new OnnxTextEmbeddingEstimator(mlCtx, opts);     │
│                                                              │
│  // MEAI (new):                                              │
│  var est = mlContext.Transforms.TextEmbedding(generator);   │
└──────┬────────────────┬───────────────────┬─────────────────┘
       │                │                   │
┌──────▼──────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ TextToken-  │  │ OnnxTextEmb │  │ EmbeddingGen-   │
│ izer        │  │ Estimator   │  │ erator          │
│ Estimator   │  │ (facade)    │  │ Estimator       │
└──────┬──────┘  └──────┬──────┘  └────────┬────────┘
       │                │                   │
┌──────▼──────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ TextToken-  │  │ chains:     │  │ Wraps any       │
│ izer        │  │ tok→score   │  │ IEmbedding-     │
│ Transformer │  │ →pool       │  │ Generator       │
│ (239 lines) │  └─────────────┘  └─────────────────┘
└──────┬──────┘
       │
┌──────▼──────────────┐
│ OnnxTextModelScorer  │
│ Transformer          │
│ (410 lines)          │
│ - InferenceSession   │
│ - Lookahead batching │
│ - Lazy cursor        │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ EmbeddingPooling     │
│ Transformer          │
│ (223 lines)          │
│ - Mean/CLS/Max pool  │
│ - L2 normalization   │
└──────────────────────┘
```

### Code shape (proposals/modular-transforms)

```
src/MLNet.TextInference.Onnx/
├── TextTokenizerEstimator.cs           (239 lines — NEW: smart tokenizer resolution)
├── TextTokenizerTransformer.cs         (235 lines — NEW: BPE/WordPiece/SentencePiece)
├── OnnxTextModelScorerEstimator.cs     (168 lines — NEW: ONNX metadata discovery)
├── OnnxTextModelScorerTransformer.cs   (410 lines — NEW: lookahead batching, lazy cursor)
├── Embeddings/
│   ├── EmbeddingPoolingEstimator.cs    (95 lines  — NEW: pooling configuration)
│   ├── EmbeddingPoolingTransformer.cs  (223 lines — NEW: Mean/CLS/Max + normalize)
│   ├── OnnxTextEmbeddingEstimator.cs   (105 lines — REFACTORED: now chains 3 transforms)
│   ├── OnnxTextEmbeddingTransformer.cs (99 lines  — REFACTORED: delegates to 3 sub-transforms)
│   ├── EmbeddingGeneratorEstimator.cs  (151 lines — NEW: provider-agnostic MEAI wrapper)
│   ├── OnnxEmbeddingGenerator.cs       (unchanged — MEAI IEmbeddingGenerator)
│   ├── EmbeddingPooling.cs             (unchanged — SIMD pooling math)
│   ├── ModelPackager.cs                (unchanged — zip save/load)
│   └── OnnxTextEmbeddingOptions.cs     (minor changes)
├── MLContextExtensions.cs              (64 lines  — EXPANDED: 5 extension methods)
└── PoolingStrategy.cs                  (unchanged)
```

**Total implementation:** ~1,370 lines in the three new transform pairs + 204 lines in refactored facade + 151 lines in MEAI estimator.

---

## Design Decisions

### 1. Three transforms, not two or four

**Decision:** Decompose into exactly `Tokenizer → Scorer → Pooler`.

**Rationale:** These are the three natural boundaries in a sentence-transformer pipeline:
- **Tokenizer** is CPU-only, stateless, and reusable across models that share a vocabulary
- **Scorer** owns the ONNX `InferenceSession` — the expensive, GPU-bound resource
- **Pooler** is a pure math operation (mean/max/CLS + normalize) — no model state

We considered finer granularity (e.g., separating normalization from pooling), but that would create transforms with trivial implementations and add pipeline boilerplate without meaningful benefit.

### 2. Dual-face transforms (ML.NET face + Direct face)

**Decision:** Each transform exposes two APIs:
- **ML.NET face** — `Transform(IDataView)` returns a lazy wrapping `IDataView` (standard ML.NET contract)
- **Direct face** — Internal methods like `Tokenize()`, `Score()`, `Pool()` for eager batch processing

**Rationale:** The ML.NET face supports composable pipelines and `.Append()` chains. The direct face enables the `OnnxEmbeddingGenerator` and `GenerateEmbeddings()` to bypass IDataView overhead when generating embeddings for a known batch of strings.

### 3. Lazy evaluation with lookahead batching

**Decision:** The scorer uses a lazy cursor pattern with upstream column caching — it reads from the tokenizer's IDataView lazily but batches ONNX calls using configurable lookahead.

**Rationale:** ML.NET cursors are pull-based (one row at a time), but ONNX inference is most efficient in batches. The scorer bridges this gap by pre-reading `BatchSize` rows, running a single ONNX call, and serving results from the buffer. This preserves ML.NET's lazy evaluation while maximizing GPU throughput.

### 4. Facade preservation (backward compatibility)

**Decision:** `OnnxTextEmbeddingEstimator` and `OnnxTextEmbeddingTransformer` remain the primary API — they just delegate to the three sub-transforms internally.

**Rationale:** Existing code that uses `new OnnxTextEmbeddingEstimator(mlContext, options)` continues to work without changes. The composable pipeline is an additional, opt-in capability for users who need it.

**Before:**
```csharp
// main branch — this is the ONLY way
var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);
var transformer = estimator.Fit(dataView);
```

**After:**
```csharp
// Still works (facade now delegates to 3 sub-transforms):
var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);

// NEW: composable pipeline
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(poolOpts));

// NEW: provider-agnostic
var estimator = mlContext.Transforms.TextEmbedding(anyGenerator);
```

### 5. Smart tokenizer resolution

**Decision:** `TokenizerPath` now accepts a **directory** (not just a file). When given a directory, the tokenizer estimator:
1. Looks for `tokenizer_config.json` and reads `tokenizer_class` to determine the type
2. Dispatches to the appropriate tokenizer: `BpeTokenizer`, `LlamaTokenizer` (SentencePiece), or `BertTokenizer` (WordPiece)
3. Falls back to heuristic detection based on known file patterns (`vocab.txt` → WordPiece, `tokenizer.json` → BPE, `tokenizer.model` → SentencePiece)

**Rationale:** HuggingFace models use different tokenizer types. Requiring the user to know which tokenizer file to point to and what type it is creates unnecessary friction. With directory-based resolution, you just point to the model directory and the library figures it out.

**Before:**
```csharp
TokenizerPath = "models/vocab.txt"  // must know the exact file
```

**After:**
```csharp
TokenizerPath = "models/"  // auto-detects tokenizer type and files
```

### 6. EmbeddingGeneratorEstimator for provider abstraction

**Decision:** Add a new estimator that wraps any `IEmbeddingGenerator<string, Embedding<float>>` as an ML.NET transform.

**Rationale:** Microsoft.Extensions.AI defines a standard embedding interface. By wrapping it as an ML.NET transform, users can build pipelines that are agnostic to the embedding provider — swap ONNX for OpenAI by changing one line.

### 7. Extension methods for fluent API

**Decision:** Expand `MLContextExtensions` from 1 method to 5, enabling the fluent `mlContext.Transforms.TokenizeText(...)` pattern.

**Rationale:** This is the idiomatic ML.NET pattern (e.g., `mlContext.Transforms.NormalizeMinMax()`). Users expect transforms to be discoverable via `mlContext.Transforms.*`.

---

## Trade-offs

### More files, more surface area

The monolith was 2 core files; the modular version is 8 core files (6 new + 2 refactored). This is intentional — each file has a single responsibility, but there's more code to navigate.

**Mitigation:** The facade hides complexity. Users who don't need composability never see the individual transforms.

### Pooling configuration requires model knowledge

With the monolith, pooling was configured automatically because the estimator discovered `HiddenDim` and `HasPooledOutput` during `Fit()`. With the composable pipeline, users must pass these values when constructing the pooler independently.

**Mitigation:** After fitting the scorer, users can read `scorer.HiddenDim` and `scorer.HasPooledOutput` to configure the pooler. The chained `.Append()` pattern handles this by fitting sequentially.

### Intermediate columns in the schema

The composable pipeline adds intermediate columns (`TokenIds`, `AttentionMask`, `TokenTypeIds`, `RawOutput`) to the `IDataView` schema. The monolith produced only `Text` → `Embedding`.

**Mitigation:** These columns are lightweight (lazy, not materialized until consumed) and are actually a feature — the IntermediateInspection sample demonstrates their value for debugging and understanding.

---

## What's New: Capabilities Unlocked

### 1. Swap pooling without re-running inference

```csharp
// Score ONCE (expensive ONNX inference)
var scored = scorer.Transform(tokenized);

// Apply different pooling (cheap math)
foreach (var strategy in new[] { MeanPooling, ClsToken, MaxPooling })
{
    var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
    {
        Pooling = strategy, HiddenDim = scorer.HiddenDim, ...
    }).Fit(scored);
    var result = pooler.Transform(scored);  // no ONNX re-inference
}
```

### 2. Inspect intermediate pipeline state

```csharp
// After tokenization — see what tokens were produced
var tokenIds = tokenized.Schema["TokenIds"];
var attentionMask = tokenized.Schema["AttentionMask"];

// After scoring — see model metadata
Console.WriteLine($"Hidden dim: {scorer.HiddenDim}");
Console.WriteLine($"Pre-pooled: {scorer.HasPooledOutput}");
```

### 3. Provider-agnostic embedding pipelines

```csharp
// Same pipeline code, different provider
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, transformer);  // or OpenAI, Azure, Ollama

var estimator = mlContext.Transforms.TextEmbedding(generator);
var model = estimator.Fit(dataView);
```

### 4. Multiple tokenizer types

```csharp
// WordPiece (BERT-based) — auto-detected from vocab.txt
TokenizerPath = "models/bert-model/"

// BPE (GPT-based) — auto-detected from tokenizer.json
TokenizerPath = "models/gpt-model/"

// SentencePiece (LLaMA-based) — auto-detected from tokenizer.model
TokenizerPath = "models/llama-model/"
```

### 5. Idiomatic ML.NET `.Append()` chains

```csharp
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(poolOpts));

var model = pipeline.Fit(trainData);
var predictions = model.Transform(testData);
```

---

## Samples Added

| Sample | Category | Pattern Demonstrated |
|--------|----------|---------------------|
| BasicUsage | Existing (enhanced) | All API surfaces in one place |
| BgeSmallEmbedding | Multi-model (rewritten) | Composable pipeline + BGE query prefix |
| E5SmallEmbedding | Multi-model (rewritten) | Composable pipeline + E5 dual prefix |
| GteSmallEmbedding | Multi-model (rewritten) | Composable pipeline + semantic search |
| ComposablePoolingComparison | **New** (B1) | 3 pooling strategies, shared inference |
| IntermediateInspection | **New** (B2) | Inspect tokens, masks, raw output |
| MeaiProviderAgnostic | **New** (B3) | Provider-agnostic MEAI transform |

---

## Migration Guide

### Existing users (facade API)

**No changes needed.** The facade API is backward compatible:

```csharp
// This code from main still works identically on proposals/modular-transforms
var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
{
    ModelPath = "models/model.onnx",
    TokenizerPath = "models/vocab.txt",  // still works (file path)
});
var transformer = estimator.Fit(dataView);
var embeddings = transformer.Transform(dataView);
```

### Adopting composable pipeline

To opt in to the composable pipeline, replace the single estimator with three:

```csharp
// Before (facade)
var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);
var transformer = estimator.Fit(dataView);

// After (composable — same result, more control)
var tokenizer = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = "models/",  // can now use directory
    InputColumnName = "Text",
    MaxTokenLength = 128
}).Fit(dataView);

var scorer = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = "models/model.onnx",
    MaxTokenLength = 128,
    BatchSize = 8
}).Fit(tokenizer.Transform(dataView));

var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
{
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    HiddenDim = scorer.HiddenDim,
    IsPrePooled = scorer.HasPooledOutput,
    SequenceLength = 128
}).Fit(scorer.Transform(tokenizer.Transform(dataView)));
```

### Using `TokenizerPath` as directory

The `TokenizerPath` option now accepts either a file or a directory:

```csharp
// File (still works)
TokenizerPath = "models/vocab.txt"

// Directory (new — auto-detects tokenizer type)
TokenizerPath = "models/"
```

# Implementation Plan

## Overview

10 ordered tasks with dependencies. Each task includes acceptance criteria and estimated file changes.

## Dependency Graph

```
tokenizer-transform ─────────────────────────┐
        │                                     │
        ▼                                     │
onnx-scorer-transform ───────────────────┐    │
        │                                │    │
        ▼                                │    │
embedding-pooling-transform ─────────┐   │    │
        │                            │   │    │
        ▼                            │   │    │
refactor-facade-estimator            │   │    │
        │                            │   │    │
        ▼                            │   │    │
refactor-facade-transformer          │   │    │
        │                            │   │    │
        ├─────────────────┐          │   │    │
        ▼                 ▼          │   │    │
update-extensions   update-packager  │   │    │
        │                 │          │   │    │
        ▼                 │          │   │    │
meai-integration          │          │   │    │
        │                 │          │   │    │
        ├─────────────────┘          │   │    │
        ▼                            │   │    │
update-sample ───────────────────────┘───┘────┘
        │
        ▼
update-docs
        │
        ▼
build-test
```

## Task 1: Create TextTokenizerEstimator + TextTokenizerTransformer

**ID:** `tokenizer-transform`
**Dependencies:** None
**Spec:** [01-text-tokenizer-transform.md](01-text-tokenizer-transform.md)

### Files to Create
| File | Lines (est.) |
|------|-------------|
| `src/MLNet.TextInference.Onnx/TextTokenizerEstimator.cs` | ~120 |
| `src/MLNet.TextInference.Onnx/TextTokenizerTransformer.cs` | ~300 (includes TokenizerDataView + TokenizerCursor) |

### Code to Extract From
- `OnnxTextEmbeddingEstimator.LoadTokenizer()` → `TextTokenizerEstimator.LoadTokenizer()` (expanded with smart resolution: directory, config, vocab file)
- `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 145-154 → `TextTokenizerTransformer.Tokenize()` (direct face) and `TokenizerCursor.MoveNext()`
- `OnnxTextEmbeddingTransformer.ReadTextColumn()` → cursor-based per-row reading in `TokenizerCursor`

### Types to Create
- `TextTokenizerOptions` — options class
- `TextTokenizerEstimator` — IEstimator<TextTokenizerTransformer>
- `TextTokenizerTransformer` — ITransformer
- `TokenizerDataView` — wrapping IDataView (lazy, no materialization)
- `TokenizerCursor` — row-by-row tokenization cursor
- `TokenizedBatch` — internal data transfer type for direct face

### Acceptance Criteria
- [ ] Tokenizer loads from vocab.txt
- [ ] `Transform()` returns wrapping IDataView (no materialization)
- [ ] Cursor produces TokenIds, AttentionMask, TokenTypeIds per row
- [ ] Padding/truncation works correctly at MaxTokenLength boundary
- [ ] Input columns are passed through via cursor delegation
- [ ] Direct face `Tokenize()` returns same results as ML.NET face
- [ ] Memory is O(1) per row
- [ ] Builds without errors

---

## Task 2: Create OnnxTextModelScorerEstimator + OnnxTextModelScorerTransformer

**ID:** `onnx-scorer-transform`
**Dependencies:** `tokenizer-transform`
**Spec:** [02-onnx-text-model-scorer-transform.md](02-onnx-text-model-scorer-transform.md)

### Files to Create
| File | Lines (est.) |
|------|-------------|
| `src/MLNet.TextInference.Onnx/OnnxTextModelScorerEstimator.cs` | ~140 |
| `src/MLNet.TextInference.Onnx/OnnxTextModelScorerTransformer.cs` | ~450 (includes ScorerDataView + ScorerCursor with lookahead batching) |

### Code to Extract From
- `OnnxTextEmbeddingEstimator.DiscoverModelMetadata()` → `OnnxTextModelScorerEstimator.DiscoverModelMetadata()`
- `OnnxTextEmbeddingEstimator.FindTensorName()` / `TryFindTensorName()` → same methods on scorer estimator
- `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 156-189 → `RunOnnxBatch()` shared by cursor and direct face

### Types to Create
- `OnnxTextModelScorerOptions` — options class
- `OnnxTextModelScorerEstimator` — IEstimator<OnnxTextModelScorerTransformer>
- `OnnxTextModelScorerTransformer` — ITransformer, IDisposable
- `ScorerDataView` — wrapping IDataView (lazy, no materialization)
- `ScorerCursor` — lookahead batching cursor (reads N rows, runs batch ONNX, serves one at a time)
- `OnnxModelMetadata` — internal record for discovered tensor metadata

### Acceptance Criteria
- [ ] Auto-discovers input/output tensor names from ONNX metadata
- [ ] Manual tensor name overrides work
- [ ] `Transform()` returns wrapping IDataView (no materialization)
- [ ] Cursor reads token columns via lookahead batching (configurable BatchSize)
- [ ] Upstream passthrough columns are cached for the current batch window
- [ ] Runs ONNX inference in configurable batch sizes
- [ ] Outputs correct shape: `float[hiddenDim]` (pre-pooled) or `float[seqLen × hiddenDim]` (unpooled)
- [ ] Direct face `Score()` returns same results as ML.NET face
- [ ] Peak memory ~6 MB regardless of dataset size
- [ ] `Dispose()` disposes InferenceSession
- [ ] Builds without errors

---

## Task 3: Create EmbeddingPoolingEstimator + EmbeddingPoolingTransformer

**ID:** `embedding-pooling-transform`
**Dependencies:** `onnx-scorer-transform`
**Spec:** [03-embedding-pooling-transform.md](03-embedding-pooling-transform.md)

### Files to Create
| File | Lines (est.) |
|------|-------------|
| `src/MLNet.TextInference.Onnx/EmbeddingPoolingEstimator.cs` | ~100 |
| `src/MLNet.TextInference.Onnx/EmbeddingPoolingTransformer.cs` | ~280 (includes PoolerDataView + PoolerCursor) |

### Code to Extract From
- `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 173-183 → `EmbeddingPoolingTransformer.Pool()` (direct face) and `PoolerCursor.MoveNext()`

### Types to Create
- `EmbeddingPoolingOptions` — options class
- `EmbeddingPoolingEstimator` — IEstimator<EmbeddingPoolingTransformer>
- `EmbeddingPoolingTransformer` — ITransformer
- `PoolerDataView` — wrapping IDataView (lazy, no materialization)
- `PoolerCursor` — per-row pooling cursor (lockstep with upstream, direct passthrough delegation)

### Files NOT Modified
- `EmbeddingPooling.cs` — unchanged, continues to provide static math
- `PoolingStrategy.cs` — unchanged

### Acceptance Criteria
- [ ] Mean, CLS, and Max pooling produce correct results
- [ ] Pre-pooled pass-through works (only normalizes)
- [ ] L2 normalization is optional
- [ ] `Transform()` returns wrapping IDataView (no materialization)
- [ ] Cursor processes in lockstep with upstream (no lookahead needed)
- [ ] Passthrough columns delegate directly to upstream cursor
- [ ] Auto-configures from scorer metadata when used via facade
- [ ] Validates HiddenDim and SequenceLength in options
- [ ] Direct face `Pool()` returns same results as ML.NET face
- [ ] Memory is O(1) per row
- [ ] Builds without errors

---

## Task 4: Refactor OnnxTextEmbeddingEstimator as Facade

**ID:** `refactor-facade-estimator`
**Dependencies:** `embedding-pooling-transform`
**Spec:** [04-facade-refactor.md](04-facade-refactor.md)

### Files to Modify
| File | Change |
|------|--------|
| `src/MLNet.TextInference.Onnx/OnnxTextEmbeddingEstimator.cs` | Major refactor: compose 3 sub-transforms |

### What Changes
- `Fit()` creates and chains TokenizerEstimator → ScorerEstimator → PoolingEstimator
- `GetOutputSchema()` delegates through the chain
- `DiscoverModelMetadata()` moves to `OnnxTextModelScorerEstimator`
- `LoadTokenizer()` moves to `TextTokenizerEstimator`
- `FindTensorName()` / `TryFindTensorName()` move to scorer estimator

### What Stays
- Constructor signature
- Public API surface
- `OnnxTextEmbeddingOptions` (unchanged)

### Acceptance Criteria
- [ ] `Fit()` returns `OnnxTextEmbeddingTransformer` (same as before)
- [ ] No public API changes
- [ ] Builds without errors

---

## Task 5: Refactor OnnxTextEmbeddingTransformer as Composite

**ID:** `refactor-facade-transformer`
**Dependencies:** `refactor-facade-estimator`
**Spec:** [04-facade-refactor.md](04-facade-refactor.md)

### Files to Modify
| File | Change |
|------|--------|
| `src/MLNet.TextInference.Onnx/OnnxTextEmbeddingTransformer.cs` | Major refactor: wrap 3 sub-transformers |

### What Changes
- Fields: replace `_session`, `_tokenizer`, tensor names → `_tokenizer`, `_scorer`, `_pooler` sub-transforms
- `Transform()`: chains wrapping DataViews: `_tokenizer.Transform() → _scorer.Transform() → _pooler.Transform()` (all lazy, no materialization until cursor is iterated)
- `GenerateEmbeddings()`: chains direct faces `Tokenize() → Score() → Pool()` (eager, batch-oriented)
- `EmbeddingDimension`: delegates to `_scorer.HiddenDim`
- `Dispose()`: disposes `_scorer` (which disposes InferenceSession)

### What Stays
- Class name and public API
- `Save()` / `Load()`
- `IsRowToRowMapper`
- `GetOutputSchema()` (delegates through chain)

### What's Removed
- `ProcessBatch()` — logic distributed to sub-transforms
- `ReadTextColumn()` — moved to tokenizer
- `BuildOutputDataView()` — moved to pooler
- `EmbeddingRow` — moved to pooler
- Direct InferenceSession, Tokenizer, tensor name fields

### Acceptance Criteria
- [ ] `Transform()` produces identical output to current implementation
- [ ] `GenerateEmbeddings()` produces identical output
- [ ] `Save()` / `Load()` round-trip works
- [ ] `EmbeddingDimension` returns correct value
- [ ] `Dispose()` works correctly
- [ ] No public API changes
- [ ] Builds without errors

---

## Task 6: Update MLContextExtensions

**ID:** `update-extensions`
**Dependencies:** `refactor-facade-transformer`
**Spec:** [05-meai-integration.md](05-meai-integration.md)

### Files to Modify
| File | Change |
|------|--------|
| `src/MLNet.TextInference.Onnx/MLContextExtensions.cs` | Add extension methods |

### New Extension Methods
- `mlContext.Transforms.TokenizeText(options)` → `TextTokenizerEstimator`
- `mlContext.Transforms.ScoreOnnxTextModel(options)` → `OnnxTextModelScorerEstimator`
- `mlContext.Transforms.PoolEmbedding(options)` → `EmbeddingPoolingEstimator`

### Acceptance Criteria
- [ ] All four extension methods work
- [ ] Existing `OnnxTextEmbedding()` method unchanged
- [ ] Builds without errors

---

## Task 7: Update ModelPackager

**ID:** `update-packager`
**Dependencies:** `refactor-facade-transformer`

### Files to Modify
| File | Change |
|------|--------|
| `src/MLNet.TextInference.Onnx/ModelPackager.cs` | Access sub-transform state via new structure |

### What Changes
- `Save()`: access model path via `transformer.Scorer.Options.ModelPath` (or keep via `transformer.Options`)
- `Load()`: reconstructs via `OnnxTextEmbeddingEstimator.Fit()` (which now creates sub-transforms)

### What Might Stay
If `OnnxTextEmbeddingTransformer` still exposes `Options` (the original `OnnxTextEmbeddingOptions`), the packager may need no changes at all — it accesses `ModelPath`, `TokenizerPath`, etc. from that options object.

### Acceptance Criteria
- [ ] `Save()` creates identical zip file
- [ ] `Load()` reconstructs working transformer
- [ ] Round-trip produces identical embeddings
- [ ] Builds without errors

---

## Task 8: Create EmbeddingGeneratorEstimator + Update MEAI Integration

**ID:** `meai-integration`
**Dependencies:** `update-extensions`
**Spec:** [05-meai-integration.md](05-meai-integration.md)

### Files to Create
| File | Lines (est.) |
|------|-------------|
| `src/MLNet.TextInference.Onnx/EmbeddingGeneratorEstimator.cs` | ~120 |

### Files to Modify
| File | Change |
|------|--------|
| `src/MLNet.TextInference.Onnx/MLContextExtensions.cs` | Add `TextEmbedding(generator)` extension |

### Files NOT Modified
- `OnnxEmbeddingGenerator.cs` — unchanged (benefits automatically from facade refactor)

### Types to Create
- `EmbeddingGeneratorOptions` — options class
- `EmbeddingGeneratorEstimator` — IEstimator<EmbeddingGeneratorTransformer>
- `EmbeddingGeneratorTransformer` — ITransformer

### Acceptance Criteria
- [ ] Works with `OnnxEmbeddingGenerator`
- [ ] Works with any `IEmbeddingGenerator<string, Embedding<float>>`
- [ ] `Transform()` produces correct embeddings
- [ ] Extension method `mlContext.Transforms.TextEmbedding(generator)` works
- [ ] Save throws NotSupportedException with clear message
- [ ] Builds without errors

---

## Task 9: Update Sample

**ID:** `update-sample`
**Dependencies:** `meai-integration`

### Files to Modify
| File | Change |
|------|--------|
| `samples/BasicUsage/Program.cs` | Add section 5: Composable Pipeline |

### New Section
Add a section demonstrating the composable pipeline alongside existing convenience API:

```csharp
// --- 5. Composable Pipeline ---
Console.WriteLine("5. Composable Pipeline");

var compPipeline = mlContext.Transforms.TokenizeText(new TextTokenizerOptions { ... })
    .Append(mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions { ... }))
    .Append(mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions { ... }));

var compChain = compPipeline.Fit(dataView);
var compResult = compChain.Transform(dataView);
// Verify results match the convenience API
```

### Acceptance Criteria
- [ ] Composable pipeline produces same embeddings as convenience API
- [ ] Max difference between the two approaches is ~0
- [ ] Sample runs to completion without errors

---

## Task 10: Update Documentation

**ID:** `update-docs`
**Dependencies:** `update-sample`

### Files to Modify
| File | Change |
|------|--------|
| `docs/architecture.md` | Update component map, add modular pipeline diagram |
| `docs/design-decisions.md` | Add section on modularization rationale and tradeoffs |
| `docs/extending.md` | Update with new extension points (post-processing transforms) |

### Key Documentation Updates

**architecture.md:**
- New component diagram showing three transforms + facades
- Updated data flow showing IDataView column progression
- Document the two faces (ML.NET face + direct face)

**design-decisions.md:**
- Why modularize (composability, reusability, inspectability)
- Memory tradeoff analysis
- Why the scorer is task-agnostic
- Why the facade preserves the efficient direct path

**extending.md:**
- How to add a new post-processing transform
- How to use the composable pipeline
- How to use the provider-agnostic EmbeddingGeneratorEstimator

### Acceptance Criteria
- [ ] Architecture docs reflect the new structure
- [ ] Design decisions explain the tradeoffs
- [ ] Extending docs show how to add new task transforms

---

## Task 11: Build and Test End-to-End

**ID:** `build-test`
**Dependencies:** `update-docs`

### Steps
1. `dotnet build` the solution
2. Run `BasicUsage` sample
3. Verify convenience API produces same results as before
4. Verify composable pipeline produces same results
5. Verify MEAI generator produces same results
6. Verify save/load round-trip works

### Acceptance Criteria
- [ ] Solution builds without errors or warnings
- [ ] Sample runs to completion
- [ ] All embedding values match between convenience and composable paths
- [ ] Save/load round-trip produces identical embeddings

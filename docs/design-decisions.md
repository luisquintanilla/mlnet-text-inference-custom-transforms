# Design Decisions

This document explains *why* every major design choice was made. It's written for developers and AI coding agents who need to understand the trade-off space before modifying or extending the `MLNet.TextInference.Onnx` solution.

## Typed-decision package boundary

Typed decisions remain in the existing `MLNet.TextInference.Onnx` assembly and
NuGet package. Their task-specific preparation, scoring, decoding, tokenizer,
asset, and numerical kernels are internal implementation details reused by the
fitted transformer path and the ML.NET schema-aware stages. The public surface
remains the existing ML.NET estimators, transformers, and
`TransformsCatalog` extension methods; no standalone core package or wrapper
facade is introduced.

## The ML.NET Constraint Landscape

ML.NET's pipeline model is built on two interfaces: `IEstimator<TTransformer>` (learns from data, produces a transformer) and `ITransformer` (applies a transformation to an `IDataView`). When you need a transform that ML.NET doesn't provide — such as running text through a HuggingFace ONNX model — you must implement these interfaces yourself.

The challenge is that ML.NET's most convenient base classes are inaccessible from external code:

```
RowToRowTransformerBase          ← private protected constructor
OneToOneTransformerBase          ← private protected constructor
MapperBase / OneToOneMapperBase  ← private protected constructor
TrivialEstimator<T>              ← private protected constructor
```

These classes handle lazy evaluation (cursor-based streaming), schema propagation, and save/load via `[LoadableClass]` attributes. From an external project, you can't use any of them.

The [ML.NET Custom Transformer Guide](https://github.com/luisquintanilla/mlnet-custom-transformer-guide) documents four approaches:

| Approach | Pattern | External Project? | Lazy Eval | Save/Load | Limitation |
|----------|---------|:------------------:|:---------:|:---------:|------------|
| **A** | `CustomMapping` lambda | ✅ | ✅ | ⚠️ factory | Static POCO schema, no lifecycle hooks |
| **B** | Production Facade + `CustomMapping` | ✅ | ✅ | ⚠️ factory | Still POCO-static under the hood |
| **C** | Direct `IEstimator`/`ITransformer` | ✅ | ✅ cursor/direct | ❌ native | Custom row mappers and task-specific portable packaging |
| **D** | `RowToRowTransformerBase` subclass | ❌ | ✅ | ✅ | Must be inside `dotnet/machinelearning` repo |

**None of the approaches give us everything we need.** Our requirements are:

1. ✅ External project (this is a prototype, not in the ML.NET repo)
2. ✅ Task-specific save/load (serialize supported assets to a portable model file)
3. ❌ No `CustomMapping` (static `[VectorType]` dimensions can't adapt to different models)
4. ✅ Resource management (ONNX InferenceSession, tokenizer lifecycle)

### Why We Chose Approach C Enhanced

Approach C gives us full control — we implement `IEstimator<T>` and `ITransformer` directly. The "enhanced" part is supplying explicit row mappers, bounded cursor batching, direct batch APIs, and task-specific zip packaging. `ICanSaveModel` is public, but the surrounding ML.NET save context and registration infrastructure is not a supported extension point for this external package, so native chain persistence remains intentionally unsupported.

**Why not A/B (CustomMapping)?** The `CustomMapping` transform requires POCO classes with compile-time `[VectorType(N)]` attributes. For embedding models, `N` varies by model (384 for MiniLM, 768 for MPNet). You'd need to recompile for each model. Additionally, the `CustomMappingFactory` save/load pattern requires assembly scanning and static state for reconstruction — fragile and unintuitive.

**Why not D (internal base classes)?** This is a prototype. We want to iterate quickly without forking the ML.NET repo. If this proves valuable, the code can be ported to Approach D later (see [extending.md](extending.md)).

## Lazy Cursors with Bounded Lookahead

`Transform()` returns a wrapping `IDataView`; it does not enumerate the input. The cursor reads at most the configured batch size, snapshots only requested passthrough columns, runs one ONNX call for that batch, and serves the cached rows one at a time. A passthrough-only projection does not activate inference or read token columns. Direct facade methods retain an eager, explicitly batched path for callers that already have in-memory text:

```csharp
for (int start = 0; start < texts.Count; start += batchSize)
{
    int count = Math.Min(batchSize, texts.Count - start);
    var batchEmbeddings = ProcessBatch(batchTexts);
    allEmbeddings.AddRange(batchEmbeddings);
}
```

This keeps memory bounded without changing row order or requiring whole-dataset materialization. Row-local snapshots are keyed by schema index and preserve scalar, text, dense, and sparse vector values. Fitted transformers own native sessions and tokenizers; mapped rows own only their managed snapshots.

## Save/Load Strategy

ML.NET's native `Model.Save()` calls the public `ICanSaveModel.Save(ModelSaveContext)` contract on each transformer in the chain. The surrounding `ModelSaveContext`/registration infrastructure is not usable by this external package in a supported way, so the current transformers deliberately reject native chain persistence. This is distinct from the task-specific portable package API.

We evaluated three options:

| Option | Portable? | Size | `mlContext.Model.Save()` compatible? |
|--------|:---------:|:----:|:------------------------------------:|
| **A — Custom zip** | ✅ | ~80 MB | ❌ |
| B — Reference paths | ❌ | ~1 KB | ❌ |
| C — TransformerChain + CustomMapping | ✅ | ~80 MB | ✅ |

**We chose Option A** — a self-contained, versioned zip file containing:

```
embedding-model.mlnet (zip)
├── model/
│   ├── <original-model-basename> — The ONNX graph (copied verbatim)
│   └── <external-data-relative-paths> — ONNX external-weight sidecars
├── tokenizer/
│   └── <original tokenizer directory/files>
├── config.json       — Serialized options and relative asset names
└── manifest.json     — Format/framework/version metadata
```

**Why self-contained?** The ONNX model IS the model — it makes no sense to save a path reference that breaks when the file moves. The zip is ~80 MB for MiniLM (mostly the ONNX file), which is comparable to ML.NET's own saved models with embedded weights. External ONNX sidecars and tokenizer companions/directories are stored under their relative asset paths. Extraction is root-bounded and validates referenced files before loading.

**Why not Option C?** It would require using `CustomMapping` internally (which the user explicitly ruled out) and the `CustomMappingFactory` assembly-scanning pattern for reconstruction.

**Loading:** `ModelPackager.Load()` extracts the zip to a temp directory, reads `config.json` to reconstruct `OnnxTextEmbeddingOptions`, then uses `OnnxTextEmbeddingEstimator.Fit()` to recreate the transformer with full auto-discovery.

## ONNX Auto-Discovery

Most ML.NET transforms require the user to manually specify input/output column mappings. We chose auto-discovery because sentence-transformer models follow a strong convention:

**Inputs:** `input_ids`, `attention_mask`, `token_type_ids` (optional)
**Outputs:** `last_hidden_state` (needs pooling) or `sentence_embedding` (pre-pooled)

The estimator probes `InferenceSession.InputMetadata` and `OutputMetadata` at `Fit()` time:

```csharp
// Discover inputs by convention
string inputIdsName = FindTensorName(inputMeta, ["input_ids"], "input_ids");
string attentionMaskName = FindTensorName(inputMeta, ["attention_mask"], "attention_mask");

// Discover outputs — prefer pre-pooled if available
var pooledName = TryFindTensorName(outputMeta, ["sentence_embedding", "pooler_output"]);
if (pooledName != null) { /* skip manual pooling */ }
else { /* use last_hidden_state + mean pooling */ }

// Embedding dimension from the last axis of the output tensor
int hiddenDim = (int)outputMeta[outputName].Dimensions.Last();
```

This mirrors how ML.NET's own `OnnxTransformer` works internally — it creates an `OnnxModel` that inspects the ONNX graph for input/output metadata. The difference is that our estimator applies domain knowledge (sentence-transformer conventions) to provide zero-configuration defaults.

**Manual override:** Every auto-discovered value can be overridden via `OnnxTextEmbeddingOptions` for non-standard models.

## GPU Execution Provider Design

ONNX inference is the computational bottleneck of the embedding pipeline. Transformer models see 5-20× speedups on GPU. The GPU support design involved several deliberate trade-offs.

### Why `OnnxRuntime.Managed` (No Native Binaries)

The library references `Microsoft.ML.OnnxRuntime.Managed` — the managed API surface only, with zero native binaries. The consuming application decides the execution provider:

| User's Package | Effect |
|---------------|--------|
| `Microsoft.ML.OnnxRuntime` | CPU native libs (default) |
| `Microsoft.ML.OnnxRuntime.Gpu` | CUDA native libs |

Both packages pull in `Managed` transitively, so the managed API (`InferenceSession`, `SessionOptions`, `OrtValue`) is always available regardless of which the user chooses.

**Why not keep `Microsoft.ML.OnnxRuntime` (the CPU package)?** If the library ships with the CPU native package, users who want GPU must deal with conflicting native binaries — the CPU package's native libs vs the GPU package's. By shipping managed-only, there's no conflict. This mirrors how ML.NET's own `Microsoft.ML.OnnxTransformer` package works.

**Trade-off:** Sample projects (and any consumer) must now explicitly reference a native runtime package. We added `Microsoft.ML.OnnxRuntime` to all sample `.csproj` files to maintain the zero-friction experience for CPU users.

### GPU Device Resolution Order

```
Per-estimator options.GpuDeviceId  →  MLContext.GpuDeviceId  →  null (CPU)
```

This follows ML.NET's established convention from [`OnnxCatalog.ApplyOnnxModel`](https://github.com/dotnet/machinelearning/blob/70d76033/src/Microsoft.ML.OnnxTransformer/OnnxCatalog.cs). The per-estimator override exists because a user might run tokenization on CPU and scoring on GPU, or target different GPU devices for different models. `MLContext.GpuDeviceId` provides a convenient "set once, apply everywhere" default.

### Why `FallbackToCpu` Defaults to `false`

Fail-fast is the right default for GPU configuration. If a user explicitly requests GPU and CUDA isn't available, they should know immediately (via an exception) rather than silently running 10× slower on CPU. Users who want graceful degradation opt in explicitly with `FallbackToCpu = true`.

When fallback is enabled, the `OnnxSessionFactory` catches only recognized CUDA/provider initialization failures and creates a CPU session. The fallback is explicit and observable through the warning callback; unrelated model errors are not disguised:

```csharp
try
{
    options.AppendExecutionProvider_CUDA(deviceId.Value);
}
catch (Exception exception) when (IsProviderInitializationFailure(exception))
{
    warning?.Invoke("CUDA execution provider setup failed; retrying CPU.");
}
```

### Why GPU Settings Are Not Serialized

`GpuDeviceId` and `FallbackToCpu` are runtime concerns, not model artifacts. A model saved on a GPU machine must load on a CPU-only machine without error. The `ModelPackager` serializes `OnnxTextEmbeddingOptions` to `config.json` inside the zip, but GPU settings are excluded from this serialization — they're set at runtime when the model is loaded and used.

### Why No Changes to Data Flow / Cursor Code

`OnnxTextModelScorerTransformer.RunOnnxBatch()` uses `OrtValue.CreateTensorValueFromMemory()` which works with CPU memory. When a GPU execution provider is active, ORT automatically copies input tensors to GPU memory before execution and copies output tensors back to CPU memory after. This is transparent — no code changes needed.

**Deferred optimization:** [IO Binding](https://onnxruntime.ai/docs/api/python/io_binding.html) would pre-allocate GPU buffers to eliminate per-call CPU↔GPU copies. This is a valid optimization for high-throughput scenarios but requires a larger refactor of `RunOnnxBatch()` and is out of scope for the initial GPU support.

### The `GetMLContext()` Reflection Workaround

The extension methods on `TransformsCatalog` (e.g., `mlContext.Transforms.ScoreOnnxTextModel(...)`) need to preserve the creating context's provider settings. `TransformsCatalog` does not publicly expose the original `MLContext` instance. The implementation therefore reflects its internal host environment, reconstructs a context with the originating seed, GPU device, CPU-fallback, and temp-path settings, and fails explicitly if that compatibility surface is unavailable.

```csharp
var environment = GetInternalEnvironment(catalog);
var recovered = new MLContext(environment.Seed);
recovered.GpuDeviceId = environment.GpuDeviceId;
recovered.FallbackToCpu = environment.FallbackToCpu;
recovered.TempFilePath = environment.TempFilePath;
```

This is fragile (it relies on an internal property name and shape) but makes the loss of context settings visible instead of silently creating an unrelated default context. It pairs with the `SchemaShape.Column` reflection workaround described below — both are friction points that disappear with Approach D.

**Approach D migration path:** Inside the ML.NET repo, this reflection is replaced by `CatalogUtils.GetEnvironment(catalog)` (an internal helper) and casting to `IHostEnvironmentInternal` for `GpuDeviceId` / `FallbackToCpu`. The rest of the GPU plumbing (`CreateSessionOptions`, options classes, resolution order) transfers directly.

## Thread Safety

**`InferenceSession`:** OnnxRuntime's documentation states that `Run()` is thread-safe for concurrent calls. The session handles internal locking. We use a single session per transformer instance.

**`Tokenizer`:** `Microsoft.ML.Tokenizers` tokenizers are stateless after construction. `EncodeToIds()` is safe to call concurrently.

**`Transform()`:** The IDataView adapter is lazy and each cursor owns its lookahead buffers; the fitted transformer owns a shared session. The direct APIs process their own batches. OnnxRuntime sessions are safe for concurrent `Run()` calls, while cursor state and tokenizer buffers are not shared across cursors.

## The SchemaShape.Column Problem

ML.NET's `SchemaShape.Column` is a `readonly struct` with a *non-public constructor*. The `GetOutputSchema()` method on `IEstimator<T>` must return a `SchemaShape` containing these columns — but external code can't construct them through normal means.

Our workaround uses reflection:

```csharp
var colCtor = typeof(SchemaShape.Column).GetConstructors(
    BindingFlags.NonPublic | BindingFlags.Instance)[0];
var outputCol = (SchemaShape.Column)colCtor.Invoke([
    outputColumnName, VectorKind.Vector, NumberDataViewType.Single, false, null
]);
```

This is a known friction point for external ML.NET transform authors. It works reliably because the constructor signature has been stable across ML.NET versions, but it's another reason to eventually move to Approach D inside the ML.NET repo. (See also the `GetMLContext()` reflection workaround in the GPU section above — both are reflection hacks that disappear with internal access.)

## The ICanSaveModel Requirement

`ITransformer` inherits from the public `ICanSaveModel` interface, which requires implementing `void Save(ModelSaveContext ctx)`. The surrounding `ModelSaveContext` registration and loader infrastructure is not a supported extension point for this external package, so we throw `NotSupportedException`:

```csharp
void ICanSaveModel.Save(ModelSaveContext ctx)
{
    throw new NotSupportedException(
        "ML.NET native save is not supported. Use transformer.Save(path) instead.");
}
```

Users call the task-specific portable package API where one exists instead of `mlContext.Model.Save(transformer, schema, "path")`. Native ML.NET persistence remains intentionally unimplemented; portable asset packaging is a separate format and does not make arbitrary ML.NET chains natively serializable.

## Tokenizer Loading: Smart Resolution via `TokenizerPath`

The all-MiniLM-L6-v2 model (and most BERT-derived sentence-transformers) uses **WordPiece** tokenization, distributed as `vocab.txt`. Other models use SentencePiece (`.model`) or BPE (`vocab.json` + `merges.txt`).

`Microsoft.ML.Tokenizers` v2.0.0 provides:
- `BertTokenizer.Create(Stream vocabStream, BertOptions?)` — for WordPiece/BERT models
- `LlamaTokenizer.Create(Stream)` — for SentencePiece models (XLMRoberta, T5, etc.)
- `BpeTokenizer.Create(Stream vocab, Stream? merges)` — for GPT-2/BPE models

Rather than requiring users to know which tokenizer type their model uses, `LoadTokenizer()` uses a **smart resolution** strategy via a single `TokenizerPath` property:

1. **Directory with `tokenizer_config.json`** → reads the HuggingFace config's `tokenizer_class` field (e.g., `"BertTokenizer"`, `"XLMRobertaTokenizer"`, `"DebertaV2Tokenizer"`, `"GPT2Tokenizer"`), maps it to the appropriate `Microsoft.ML.Tokenizers` class, and loads sibling vocab files. Also applies config options like `do_lower_case`.
2. **Directory without config** → scans for known files (`vocab.txt` → BERT, `tokenizer.model`/`sentencepiece.bpe.model`/`spm.model` → SentencePiece, `tokenizer.json` → BPE/WordPiece fast tokenizer as last resort).
3. **`tokenizer_config.json` file** → same as (1), uses the file's parent directory for sibling resolution.
4. **`tokenizer.json` file** → HuggingFace fast tokenizer format. Parses the `model.type` field and extracts vocab/merges from JSON to construct `BpeTokenizer` (for BPE models) or `BertTokenizer` (for WordPiece models). Also reads the `normalizer.lowercase` setting for uncased models. Throws `NotSupportedException` for Unigram models (use the `.model` protobuf file instead).
5. **Direct vocab file** → infers type from extension (`.txt` → BERT, `.model` → SentencePiece).
6. **Pre-constructed `Tokenizer` instance** → the `Tokenizer` property on `TextTokenizerOptions` bypasses all resolution. Use this for exotic formats or shared tokenizer instances.

This design mirrors HuggingFace's `AutoTokenizer.from_pretrained(path)` pattern — one input, smart resolution — while keeping the API surface minimal (two properties: `Tokenizer` and `TokenizerPath`).

## Modularization: Why Decompose Into Three Transforms

The original monolithic `OnnxTextEmbeddingTransformer` bundled tokenization, ONNX inference, and pooling into a single class. This was refactored into three composable transforms:

| Transform | Responsibility | Reusability |
|-----------|---------------|-------------|
| `TextTokenizerTransformer` | Text → token IDs + attention mask | Any transformer model |
| `OnnxTextModelScorerTransformer` | Token columns → raw ONNX output | Any transformer ONNX model |
| `EmbeddingPoolingTransformer` | Raw output → pooled embedding | Embedding generation |

### Why Modularize?

1. **Composability**: ML.NET's design is composable pipelines of single-responsibility transforms. The monolith violated this.
2. **Reusability**: Conventional encoder tasks often start with tokenized text fed through an ONNX model, so the shared tokenizer/scorer is useful for classification, NER, QA, reranking, and embeddings where their contracts match. Typed decisions use separate model-specific packing and scoring while reusing lower-level tokenizer, runtime, numerical, and mapping helpers.
3. **Inspectability**: Users can inspect intermediate results (what tokens were produced? what does the raw model output look like?).
4. **Extensibility**: Adding a new task (e.g., text classification) requires only a new post-processing transform, not a new end-to-end pipeline.
5. **Testability**: Each transform can be unit-tested in isolation.

### Why Keep the Facade?

The `OnnxTextEmbeddingEstimator`/`OnnxTextEmbeddingTransformer` remain as a convenience facade that chains all three transforms internally. This preserves the existing public API (zero breaking changes) while allowing advanced users to compose the transforms directly. Future tasks (classification, NER, reranking, QA) will each get their own facade following the same pattern.

## Why OnnxTextModelScorer Stays Generic

The scorer is intentionally named `OnnxTextModelScorerTransformer`, not `OnnxTextEncoderScorerTransformer` or `OnnxEmbeddingScorerTransformer`. This is deliberate:

1. **Task-agnostic by design**: The scorer runs *any* ONNX encoder model and outputs raw tensors. It has no knowledge of what the downstream task will be — embeddings, classification, NER, reranking, or QA. Naming it after a specific task or architecture would be misleading.

2. **Shared across conventional encoder tasks**: The scorer is reused by task transforms whose graphs accept the conventional encoder tensor contract. Typed decisions deliberately have a separate scorer because their five named inputs and grouped marker outputs do not fit that contract.

3. **Encoder agnostic**: While the current focus is encoder transformers (BERT, RoBERTa, etc.), the scorer's contract is simply "take token columns, run ONNX, output raw tensors." This could theoretically work with non-encoder architectures that accept the same input format.

4. **Convention over configuration**: The name reflects what the class *does* (scores text via an ONNX model) rather than what architecture it targets. This follows the ML.NET naming convention where transforms are named by their action, not their implementation detail.

### Lazy vs Eager Evaluation

The modular transforms use **lazy evaluation** via custom `IDataView`/cursor wrappers. `Transform()` returns a wrapping `IDataView` — no data is materialized. Computation happens on-demand when a cursor iterates.

The facade's `GenerateEmbeddings()` (used by MEAI) uses **eager evaluation** via the transforms' direct faces (`Tokenize()` → `Score()` → `Pool()`), bypassing `IDataView` entirely for zero-overhead batch processing.

### Memory Tradeoff

Lazy evaluation eliminates the intermediate materialization concern. Peak memory is bounded by `BatchSize × rowSize` (~6 MB for batch=32 with a 384-dim model), regardless of dataset size. The scorer cursor achieves batch throughput via lookahead batching — reading N rows ahead, running a single `session.Run()`, then serving cached results one at a time.

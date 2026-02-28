# Extending

This document covers how to modify, extend, and harden the solution. Each section includes the specific files to change and the patterns to follow.

## How to Add a New Task

The platform is designed so that adding a new task requires only a few new files — the shared tokenizer and scorer are reused. All encoder tasks (embeddings, classification, NER, reranking, QA) already follow this pattern. Follow these steps:

### Step 1: Create an options class

Create a new file in the appropriate task subdirectory (e.g., `src/MLNet.TextInference.Onnx/Classification/`):

```csharp
namespace MLNet.TextInference.Onnx.Classification;

public class SoftmaxClassificationOptions
{
    public string InputColumnName { get; set; } = "RawOutput";
    public string ProbabilitiesColumnName { get; set; } = "Probabilities";
    public string PredictedLabelColumnName { get; set; } = "PredictedLabel";
    public string[]? Labels { get; set; }  // e.g., ["negative", "positive"]
}
```

### Step 2: Create a post-processing estimator + transformer

Follow the same pattern as `EmbeddingPoolingEstimator` / `EmbeddingPoolingTransformer`:

- The **estimator** validates options and returns a fitted transformer
- The **transformer** implements `ITransformer` with both ML.NET and direct faces:
  - `Transform(IDataView)` — returns a lazy wrapping `IDataView`
  - Internal batch method (e.g., `Classify()`) — eager, for the facade's direct path

```csharp
namespace MLNet.TextInference.Onnx.Classification;

public class SoftmaxClassificationEstimator
    : IEstimator<SoftmaxClassificationTransformer>
{
    public SoftmaxClassificationTransformer Fit(IDataView input) { ... }
    public SchemaShape GetOutputSchema(SchemaShape inputSchema) { ... }
}
```

The transformer reads the `RawOutput` column from the scorer, applies task-specific logic (softmax, argmax, BIO decoding, etc.), and outputs new columns.

### Step 3: Create a facade estimator

Create a convenience facade that chains tokenizer → scorer → your post-processor:

```csharp
namespace MLNet.TextInference.Onnx.Classification;

public class OnnxTextClassificationEstimator
    : IEstimator<OnnxTextClassificationTransformer>
{
    public OnnxTextClassificationEstimator(MLContext mlContext, OnnxTextClassificationOptions options) { ... }

    public OnnxTextClassificationTransformer Fit(IDataView input)
    {
        // 1. Create and fit tokenizer
        // 2. Create and fit scorer
        // 3. Create and fit SoftmaxClassification post-processor
        // 4. Return composite transformer
    }
}
```

### Step 4: Add extension methods to MLContextExtensions.cs

Add fluent API extensions for both the post-processor and the facade:

```csharp
// Post-processor extension
public static SoftmaxClassificationEstimator SoftmaxClassify(
    this TransformsCatalog catalog, SoftmaxClassificationOptions options)
{
    return new SoftmaxClassificationEstimator(options);
}

// Facade extension
public static OnnxTextClassificationEstimator OnnxTextClassification(
    this TransformsCatalog catalog, OnnxTextClassificationOptions options)
{
    var mlContext = GetMLContext(catalog);
    return new OnnxTextClassificationEstimator(mlContext, options);
}
```

### Step 5: Create a sample

Add a new sample directory (e.g., `samples/SentimentClassification/`) demonstrating both the composable pipeline and the facade:

```csharp
// Composable pipeline
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.SoftmaxClassify(classOpts));

// Facade
var estimator = mlContext.Transforms.OnnxTextClassification(options);
```

### Summary checklist for a new task

- [ ] Options class in `src/MLNet.TextInference.Onnx/<Task>/`
- [ ] Post-processing estimator + transformer
- [ ] Facade estimator + transformer (chains tokenizer → scorer → post-processor)
- [ ] Extension methods in `MLContextExtensions.cs`
- [ ] Sample in `samples/`
- [ ] Update the task status table in `README.md`
- [ ] Update `docs/architecture.md` component map

## Adding New Pooling Strategies

**File to modify:** `src/MLNet.TextInference.Onnx/Embeddings/EmbeddingPooling.cs`

1. Add a value to the `PoolingStrategy` enum in `Embeddings/PoolingStrategy.cs`:

```csharp
public enum PoolingStrategy
{
    MeanPooling,
    ClsToken,
    MaxPooling,
    WeightedMeanPooling  // ← new
}
```

2. Add a private method and a case to the `Pool()` switch in `Embeddings/EmbeddingPooling.cs`:

```csharp
PoolingStrategy.WeightedMeanPooling => WeightedMeanPool(
    hiddenStates, attentionMask, b, seqLen, hiddenDim),
```

3. Implement the method following the existing pattern — use `TensorPrimitives` for SIMD math:

```csharp
private static float[] WeightedMeanPool(
    ReadOnlySpan<float> hiddenStates,
    ReadOnlySpan<long> attentionMask,
    int batchIdx, int seqLen, int hiddenDim)
{
    var embedding = new float[hiddenDim];
    float weightSum = 0;

    for (int s = 0; s < seqLen; s++)
    {
        if (attentionMask[batchIdx * seqLen + s] > 0)
        {
            float weight = (float)(s + 1) / seqLen;  // linear position weight
            int offset = (batchIdx * seqLen + s) * hiddenDim;
            ReadOnlySpan<float> tokenEmbed = hiddenStates.Slice(offset, hiddenDim);

            // Scale and accumulate
            var scaled = new float[hiddenDim];
            TensorPrimitives.Multiply(tokenEmbed, weight, scaled);
            TensorPrimitives.Add(embedding, scaled, embedding);
            weightSum += weight;
        }
    }

    if (weightSum > 0)
        TensorPrimitives.Divide(embedding, weightSum, embedding);

    return embedding;
}
```

## Supporting New Tokenizer Formats

**File to modify:** `src/MLNet.TextInference.Onnx/TextTokenizerEstimator.cs` — `LoadTokenizer()` method

`LoadTokenizer()` uses smart resolution to handle directories, HuggingFace config files, and direct vocab files. It currently supports:

| Source | Tokenizer | Vocab Files |
|--------|-----------|-------------|
| `tokenizer_class: "BertTokenizer"` | `BertTokenizer` | `vocab.txt` |
| `tokenizer_class: "XLMRobertaTokenizer"` | `LlamaTokenizer` | `sentencepiece.bpe.model`, `tokenizer.model` |
| `tokenizer_class: "LlamaTokenizer"` | `LlamaTokenizer` | `tokenizer.model` |
| `tokenizer_class: "GPT2Tokenizer"` | `BpeTokenizer` | `vocab.json` + `merges.txt` |
| `tokenizer_class: "RobertaTokenizer"` | `BpeTokenizer` | `vocab.json` + `merges.txt` |
| Direct `.txt` file | `BertTokenizer` | (the file itself) |
| Direct `.model` file | `LlamaTokenizer` | (the file itself) |

### Adding a new tokenizer type

To add support for a new `tokenizer_class` value (e.g., a future `MistralTokenizer`):

1. Add a case to the `LoadFromConfig()` switch:

```csharp
"MistralTokenizer" => LoadSentencePieceFromDirectory(directory),
```

2. If the new tokenizer uses a completely different format, add a new `LoadXxxFromDirectory()` method:

```csharp
private static Tokenizer LoadCustomFromDirectory(string directory)
{
    var vocabPath = Path.Combine(directory, "custom_vocab.dat");
    if (!File.Exists(vocabPath))
        throw new FileNotFoundException(
            $"Custom tokenizer requires custom_vocab.dat in '{directory}'.");
    // Construct and return the tokenizer
}
```

3. For tokenizer types not supported by `Microsoft.ML.Tokenizers`, users can always inject a pre-constructed instance via `TextTokenizerOptions.Tokenizer`.

### HuggingFace config resolution

When `TokenizerPath` points to a directory or `tokenizer_config.json`, the loader:
1. Reads `tokenizer_class` from the JSON (strips `"Fast"` suffix: `BertTokenizerFast` → `BertTokenizer`)
2. Dispatches to the appropriate factory method
3. Applies config properties (e.g., `do_lower_case` → `BertOptions.LowerCaseBeforeTokenization`)
4. Finds sibling vocab files in the same directory

This mirrors HuggingFace's `AutoTokenizer.from_pretrained()` pattern.

**Also update `ModelPackager`:** When saving, ensure all required vocab files are bundled for the tokenizer type being used.

## Using Different ONNX Models

The transform is model-agnostic. Any ONNX model that follows the sentence-transformer convention works:

### Model Requirements

| Input | Type | Shape | Required? |
|-------|------|-------|:---------:|
| `input_ids` | int64 | `[batch, seq_len]` | ✅ |
| `attention_mask` | int64 | `[batch, seq_len]` | ✅ |
| `token_type_ids` | int64 | `[batch, seq_len]` | ❌ auto-detected |

| Output | Type | Shape | Behavior |
|--------|------|-------|----------|
| `last_hidden_state` | float32 | `[batch, seq_len, hidden_dim]` | Pooling applied |
| `sentence_embedding` | float32 | `[batch, hidden_dim]` | Used directly |

### Non-Standard Models

If tensor names differ from the convention, use the override options:

```csharp
var options = new OnnxTextEmbeddingOptions
{
    ModelPath = "custom-model.onnx",
    TokenizerPath = "models/custom-model/",       // directory with tokenizer_config.json
    // OR: TokenizerPath = "vocab.txt",            // direct file still works
    InputIdsName = "tokens",              // override
    AttentionMaskName = "mask",           // override
    OutputTensorName = "embeddings",      // override
};
```

### Exporting Models to ONNX

If a model isn't available in ONNX format, export it with `optimum-cli`:

```bash
pip install optimum[exporters]
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./onnx-model/
```

This creates `model.onnx` along with `vocab.txt` (or `tokenizer.json` depending on the model).

## Path to Lazy Cursor-Based Evaluation

The modular transforms now implement **lazy cursor-based evaluation** via custom `IDataView`/cursor wrappers. Each transform's `Transform()` returns a wrapping `IDataView` — no data is materialized until a cursor iterates.

The scorer cursor uses **lookahead batching**: it reads `BatchSize` rows from the upstream tokenizer cursor, packs them into a single ONNX batch, runs inference once, then serves cached results one at a time. This gives batch throughput with lazy memory semantics.

## Using the Composable Pipeline

Instead of the convenience facade, you can compose transforms directly for full control over each step:

```csharp
var mlContext = new MLContext();

// Step 1: Tokenize
var tokenizer = mlContext.Transforms.TokenizeText(new TextTokenizerOptions
{
    TokenizerPath = "models/all-MiniLM-L6-v2/",  // directory — auto-detects from tokenizer_config.json
    InputColumnName = "Text",
    MaxTokenLength = 128
}).Fit(dataView);

var tokenized = tokenizer.Transform(dataView);

// Step 2: Score with ONNX (task-agnostic)
var scorer = mlContext.Transforms.ScoreOnnxTextModel(new OnnxTextModelScorerOptions
{
    ModelPath = "model.onnx",
    MaxTokenLength = 128,
    BatchSize = 32
}).Fit(tokenized);

var scored = scorer.Transform(tokenized);

// Step 3: Pool into embeddings
var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
{
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    HiddenDim = scorer.HiddenDim,
    IsPrePooled = scorer.HasPooledOutput,
    SequenceLength = scorer.HasPooledOutput ? 1 : 128
}).Fit(scored);

var pooled = pooler.Transform(scored);
```

### Why Compose Directly?

- **Swap pooling** without re-running inference (just change the pooler)
- **Inspect intermediates** (token IDs, attention masks, raw model output)
- **Reuse** the tokenizer + scorer for non-embedding tasks (classification, NER, etc.)

## Adding New Post-Processing Transforms

The composable architecture makes it easy to add new task-specific transforms that consume the scorer's raw output. See [How to Add a New Task](#how-to-add-a-new-task) above for the full step-by-step guide.

```csharp
// Embeddings
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(poolingOpts));

// Text classification
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.SoftmaxClassify(classOpts));

// Named entity recognition
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.DecodeNerEntities(nerOpts));
```

Each post-processing transform follows the same pattern:
1. Read the raw output column from the scorer
2. Apply task-specific logic (softmax, argmax, span extraction, etc.)
3. Output the result as a new column

## Using the Provider-Agnostic EmbeddingGeneratorEstimator

The `EmbeddingGeneratorEstimator` wraps any `IEmbeddingGenerator<string, Embedding<float>>` as an ML.NET pipeline step:

```csharp
// Works with any MEAI provider — only the generator construction changes
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, transformer);
    // OR: new OpenAIClient(...).AsEmbeddingGenerator()
    // OR: new OllamaEmbeddingGenerator(...)

var estimator = mlContext.Transforms.TextEmbedding(generator);
var transformer = estimator.Fit(dataView);
var result = transformer.Transform(dataView);
```

## Path to Approach D: Inside ML.NET

If this transform is adopted into `dotnet/machinelearning`, it would use the internal base classes for full ML.NET integration:

### Changes Required

1. **Subclass `RowToRowTransformerBase`** instead of implementing `ITransformer` directly:

```csharp
public sealed class OnnxTextEmbeddingTransformer
    : RowToRowTransformerBase  // ← private protected constructor accessible from within ML.NET
{
    private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema)
    {
        return new OnnxEmbeddingMapper(this, inputSchema);
    }
}
```

2. **Implement `MapperBase`** for lazy cursor-based evaluation:

```csharp
private sealed class OnnxEmbeddingMapper : MapperBase
{
    protected override Delegate MakeGetter(DataViewRow input, int iinfo, ...)
    {
        // Return a delegate that computes the embedding for the current row
        ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) => { ... };
        return getter;
    }
}
```

3. **Add `[LoadableClass]` attributes** for native save/load:

```csharp
[assembly: LoadableClass(typeof(OnnxTextEmbeddingTransformer), null,
    typeof(SignatureLoadModel), "ONNX Text Embedding", LoaderSignature)]
```

4. **Implement `SaveModel()`** with `ModelSaveContext`:

```csharp
private protected override void SaveModel(ModelSaveContext ctx)
{
    ctx.SetVersionInfo(GetVersionInfo());
    ctx.SaveBinaryStream("Model", w => { /* write ONNX bytes */ });
    ctx.SaveBinaryStream("Tokenizer", w => { /* write vocab bytes */ });
    ctx.Writer.Write(_options.MaxTokenLength);
    ctx.Writer.Write((int)_options.Pooling);
    // ... write all config
}
```

5. **Add static `Create()` factory** for loading:

```csharp
private static OnnxTextEmbeddingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
{
    ctx.CheckVersionInfo(GetVersionInfo());
    var modelBytes = ctx.LoadBinaryStream("Model");
    var vocabBytes = ctx.LoadBinaryStream("Tokenizer");
    // ... read config, construct transformer
}
```

This would give full `mlContext.Model.Save()` / `mlContext.Model.Load()` support and lazy cursor-based evaluation through `MapperBase`.

## Production Hardening

### GPU Support

GPU (CUDA) inference is now supported. The library references `Microsoft.ML.OnnxRuntime.Managed` (managed API only, no native binaries). Consuming applications choose their execution provider by referencing the appropriate native package (`Microsoft.ML.OnnxRuntime` for CPU, `Microsoft.ML.OnnxRuntime.Gpu` for CUDA).

Both `OnnxTextModelScorerOptions` and `OnnxTextEmbeddingOptions` expose `GpuDeviceId` and `FallbackToCpu` properties. The resolution order follows ML.NET convention:

1. Per-estimator `options.GpuDeviceId` (explicit override)
2. `mlContext.GpuDeviceId` (context-level default)
3. `null` = CPU

```csharp
// Context-level GPU
mlContext.GpuDeviceId = 0;

// Or per-estimator override with graceful fallback
var scorerOptions = new OnnxTextModelScorerOptions
{
    ModelPath = "model.onnx",
    GpuDeviceId = 0,
    FallbackToCpu = true,
};
```

**Future extensions** not yet implemented:
- **DirectML support** — `sessionOptions.AppendExecutionProvider_DML(deviceId)`
- **Multiple EP fallback chains** (e.g., TensorRT → CUDA → CPU)
- **Raw `SessionOptions` escape hatch** — exposing a user-supplied `SessionOptions` for advanced config (thread pools, memory arenas, IO Binding)

### Session Pooling

For high-throughput scenarios, a pool of `InferenceSession` instances can be used to parallelize inference across CPU threads:

```csharp
private readonly ObjectPool<InferenceSession> _sessionPool;
```

### Error Handling

The prototype has minimal error handling. Production code should:
- Validate tensor dimensions match expected shapes after ONNX run
- Handle OnnxRuntime exceptions (OOM, invalid model, etc.) gracefully
- Add telemetry / logging for inference timing
- Validate tokenizer output length doesn't exceed model's max sequence length

### Quantized Models

For deployment on edge devices, use quantized ONNX models (INT8 or FP16). The transform works without modification — OnnxRuntime handles quantized inference transparently. Just point `ModelPath` to the quantized `.onnx` file.

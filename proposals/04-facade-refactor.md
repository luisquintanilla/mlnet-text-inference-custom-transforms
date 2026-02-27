# Facade Refactoring: OnnxTextEmbeddingEstimator / OnnxTextEmbeddingTransformer

## Purpose

Refactor the existing `OnnxTextEmbeddingEstimator` and `OnnxTextEmbeddingTransformer` to internally compose the three modular transforms while preserving the exact same public API. No breaking changes for existing users.

## Files to Modify

| File | Change Type |
|------|-------------|
| `src/MLNet.TextInference.Onnx/OnnxTextEmbeddingEstimator.cs` | Major refactor |
| `src/MLNet.TextInference.Onnx/OnnxTextEmbeddingTransformer.cs` | Major refactor |
| `src/MLNet.TextInference.Onnx/OnnxTextEmbeddingOptions.cs` | Unchanged |

## What Stays the Same

- `OnnxTextEmbeddingOptions` — the user-facing options class is **unchanged**
- Public API of both estimator and transformer — **unchanged**
- `estimator.Fit(data)` returns `OnnxTextEmbeddingTransformer` — **unchanged**
- `transformer.Transform(data)` returns IDataView with Text + Embedding columns — **unchanged**
- `transformer.GenerateEmbeddings(texts)` returns `float[][]` — **unchanged**
- `transformer.Save(path)` / `OnnxTextEmbeddingTransformer.Load(mlContext, path)` — **unchanged**
- `transformer.EmbeddingDimension` — **unchanged**

## Refactored Estimator

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an OnnxTextEmbeddingTransformer.
/// Internally composes TextTokenizerEstimator → OnnxTextModelScorerEstimator → EmbeddingPoolingEstimator.
/// </summary>
public sealed class OnnxTextEmbeddingEstimator : IEstimator<OnnxTextEmbeddingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextEmbeddingOptions _options;

    public OnnxTextEmbeddingEstimator(MLContext mlContext, OnnxTextEmbeddingOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
        if (!File.Exists(options.TokenizerPath) && !Directory.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer path not found: {options.TokenizerPath}");
    }

    public OnnxTextEmbeddingTransformer Fit(IDataView input)
    {
        // Validate input schema has the text column
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // 1. Create and fit the tokenizer
        var tokenizerOptions = new TextTokenizerOptions
        {
            TokenizerPath = _options.TokenizerPath,
            InputColumnName = _options.InputColumnName,
            MaxTokenLength = _options.MaxTokenLength,
        };
        var tokenizerEstimator = new TextTokenizerEstimator(_mlContext, tokenizerOptions);
        var tokenizerTransformer = tokenizerEstimator.Fit(input);

        // 2. Create and fit the scorer
        // We need tokenized data to validate scorer input schema
        var tokenizedData = tokenizerTransformer.Transform(input);

        var scorerOptions = new OnnxTextModelScorerOptions
        {
            ModelPath = _options.ModelPath,
            MaxTokenLength = _options.MaxTokenLength,
            BatchSize = _options.BatchSize,
            // Use auto-discovery overrides from the user's options
            InputIdsTensorName = _options.InputIdsName,
            AttentionMaskTensorName = _options.AttentionMaskName,
            TokenTypeIdsTensorName = _options.TokenTypeIdsName,
            OutputTensorName = _options.OutputTensorName,
        };
        var scorerEstimator = new OnnxTextModelScorerEstimator(_mlContext, scorerOptions);
        var scorerTransformer = scorerEstimator.Fit(tokenizedData);

        // 3. Create and fit the pooler (auto-configured from scorer metadata)
        var scoredData = scorerTransformer.Transform(tokenizedData);

        var poolingOptions = new EmbeddingPoolingOptions
        {
            OutputColumnName = _options.OutputColumnName,
            Pooling = _options.Pooling,
            Normalize = _options.Normalize,
            // Auto-configured from scorer metadata:
            HiddenDim = scorerTransformer.HiddenDim,
            IsPrePooled = scorerTransformer.HasPooledOutput,
            SequenceLength = scorerTransformer.HasPooledOutput ? 0 : _options.MaxTokenLength,
        };
        var poolingEstimator = new EmbeddingPoolingEstimator(_mlContext, poolingOptions);
        var poolingTransformer = poolingEstimator.Fit(scoredData);

        // Return composite transformer
        return new OnnxTextEmbeddingTransformer(
            _mlContext, _options,
            tokenizerTransformer, scorerTransformer, poolingTransformer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Same behavior as current: validates input column, probes model
        // for embedding dimension, returns schema with output column.
        // Can delegate through the three sub-estimators' GetOutputSchema.
    }
}
```

## Refactored Transformer

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that generates text embeddings using a local ONNX model.
/// Internally composes tokenization → ONNX inference → pooling using three sub-transforms.
/// </summary>
public sealed class OnnxTextEmbeddingTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextEmbeddingOptions _options;

    // Internal sub-transforms
    private readonly TextTokenizerTransformer _tokenizer;
    private readonly OnnxTextModelScorerTransformer _scorer;
    private readonly EmbeddingPoolingTransformer _pooler;

    public bool IsRowToRowMapper => true;

    internal OnnxTextEmbeddingOptions Options => _options;
    public int EmbeddingDimension => _scorer.HiddenDim;

    // Expose sub-transforms for advanced users
    internal TextTokenizerTransformer Tokenizer => _tokenizer;
    internal OnnxTextModelScorerTransformer Scorer => _scorer;
    internal EmbeddingPoolingTransformer Pooler => _pooler;

    internal OnnxTextEmbeddingTransformer(
        MLContext mlContext,
        OnnxTextEmbeddingOptions options,
        TextTokenizerTransformer tokenizer,
        OnnxTextModelScorerTransformer scorer,
        EmbeddingPoolingTransformer pooler)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
        _scorer = scorer;
        _pooler = pooler;
    }

    /// <summary>
    /// ML.NET face: chains the three sub-transforms via IDataView.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var tokenized = _tokenizer.Transform(input);
        var scored = _scorer.Transform(tokenized);
        var pooled = _pooler.Transform(scored);
        return pooled;
    }

    /// <summary>
    /// Direct face: generates embeddings without IDataView overhead.
    /// Used by the MEAI wrapper (OnnxEmbeddingGenerator).
    /// Chains the three sub-transforms' direct faces.
    /// </summary>
    internal float[][] GenerateEmbeddings(IReadOnlyList<string> texts)
    {
        if (texts.Count == 0)
            return [];

        // Direct face: no IDataView, no materialization, no schema overhead
        // This is the fast path for MEAI usage.

        var allEmbeddings = new List<float[]>(texts.Count);
        int batchSize = _options.BatchSize;

        for (int start = 0; start < texts.Count; start += batchSize)
        {
            int count = Math.Min(batchSize, texts.Count - start);
            var batchTexts = new List<string>(count);
            for (int i = start; i < start + count; i++)
                batchTexts.Add(texts[i]);

            // Chain direct faces
            var tokenized = _tokenizer.Tokenize(batchTexts);
            var scored = _scorer.Score(tokenized);

            // For the direct path, we also need attention masks for pooling
            var attentionMasks = tokenized.AttentionMasks;
            var embeddings = _pooler.Pool(scored, attentionMasks);

            allEmbeddings.AddRange(embeddings);
        }

        return [.. allEmbeddings];
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        // Delegate through the chain
        var tokSchema = _tokenizer.GetOutputSchema(inputSchema);
        var scorerSchema = _scorer.GetOutputSchema(tokSchema);
        return _pooler.GetOutputSchema(scorerSchema);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "ML.NET native save is not supported. Use transformer.Save(path) instead.");

    /// <summary>
    /// Saves the transformer to a self-contained zip file.
    /// </summary>
    public void Save(string path) => ModelPackager.Save(this, path);

    /// <summary>
    /// Loads a transformer from a saved zip file.
    /// </summary>
    public static OnnxTextEmbeddingTransformer Load(MLContext mlContext, string path)
        => ModelPackager.Load(mlContext, path);

    public void Dispose()
    {
        _scorer.Dispose();
    }
}
```

## Key Design Decisions

### GenerateEmbeddings uses direct faces for zero overhead

The MEAI path (`GenerateEmbeddings()`) chains the three transforms' **direct faces** (`Tokenize()` → `Score()` → `Pool()`), never touching IDataView. This preserves the current performance characteristics: batch-oriented, minimal allocation, SIMD-accelerated pooling.

### Transform() chains wrapping IDataViews (lazy)

The ML.NET pipeline path (`Transform()`) chains the three transforms' **IDataView faces**. Each `Transform()` returns a wrapping IDataView — no materialization occurs until a consumer iterates the cursor. The cursor chain (PoolerCursor → ScorerCursor → TokenizerCursor → InputCursor) processes data lazily with the scorer using lookahead batching for ONNX throughput.

This means `Transform()` is **O(1) in time and memory** — it just wraps. All computation happens on cursor iteration. Peak memory is ~6 MB per batch regardless of dataset size.

### Sub-transforms are exposed internally

`Tokenizer`, `Scorer`, and `Pooler` properties are `internal` — visible within the assembly but not to external consumers. This allows:
- `ModelPackager` to access sub-transform state for save/load
- Advanced internal composition
- Testing individual sub-transforms

### No behavioral changes

The same math, same batching, same auto-discovery, same tokenizer loading. The refactoring is purely structural.

## Impact on ModelPackager

`ModelPackager.Save()` doesn't change its output format — it still creates the same zip with `model.onnx`, `vocab.txt`, `config.json`, `manifest.json`. It accesses the model path and tokenizer path via `transformer.Options` (which is the original `OnnxTextEmbeddingOptions`, unchanged).

`ModelPackager.Load()` still reconstructs via `OnnxTextEmbeddingEstimator.Fit()`, which now internally creates the three sub-transforms. The saved config contains all the information needed to reconstruct.

## Migration Path

This is an internal refactoring. External consumers see:

| API | Before | After | Change? |
|-----|--------|-------|---------|
| `new OnnxTextEmbeddingEstimator(mlContext, options)` | ✅ | ✅ | No |
| `estimator.Fit(data)` | ✅ Returns transformer | ✅ Returns transformer | No |
| `transformer.Transform(data)` | ✅ Text + Embedding | ✅ Text + Embedding | No |
| `transformer.EmbeddingDimension` | ✅ int | ✅ int | No |
| `transformer.Save(path)` | ✅ | ✅ | No |
| `OnnxTextEmbeddingTransformer.Load(ctx, path)` | ✅ | ✅ | No |
| `new OnnxEmbeddingGenerator(mlContext, transformer)` | ✅ | ✅ | No |

## Acceptance Criteria

1. All existing public API behavior is preserved
2. `Transform()` produces identical output (Text + Embedding columns, same values)
3. `GenerateEmbeddings()` produces identical output (same float[][] values)
4. `Save()` / `Load()` round-trip works
5. `OnnxEmbeddingGenerator` continues to work unchanged
6. Sample program (`BasicUsage/Program.cs`) runs identically

# MEAI Integration: EmbeddingGeneratorEstimator + OnnxEmbeddingGenerator Refactoring

## Purpose

Two related changes:

1. **New**: `EmbeddingGeneratorEstimator` / `EmbeddingGeneratorTransformer` — a provider-agnostic ML.NET transform that wraps ANY `IEmbeddingGenerator<string, Embedding<float>>`. Allows using OpenAI, Azure OpenAI, Ollama, or any MEAI provider as an ML.NET pipeline step.

2. **Refactor**: `OnnxEmbeddingGenerator` to use the three sub-transforms' direct faces internally, making it composable at the MEAI level.

## Relationship to the Three-Way Split

The three-way split (tokenizer → scorer → pooler) decomposes the ONNX pipeline at the **ML.NET level** for composable pipelines with inspectable intermediates.

The `EmbeddingGeneratorEstimator` wraps any `IEmbeddingGenerator` at the **MEAI level** for provider-agnostic embedding generation within ML.NET pipelines.

These are complementary, not competing:

```
                        ML.NET Pipeline Users
                     ┌──────────┴──────────┐
                     │                     │
           Want ONNX-specific        Want provider-agnostic
           composable pipeline       embedding generation
                     │                     │
                     ▼                     ▼
          TokenizerTransformer    EmbeddingGeneratorEstimator
           → ScorerTransformer     wraps IEmbeddingGenerator
            → PoolingTransformer       │
                     │            ┌────┼────┐
                     │            │    │    │
                     │          ONNX OpenAI Ollama
                     │            │
                     ▼            ▼
              OnnxTextEmbedding  OnnxEmbeddingGenerator
              Estimator (facade)  (uses direct faces)
```

## Files to Create

| File | Contents |
|------|----------|
| `src/MLNet.TextInference.Onnx/EmbeddingGeneratorEstimator.cs` | Estimator + transformer + options |

## Files to Modify

| File | Change |
|------|--------|
| `src/MLNet.TextInference.Onnx/OnnxEmbeddingGenerator.cs` | Refactor to use sub-transform direct faces |
| `src/MLNet.TextInference.Onnx/MLContextExtensions.cs` | Add extension method |

## EmbeddingGeneratorEstimator

### Options

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the provider-agnostic embedding generator transform.
/// </summary>
public class EmbeddingGeneratorOptions
{
    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output embedding column. Default: "Embedding".</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>
    /// Batch size for embedding generation. Default: 32.
    /// The generator may further subdivide batches internally.
    /// </summary>
    public int BatchSize { get; set; } = 32;
}
```

### Estimator

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that wraps any IEmbeddingGenerator to produce embeddings within a pipeline.
/// Provider-agnostic — works with ONNX, OpenAI, Azure OpenAI, Ollama, or any MEAI implementation.
/// </summary>
public sealed class EmbeddingGeneratorEstimator : IEstimator<EmbeddingGeneratorTransformer>
{
    private readonly MLContext _mlContext;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _generator;
    private readonly EmbeddingGeneratorOptions _options;

    public EmbeddingGeneratorEstimator(
        MLContext mlContext,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        EmbeddingGeneratorOptions? options = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _options = options ?? new EmbeddingGeneratorOptions();
    }

    public EmbeddingGeneratorTransformer Fit(IDataView input)
    {
        // Validate input schema has text column
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        return new EmbeddingGeneratorTransformer(_mlContext, _generator, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Validate input column
        // Add output column (Vector<float>)
        // Dimension may not be known until runtime for some providers
        // Use the generator's metadata if available
        var dim = _generator.Metadata.Dimensions;
        // ...
    }
}
```

### Transformer

The `EmbeddingGeneratorTransformer` uses **eager evaluation** (not the lazy IDataView/cursor pattern used by the three core transforms). This is deliberate:

- `IEmbeddingGenerator.GenerateAsync()` is inherently batch-oriented — there's no concept of a per-row cursor
- Remote providers (OpenAI, Azure) want batch calls, not per-row calls
- The async/sync bridge (`GetAwaiter().GetResult()`) is best amortized over batches
- For ONNX specifically, users should prefer the three-way composable pipeline for lazy evaluation

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that generates embeddings using any IEmbeddingGenerator.
/// </summary>
public sealed class EmbeddingGeneratorTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _generator;
    private readonly EmbeddingGeneratorOptions _options;

    public bool IsRowToRowMapper => true;

    internal IEmbeddingGenerator<string, Embedding<float>> Generator => _generator;

    internal EmbeddingGeneratorTransformer(
        MLContext mlContext,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        EmbeddingGeneratorOptions options)
    {
        _mlContext = mlContext;
        _generator = generator;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        // 1. Read text column
        var texts = ReadTextColumn(input);
        if (texts.Count == 0)
            return BuildOutputDataView(texts, []);

        // 2. Generate embeddings in batches
        var allEmbeddings = new List<float[]>(texts.Count);

        for (int start = 0; start < texts.Count; start += _options.BatchSize)
        {
            int count = Math.Min(_options.BatchSize, texts.Count - start);
            var batchTexts = texts.GetRange(start, count);

            // MEAI is async, ML.NET is sync — must block
            var result = _generator.GenerateAsync(batchTexts).GetAwaiter().GetResult();

            foreach (var embedding in result)
            {
                allEmbeddings.Add(embedding.Vector.ToArray());
            }
        }

        // 3. Build output
        return BuildOutputDataView(texts, allEmbeddings);
    }

    // Standard plumbing methods...

    private List<string> ReadTextColumn(IDataView dataView)
    {
        // Same pattern as OnnxTextEmbeddingTransformer.ReadTextColumn()
    }

    private IDataView BuildOutputDataView(List<string> texts, List<float[]> embeddings)
    {
        // Same pattern as current implementation
    }
}
```

## The Async/Sync Bridge

`IEmbeddingGenerator.GenerateAsync()` is async. ML.NET's `ITransformer.Transform()` is sync. This is an inherent tension:

| Provider | GenerateAsync behavior | .GetAwaiter().GetResult() impact |
|----------|----------------------|----------------------------------|
| **OnnxEmbeddingGenerator** | Returns `Task.FromResult()` (synchronous) | Zero impact — unwraps instantly |
| **OpenAI / Azure** | Makes HTTP call (truly async) | Blocks a thread pool thread |
| **Ollama (local)** | Makes local HTTP call | Blocks briefly |

For ONNX (the primary use case), there's no problem. For remote providers used in ML.NET pipelines, thread blocking is acceptable because:
1. ML.NET pipelines are inherently synchronous (batch processing)
2. The alternative (async ITransformer) doesn't exist in ML.NET
3. Users who need async should use `IEmbeddingGenerator` directly, not through ML.NET

## OnnxEmbeddingGenerator Refactoring

The existing `OnnxEmbeddingGenerator` is refactored minimally. It already delegates to `_transformer.GenerateEmbeddings()`. After the facade refactoring, `GenerateEmbeddings()` uses the direct faces internally — so the generator automatically benefits from the modularization without changing its own code.

```csharp
// OnnxEmbeddingGenerator.GenerateAsync — UNCHANGED
public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
    IEnumerable<string> values,
    EmbeddingGenerationOptions? options = null,
    CancellationToken cancellationToken = default)
{
    cancellationToken.ThrowIfCancellationRequested();

    var textList = values as IReadOnlyList<string> ?? values.ToList();
    var embeddings = _transformer.GenerateEmbeddings(textList);
    // ↑ This now internally chains: Tokenize() → Score() → Pool()
    //   via direct faces — same performance as monolith

    var result = new GeneratedEmbeddings<Embedding<float>>(
        embeddings.Select(e => new Embedding<float>(e)));

    return Task.FromResult(result);
}
```

## MLContextExtensions Updates

```csharp
public static class MLContextExtensions
{
    // Existing convenience method (unchanged)
    public static OnnxTextEmbeddingEstimator OnnxTextEmbedding(
        this TransformsCatalog catalog,
        OnnxTextEmbeddingOptions options)
    {
        return new OnnxTextEmbeddingEstimator(catalog.GetMLContext(), options);
    }

    // NEW: Provider-agnostic embedding transform
    public static EmbeddingGeneratorEstimator TextEmbedding(
        this TransformsCatalog catalog,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        EmbeddingGeneratorOptions? options = null)
    {
        return new EmbeddingGeneratorEstimator(catalog.GetMLContext(), generator, options);
    }

    // NEW: Individual composable transforms
    public static TextTokenizerEstimator TokenizeText(
        this TransformsCatalog catalog,
        TextTokenizerOptions options)
    {
        return new TextTokenizerEstimator(catalog.GetMLContext(), options);
    }

    public static OnnxTextModelScorerEstimator ScoreOnnxTextModel(
        this TransformsCatalog catalog,
        OnnxTextModelScorerOptions options)
    {
        return new OnnxTextModelScorerEstimator(catalog.GetMLContext(), options);
    }

    public static EmbeddingPoolingEstimator PoolEmbedding(
        this TransformsCatalog catalog,
        EmbeddingPoolingOptions options)
    {
        return new EmbeddingPoolingEstimator(catalog.GetMLContext(), options);
    }
}
```

## Save/Load for EmbeddingGeneratorTransformer

`IEmbeddingGenerator` has no save/load contract. The `EmbeddingGeneratorTransformer` cannot be saved:

```csharp
void ICanSaveModel.Save(ModelSaveContext ctx)
    => throw new NotSupportedException(
        "Cannot save a provider-agnostic embedding transform. " +
        "Use OnnxTextEmbeddingTransformer.Save() for ONNX-backed transforms.");
```

This is by design. Remote providers (OpenAI) can't be "saved" — they need API keys and endpoints that are configuration, not model state. Users who need save/load use the ONNX-specific facade.

## Acceptance Criteria

1. `EmbeddingGeneratorEstimator` accepts any `IEmbeddingGenerator<string, Embedding<float>>`
2. `Transform()` produces correct embeddings via the generator
3. Works with `OnnxEmbeddingGenerator` (synchronous)
4. Works with hypothetical remote generators (async, blocking is acceptable)
5. `OnnxEmbeddingGenerator` continues to work unchanged
6. New extension methods are added to `MLContextExtensions`
7. Save/load throws `NotSupportedException` with clear message

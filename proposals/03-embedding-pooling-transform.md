# Transform 3: EmbeddingPoolingEstimator / EmbeddingPoolingTransformer

## Purpose

Applies pooling and normalization to raw model output to produce a fixed-length embedding vector. This is the **task-specific post-processing step** for embedding generation. It is one of potentially many post-processing transforms that can sit downstream of `OnnxTextModelScorerTransformer`.

## Why a Separate Transform?

1. **Swappability**: Users can swap pooling strategy without re-running inference. Re-inference is expensive (~10ms per batch); re-pooling is ~0.01ms.
2. **Composability**: Mean pooling, CLS pooling, and max pooling produce different embedding spaces. Being able to switch at the pipeline level is valuable for experimentation.
3. **Separation of concerns**: L2 normalization is a well-defined operation that ML.NET already has (`LpNormNormalizingEstimator`). Our transform bundles it as a convenience, but users could use ML.NET's built-in normalizer instead.
4. **Future post-processing**: Other post-processing transforms (Matryoshka truncation, binary quantization, PCA reduction) can be added alongside this one.

## Files to Create

| File | Contents |
|------|----------|
| `src/MLNet.TextInference.Onnx/EmbeddingPoolingEstimator.cs` | Estimator + options class |
| `src/MLNet.TextInference.Onnx/EmbeddingPoolingTransformer.cs` | Transformer |

## Options Class

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the embedding pooling transform.
/// Reduces raw model output to a fixed-length embedding vector.
/// </summary>
public class EmbeddingPoolingOptions
{
    /// <summary>
    /// Name of the input column containing raw model output.
    /// For unpooled models: VBuffer&lt;float&gt; of length seqLen × hiddenDim.
    /// For pre-pooled models: VBuffer&lt;float&gt; of length hiddenDim.
    /// Default: "RawOutput".
    /// </summary>
    public string InputColumnName { get; set; } = "RawOutput";

    /// <summary>
    /// Name of the attention mask column. Required for mean and max pooling
    /// (to exclude padding tokens). Not needed for CLS pooling or pre-pooled input.
    /// Default: "AttentionMask".
    /// </summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Name of the output embedding column. Default: "Embedding".</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>
    /// Pooling strategy for reducing per-token outputs to a single vector.
    /// Ignored when IsPrePooled is true.
    /// Default: MeanPooling.
    /// </summary>
    public PoolingStrategy Pooling { get; set; } = PoolingStrategy.MeanPooling;

    /// <summary>Whether to L2-normalize the output embeddings. Default: true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>
    /// Hidden dimension of the model output.
    /// When used via the facade, this is auto-set from scorer metadata.
    /// When used standalone, must be specified by the user.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Sequence length of the unpooled model output.
    /// Only needed for unpooled models. When used via the facade, auto-set from scorer.
    /// </summary>
    public int SequenceLength { get; set; }

    /// <summary>
    /// Whether the input is already pooled (e.g., sentence_embedding output).
    /// When true, only normalization is applied (pooling strategy is ignored).
    /// When used via the facade, auto-set from scorer metadata.
    /// Default: false.
    /// </summary>
    public bool IsPrePooled { get; set; }
}
```

## Estimator

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an EmbeddingPoolingTransformer.
/// Trivial estimator — validates schema and passes configuration through.
/// </summary>
public sealed class EmbeddingPoolingEstimator : IEstimator<EmbeddingPoolingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly EmbeddingPoolingOptions _options;

    public EmbeddingPoolingEstimator(MLContext mlContext, EmbeddingPoolingOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (options.HiddenDim <= 0)
            throw new ArgumentException("HiddenDim must be positive.", nameof(options));

        if (!options.IsPrePooled && options.SequenceLength <= 0)
            throw new ArgumentException(
                "SequenceLength must be positive for unpooled models.", nameof(options));
    }

    /// <summary>
    /// Creates an EmbeddingPoolingEstimator that auto-configures from scorer metadata.
    /// Used internally by the facade.
    /// </summary>
    internal EmbeddingPoolingEstimator(
        MLContext mlContext,
        EmbeddingPoolingOptions options,
        OnnxTextModelScorerTransformer scorer)
        : this(mlContext, ConfigureFromScorer(options, scorer))
    {
    }

    private static EmbeddingPoolingOptions ConfigureFromScorer(
        EmbeddingPoolingOptions options,
        OnnxTextModelScorerTransformer scorer)
    {
        // Auto-fill dimensions from scorer metadata
        options.HiddenDim = scorer.HiddenDim;
        options.IsPrePooled = scorer.HasPooledOutput;
        if (!scorer.HasPooledOutput)
            options.SequenceLength = scorer.Options.MaxTokenLength;
        return options;
    }

    public EmbeddingPoolingTransformer Fit(IDataView input)
    {
        // Validate input schema has the raw output column
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // Validate attention mask column exists (if needed for pooling)
        if (!_options.IsPrePooled && _options.Pooling != PoolingStrategy.ClsToken)
        {
            var maskCol = input.Schema.GetColumnOrNull(_options.AttentionMaskColumnName);
            if (maskCol == null)
                throw new ArgumentException(
                    $"Input schema does not contain column '{_options.AttentionMaskColumnName}'. " +
                    $"Required for {_options.Pooling} pooling.");
        }

        return new EmbeddingPoolingTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Validate input column exists
        // Add output column: Vector<float> of size HiddenDim
        var result = inputSchema.ToDictionary(x => x.Name);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var outputCol = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);
        result[_options.OutputColumnName] = outputCol;

        return new SchemaShape(result.Values);
    }
}
```

## Transformer

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that pools raw model output into fixed-length embeddings.
/// Supports mean, CLS, and max pooling, plus optional L2 normalization.
///
/// Lazy evaluation: Transform() returns a wrapping IDataView. Pooling is computed
/// per-row as the cursor advances. Pooling is cheap (~microseconds per row),
/// so no batching is needed.
/// </summary>
public sealed class EmbeddingPoolingTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly EmbeddingPoolingOptions _options;

    public bool IsRowToRowMapper => true;

    internal EmbeddingPoolingOptions Options => _options;
    public int EmbeddingDimension => _options.HiddenDim;

    internal EmbeddingPoolingTransformer(
        MLContext mlContext,
        EmbeddingPoolingOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    /// <summary>
    /// ML.NET face: returns a wrapping IDataView. No computation happens here.
    /// Pooling occurs lazily per-row when a cursor iterates.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        return new PoolerDataView(input, _options);
    }

    /// <summary>
    /// Direct face: pool raw outputs without IDataView overhead.
    /// Used by the facade and MEAI generator.
    /// </summary>
    internal float[][] Pool(float[][] rawOutputs, long[][]? attentionMasks)
    {
        if (_options.IsPrePooled)
        {
            if (_options.Normalize)
            {
                for (int i = 0; i < rawOutputs.Length; i++)
                    L2Normalize(rawOutputs[i]);
            }
            return rawOutputs;
        }

        int hiddenDim = _options.HiddenDim;
        int seqLen = _options.SequenceLength;
        var embeddings = new float[rawOutputs.Length][];

        for (int i = 0; i < rawOutputs.Length; i++)
        {
            ReadOnlySpan<float> hiddenStates = rawOutputs[i];
            ReadOnlySpan<long> mask = attentionMasks![i];

            embeddings[i] = _options.Pooling switch
            {
                PoolingStrategy.MeanPooling =>
                    EmbeddingPooling.Pool(hiddenStates, mask, 1, seqLen, hiddenDim,
                        PoolingStrategy.MeanPooling, false)[0],
                PoolingStrategy.ClsToken =>
                    EmbeddingPooling.Pool(hiddenStates, mask, 1, seqLen, hiddenDim,
                        PoolingStrategy.ClsToken, false)[0],
                PoolingStrategy.MaxPooling =>
                    EmbeddingPooling.Pool(hiddenStates, mask, 1, seqLen, hiddenDim,
                        PoolingStrategy.MaxPooling, false)[0],
                _ => throw new ArgumentOutOfRangeException()
            };

            if (_options.Normalize)
                L2Normalize(embeddings[i]);
        }

        return embeddings;
    }

    /// <summary>
    /// Pools a single row's raw output. Used by the cursor.
    /// </summary>
    internal float[] PoolSingleRow(ReadOnlySpan<float> rawOutput, ReadOnlySpan<long> attentionMask)
    {
        float[] embedding;

        if (_options.IsPrePooled)
        {
            embedding = rawOutput.ToArray();
        }
        else
        {
            embedding = EmbeddingPooling.Pool(
                rawOutput, attentionMask, 1, _options.SequenceLength, _options.HiddenDim,
                _options.Pooling, false)[0];
        }

        if (_options.Normalize)
            L2Normalize(embedding);

        return embedding;
    }

    private static void L2Normalize(Span<float> embedding)
    {
        float norm = TensorPrimitives.Norm(embedding);
        if (norm > 0)
            TensorPrimitives.Divide(embedding, norm, embedding);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single, _options.HiddenDim));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();
    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}
```

## Lazy IDataView and Cursor

The pooler cursor is the simplest of the three — it reads one row from the upstream scorer
cursor, applies pooling math, and returns the result. Pooling is cheap (~microseconds),
so no batching or caching is needed.

```csharp
/// <summary>
/// Wrapping IDataView that adds the pooled embedding column to the upstream schema.
/// </summary>
internal sealed class PoolerDataView : IDataView
{
    private readonly IDataView _input;
    private readonly EmbeddingPoolingOptions _options;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal PoolerDataView(IDataView input, EmbeddingPoolingOptions options)
    {
        _input = input;
        _options = options;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);
        builder.AddColumn(options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single, options.HiddenDim));
        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        // Need raw output + attention mask from upstream for pooling
        var upstreamCols = new List<DataViewSchema.Column>();
        foreach (var col in columnsNeeded)
        {
            var inputCol = _input.Schema.GetColumnOrNull(col.Name);
            if (inputCol != null)
                upstreamCols.Add(inputCol.Value);
        }

        // Always need raw output and attention mask for pooling computation
        upstreamCols.Add(_input.Schema[_options.InputColumnName]);
        if (!_options.IsPrePooled && _options.Pooling != PoolingStrategy.ClsToken)
            upstreamCols.Add(_input.Schema[_options.AttentionMaskColumnName]);

        var inputCursor = _input.GetRowCursor(upstreamCols.Distinct(), rand);
        return new PoolerCursor(this, inputCursor, _options);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor that pools one row at a time from the upstream scorer cursor.
/// No batching needed — pooling is a pure math operation on a single row's data.
///
/// Unlike the scorer cursor, this cursor does NOT need to cache upstream values
/// because it processes row-by-row in lockstep with the upstream cursor.
/// Passthrough columns can be delegated directly to the upstream cursor.
/// </summary>
internal sealed class PoolerCursor : DataViewRowCursor
{
    private readonly PoolerDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly EmbeddingPoolingOptions _options;

    private float[]? _currentEmbedding;

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _inputCursor.Position;
    public override long Batch => _inputCursor.Batch;

    internal PoolerCursor(
        PoolerDataView parent,
        DataViewRowCursor inputCursor,
        EmbeddingPoolingOptions options)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _options = options;
    }

    public override bool MoveNext()
    {
        if (!_inputCursor.MoveNext())
            return false;

        // Read raw output from upstream
        var rawOutputCol = _inputCursor.Schema[_options.InputColumnName];
        var rawOutputGetter = _inputCursor.GetGetter<VBuffer<float>>(rawOutputCol);
        VBuffer<float> rawOutputBuffer = default;
        rawOutputGetter(ref rawOutputBuffer);

        // Read attention mask (if needed)
        long[]? attentionMask = null;
        if (!_options.IsPrePooled && _options.Pooling != PoolingStrategy.ClsToken)
        {
            var maskCol = _inputCursor.Schema[_options.AttentionMaskColumnName];
            var maskGetter = _inputCursor.GetGetter<VBuffer<long>>(maskCol);
            VBuffer<long> maskBuffer = default;
            maskGetter(ref maskBuffer);
            attentionMask = maskBuffer.DenseValues().ToArray();
        }

        // Pool this single row
        ReadOnlySpan<float> rawOutput = rawOutputBuffer.DenseValues().ToArray();
        ReadOnlySpan<long> mask = attentionMask ?? [];

        if (_options.IsPrePooled)
        {
            _currentEmbedding = rawOutput.ToArray();
            if (_options.Normalize)
            {
                float norm = TensorPrimitives.Norm(_currentEmbedding);
                if (norm > 0)
                    TensorPrimitives.Divide(_currentEmbedding, norm, _currentEmbedding);
            }
        }
        else
        {
            _currentEmbedding = EmbeddingPooling.Pool(
                rawOutput, mask, 1, _options.SequenceLength, _options.HiddenDim,
                _options.Pooling, _options.Normalize)[0];
        }

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        // For the embedding output column, return the pooled result
        if (column.Name == _options.OutputColumnName)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            {
                var editor = VBufferEditor.Create(ref value, _currentEmbedding!.Length);
                _currentEmbedding.AsSpan().CopyTo(editor.Values);
                value = editor.Commit();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // For all passthrough columns, delegate directly to upstream cursor
        // (safe because we process in lockstep — no lookahead)
        var inputCol = _inputCursor.Schema.GetColumnOrNull(column.Name);
        if (inputCol != null)
            return _inputCursor.GetGetter<TValue>(inputCol.Value);

        throw new InvalidOperationException($"Unknown column: {column.Name}");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => _inputCursor.GetIdGetter();

    public override bool IsColumnActive(DataViewSchema.Column column) => true;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _inputCursor.Dispose();
        base.Dispose(disposing);
    }
}
```

### Why the Pooler Cursor Is Simpler Than the Scorer Cursor

The scorer cursor needs lookahead batching (read N rows ahead, cache results). This creates
a mismatch between the upstream cursor position and the downstream position, requiring
upstream column value caching.

The pooler cursor processes **one row at a time** in lockstep with upstream. When `MoveNext()`
is called, it moves the upstream cursor forward one row, reads the raw output and attention
mask, pools them, and stores the result. Passthrough columns can be delegated directly to the
upstream cursor because it's pointing at the same row. No caching needed.

## Code to Extract From Existing Files

| Source | What to Extract | Target |
|--------|----------------|--------|
| `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 173-183 | Pooling dispatch logic | `EmbeddingPoolingTransformer.Pool()` and `PoolerCursor.MoveNext()` |
| `EmbeddingPooling.cs` | Unchanged — continues to provide the static math | Used by `EmbeddingPoolingTransformer` and `PoolerCursor` |

## Relationship to EmbeddingPooling.cs

`EmbeddingPooling.cs` (the existing static class) is **not modified**. It provides the SIMD-accelerated math (`MeanPool`, `ClsPool`, `MaxPool`, `L2Normalize`). Both the direct face `Pool()` and the lazy cursor call into it.

The `EmbeddingPooling` static class methods currently take batch-oriented parameters (`batchSize`, `batchIdx`). When called from the per-row cursor, we pass `batchSize=1, batchIdx=0`. This works correctly but is slightly wasteful (the batch loop runs once). A minor optimization would be to add per-row convenience methods to `EmbeddingPooling`, but it's not necessary for correctness.

## Approach D Migration Notes

When migrating to ML.NET, this transform becomes:

```
EmbeddingPoolingTransformer : OneToOneTransformerBase
  Mapper : OneToOneMapperBase
    MakeGetter() → pooling logic from PoolerCursor.MoveNext()
    GetOutputColumnsCore() → embedding column definition
    SaveModel() → serialize pooling options
```

The `PoolerDataView` and `PoolerCursor` get **deleted** — `OneToOneTransformerBase` handles
cursor creation, column passthrough, and schema propagation. The pooling math in `MakeGetter()`
is identical to the cursor's `MoveNext()` logic.

## Future Post-Processing Transforms

The embedding pooling transform is the first in what could be a family of post-processing transforms:

| Transform | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **EmbeddingPoolingTransformer** (this) | Mean/CLS/Max + normalize | Raw hidden states | `float[hiddenDim]` |
| **MatryoshkaTruncationTransformer** | Truncate to N dims | `float[hiddenDim]` | `float[N]` where N < hiddenDim |
| **BinaryQuantizationTransformer** | 1-bit per dim | `float[hiddenDim]` | `byte[hiddenDim/8]` |
| **SoftmaxClassificationTransformer** | Logits → probabilities | `float[numClasses]` | `float[numClasses]` |
| **NerDecodingTransformer** | Per-token logits → entities | `float[seqLen × numLabels]` | Entity spans |

All share the same lazy cursor pattern: wrap upstream IDataView, compute per-row in `MoveNext()`, delegate passthrough columns to upstream cursor. Each would be ~150 lines of boilerplate that gets eliminated when migrating to Approach D.

## Acceptance Criteria

1. `EmbeddingPoolingEstimator` validates HiddenDim and SequenceLength
2. Auto-configures from `OnnxTextModelScorerTransformer` metadata when used via facade
3. `Fit()` validates that input and attention mask columns exist
4. `Transform()` returns a wrapping IDataView (no materialization)
5. Cursor computes pooling per-row on demand (mean, CLS, or max)
6. Pre-pooled pass-through works (only normalizes)
7. L2 normalization is optional and works correctly
8. Passthrough columns are delegated directly to upstream cursor (zero copy)
9. `Pool()` (direct face) returns the same results without IDataView overhead
10. Embedding dimension is exposed via `EmbeddingDimension` property
11. Memory usage is O(1) per row, not O(N) for N total rows

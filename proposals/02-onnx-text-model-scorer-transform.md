# Transform 2: OnnxTextModelScorerEstimator / OnnxTextModelScorerTransformer

## Purpose

Runs ONNX inference on tokenized text inputs. This is the **universal second step** for any transformer-based ONNX model task. It takes token columns (produced by `TextTokenizerTransformer`) and outputs the raw model tensor. It is intentionally **task-agnostic** — it doesn't know whether the output will be pooled into embeddings, softmaxed into class probabilities, or decoded into entity spans.

## Why "OnnxTextModelScorer" and not "OnnxScorer"?

- ML.NET already has `OnnxScoringEstimator` / `OnnxTransformer` in `Microsoft.ML.OnnxTransformer` — a general-purpose ONNX inference transform that takes arbitrary named columns.
- Our transform is specialized for **transformer-architecture text models**: it expects tokenized input (`input_ids`, `attention_mask`, `token_type_ids`) and auto-discovers tensor names using transformer-model conventions.
- "TextModel" makes clear this is for BERT/GPT-style models, not arbitrary ONNX (e.g., image classifiers, tabular models).

## Files to Create

| File | Contents |
|------|----------|
| `src/MLNet.TextInference.Onnx/OnnxTextModelScorerEstimator.cs` | Estimator + options class |
| `src/MLNet.TextInference.Onnx/OnnxTextModelScorerTransformer.cs` | Transformer |

## Options Class

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the ONNX text model scorer transform.
/// Runs inference on a transformer-architecture ONNX model (BERT, MiniLM, etc.).
/// </summary>
public class OnnxTextModelScorerOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    // --- Input column names (must match tokenizer output) ---

    /// <summary>Name of the input token IDs column. Default: "TokenIds".</summary>
    public string TokenIdsColumnName { get; set; } = "TokenIds";

    /// <summary>Name of the input attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>
    /// Name of the input token type IDs column. Default: "TokenTypeIds".
    /// Set to null if the model doesn't use token type IDs.
    /// </summary>
    public string? TokenTypeIdsColumnName { get; set; } = "TokenTypeIds";

    // --- Output ---

    /// <summary>Name of the output column for raw model output. Default: "RawOutput".</summary>
    public string OutputColumnName { get; set; } = "RawOutput";

    // --- Inference configuration ---

    /// <summary>
    /// Maximum sequence length. Must match the tokenizer's MaxTokenLength.
    /// Default: 128.
    /// </summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>Batch size for ONNX inference. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;

    // --- ONNX tensor name overrides (null = auto-detect) ---

    /// <summary>ONNX input tensor name for token IDs. Null = auto-detect ("input_ids").</summary>
    public string? InputIdsTensorName { get; set; }

    /// <summary>ONNX input tensor name for attention mask. Null = auto-detect ("attention_mask").</summary>
    public string? AttentionMaskTensorName { get; set; }

    /// <summary>ONNX input tensor name for token type IDs. Null = auto-detect ("token_type_ids" if present).</summary>
    public string? TokenTypeIdsTensorName { get; set; }

    /// <summary>
    /// ONNX output tensor name. Null = auto-detect.
    /// Auto-detection prefers "sentence_embedding" / "pooler_output" (pre-pooled),
    /// falls back to "last_hidden_state" / "output" (unpooled).
    /// </summary>
    public string? OutputTensorName { get; set; }
}
```

### Design Note: Tensor Name Options vs. Column Name Options

The options class distinguishes two naming layers:
- **Column names** (`TokenIdsColumnName`, etc.): IDataView column names that coordinate with the tokenizer transform
- **Tensor names** (`InputIdsTensorName`, etc.): ONNX graph tensor names for the native inference session

This separation is important because column names are a pipeline concern (how transforms communicate) while tensor names are a model concern (what the ONNX graph expects). Users override tensor names only for non-standard models; column names only when composing with differently-named upstream transforms.

## Estimator

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an OnnxTextModelScorerTransformer.
/// Fit() validates the input schema, loads the ONNX model, and auto-discovers tensor metadata.
/// </summary>
public sealed class OnnxTextModelScorerEstimator : IEstimator<OnnxTextModelScorerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextModelScorerOptions _options;

    public OnnxTextModelScorerEstimator(MLContext mlContext, OnnxTextModelScorerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
    }

    public OnnxTextModelScorerTransformer Fit(IDataView input)
    {
        // Validate input schema has token columns
        ValidateColumn(input.Schema, _options.TokenIdsColumnName);
        ValidateColumn(input.Schema, _options.AttentionMaskColumnName);
        if (_options.TokenTypeIdsColumnName != null)
            ValidateColumn(input.Schema, _options.TokenTypeIdsColumnName);

        // Load ONNX model and auto-discover tensor metadata
        var session = new InferenceSession(_options.ModelPath);
        var metadata = DiscoverModelMetadata(session);

        return new OnnxTextModelScorerTransformer(
            _mlContext, _options, session, metadata);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Validate input columns exist
        // Probe model to get output dimensions
        // Add output column (VBuffer<float>)
    }

    /// <summary>
    /// Auto-discovers ONNX tensor names and output shape.
    /// Extracted from existing OnnxTextEmbeddingEstimator.DiscoverModelMetadata().
    /// </summary>
    internal OnnxModelMetadata DiscoverModelMetadata(InferenceSession session)
    {
        var inputMeta = session.InputMetadata;
        var outputMeta = session.OutputMetadata;

        // Discover input tensor names (same logic as current implementation)
        string inputIdsName = _options.InputIdsTensorName
            ?? FindTensorName(inputMeta, ["input_ids"], "input_ids");
        string attentionMaskName = _options.AttentionMaskTensorName
            ?? FindTensorName(inputMeta, ["attention_mask"], "attention_mask");
        string? tokenTypeIdsName = _options.TokenTypeIdsTensorName
            ?? TryFindTensorName(inputMeta, ["token_type_ids"]);

        // Discover output tensor name
        // (same priority logic as current: prefer sentence_embedding, fall back to last_hidden_state)
        string outputName;
        bool hasPooledOutput;
        int hiddenDim;
        int outputRank;

        if (_options.OutputTensorName != null)
        {
            outputName = _options.OutputTensorName;
            var dims = outputMeta[outputName].Dimensions;
            hasPooledOutput = !dims.Contains(-1) && dims.Length == 2;
            hiddenDim = (int)dims.Last();
            outputRank = dims.Length;
        }
        else
        {
            var pooledName = TryFindTensorName(outputMeta, ["sentence_embedding", "pooler_output"]);
            if (pooledName != null)
            {
                outputName = pooledName;
                hasPooledOutput = true;
                hiddenDim = (int)outputMeta[pooledName].Dimensions.Last();
                outputRank = 2;
            }
            else
            {
                outputName = FindTensorName(outputMeta,
                    ["last_hidden_state", "output", "hidden_states"],
                    outputMeta.Keys.First());
                hasPooledOutput = false;
                hiddenDim = (int)outputMeta[outputName].Dimensions.Last();
                outputRank = 3;
            }
        }

        if (hiddenDim <= 0)
            throw new InvalidOperationException(
                $"Could not determine hidden dimension from ONNX output '{outputName}'.");

        return new OnnxModelMetadata(
            inputIdsName, attentionMaskName, tokenTypeIdsName,
            outputName, hiddenDim, hasPooledOutput, outputRank);
    }

    // FindTensorName / TryFindTensorName — same as current implementation
}

/// <summary>
/// Discovered ONNX model tensor metadata. Immutable record.
/// </summary>
internal sealed record OnnxModelMetadata(
    string InputIdsName,
    string AttentionMaskName,
    string? TokenTypeIdsName,
    string OutputTensorName,
    int HiddenDim,
    bool HasPooledOutput,
    int OutputRank);
```

## Transformer

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that runs ONNX inference on tokenized text inputs.
/// Task-agnostic — outputs the raw model tensor for downstream post-processing.
///
/// Lazy evaluation with lookahead batching: Transform() returns a wrapping IDataView.
/// The cursor reads ahead BatchSize rows from the upstream tokenizer cursor,
/// runs a single ONNX session.Run() call, then serves results one at a time.
/// This gives batch throughput (~1ms/item at batch=32) with lazy memory semantics
/// (~6 MB peak instead of ~1.9 GB for 10K rows).
/// </summary>
public sealed class OnnxTextModelScorerTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextModelScorerOptions _options;
    private readonly InferenceSession _session;
    private readonly OnnxModelMetadata _metadata;

    public bool IsRowToRowMapper => true;

    internal OnnxTextModelScorerOptions Options => _options;

    /// <summary>Hidden dimension of the model output.</summary>
    public int HiddenDim => _metadata.HiddenDim;

    /// <summary>Whether the model outputs pre-pooled embeddings (e.g., sentence_embedding).</summary>
    public bool HasPooledOutput => _metadata.HasPooledOutput;

    /// <summary>Auto-discovered ONNX metadata.</summary>
    internal OnnxModelMetadata Metadata => _metadata;

    internal OnnxTextModelScorerTransformer(
        MLContext mlContext,
        OnnxTextModelScorerOptions options,
        InferenceSession session,
        OnnxModelMetadata metadata)
    {
        _mlContext = mlContext;
        _options = options;
        _session = session;
        _metadata = metadata;
    }

    /// <summary>
    /// ML.NET face: returns a wrapping IDataView. No computation happens here.
    /// ONNX inference occurs lazily in the cursor via lookahead batching.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        return new ScorerDataView(input, this);
    }

    /// <summary>
    /// Direct face: run ONNX inference on pre-tokenized input without IDataView overhead.
    /// Used by the facade and MEAI generator.
    /// </summary>
    internal float[][] Score(TokenizedBatch batch)
    {
        return Score(batch.TokenIds, batch.AttentionMasks, batch.TokenTypeIds);
    }

    /// <summary>
    /// Runs ONNX inference in batches. Used by both the direct face and the cursor.
    /// </summary>
    internal float[][] Score(long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds)
    {
        int totalRows = tokenIds.Length;
        int batchSize = _options.BatchSize;
        int seqLen = _options.MaxTokenLength;
        var allOutputs = new List<float[]>(totalRows);

        for (int start = 0; start < totalRows; start += batchSize)
        {
            int count = Math.Min(batchSize, totalRows - start);
            var batchOutputs = RunOnnxBatch(
                tokenIds, attentionMasks, tokenTypeIds,
                start, count, seqLen);
            allOutputs.AddRange(batchOutputs);
        }

        return [.. allOutputs];
    }

    /// <summary>
    /// Runs a single ONNX inference batch. Core inference logic shared by
    /// the direct face and the cursor's lookahead batching.
    /// </summary>
    internal float[][] RunOnnxBatch(
        long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds,
        int startIdx, int batchSize, int seqLen)
    {
        // Build flat arrays from per-row arrays for this batch
        var idsArray = new long[batchSize * seqLen];
        var maskArray = new long[batchSize * seqLen];
        var typeIdsArray = _metadata.TokenTypeIdsName != null ? new long[batchSize * seqLen] : null;

        for (int b = 0; b < batchSize; b++)
        {
            Array.Copy(tokenIds[startIdx + b], 0, idsArray, b * seqLen, seqLen);
            Array.Copy(attentionMasks[startIdx + b], 0, maskArray, b * seqLen, seqLen);
            if (typeIdsArray != null && tokenTypeIds != null)
                Array.Copy(tokenTypeIds[startIdx + b], 0, typeIdsArray, b * seqLen, seqLen);
        }

        // Create OrtValues from flat backing arrays (zero-copy)
        var inputs = new Dictionary<string, OrtValue>
        {
            [_metadata.InputIdsName] = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]),
            [_metadata.AttentionMaskName] = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen])
        };

        if (_metadata.TokenTypeIdsName != null && typeIdsArray != null)
            inputs[_metadata.TokenTypeIdsName] = OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);

        try
        {
            using var results = _session.Run(new RunOptions(), inputs, [_metadata.OutputTensorName]);
            var output = results[0];
            var outputSpan = output.GetTensorDataAsSpan<float>();

            var batchOutputs = new float[batchSize][];

            if (_metadata.HasPooledOutput)
            {
                for (int b = 0; b < batchSize; b++)
                    batchOutputs[b] = outputSpan.Slice(b * _metadata.HiddenDim, _metadata.HiddenDim).ToArray();
            }
            else
            {
                int rowSize = seqLen * _metadata.HiddenDim;
                for (int b = 0; b < batchSize; b++)
                    batchOutputs[b] = outputSpan.Slice(b * rowSize, rowSize).ToArray();
            }

            return batchOutputs;
        }
        finally
        {
            foreach (var ortValue in inputs.Values)
                ortValue.Dispose();
        }
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);

        int outputSize = _metadata.HasPooledOutput
            ? _metadata.HiddenDim
            : _options.MaxTokenLength * _metadata.HiddenDim;

        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single, outputSize));

        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();
    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    public void Dispose() => _session.Dispose();
}
```

## Lazy IDataView and Cursor with Lookahead Batching

This is the most complex cursor in the pipeline. It reads ahead `BatchSize` rows from the
upstream tokenizer cursor, packs them into a single ONNX batch, runs inference once, then
serves results one at a time. This gives batch throughput with lazy memory semantics.

```csharp
/// <summary>
/// Wrapping IDataView that adds ONNX model output to the upstream schema.
/// No inference happens here — it's all in the cursor.
/// </summary>
internal sealed class ScorerDataView : IDataView
{
    private readonly IDataView _input;
    private readonly OnnxTextModelScorerTransformer _scorer;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal ScorerDataView(IDataView input, OnnxTextModelScorerTransformer scorer)
    {
        _input = input;
        _scorer = scorer;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);

        int outputSize = scorer.HasPooledOutput
            ? scorer.HiddenDim
            : scorer.Options.MaxTokenLength * scorer.HiddenDim;

        builder.AddColumn(scorer.Options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single, outputSize));

        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        // Always need token columns from upstream for inference
        var options = _scorer.Options;
        var upstreamCols = new List<DataViewSchema.Column>();

        // Add all requested passthrough columns
        foreach (var col in columnsNeeded)
        {
            var inputCol = _input.Schema.GetColumnOrNull(col.Name);
            if (inputCol != null)
                upstreamCols.Add(inputCol.Value);
        }

        // Always need token columns for ONNX inference
        upstreamCols.Add(_input.Schema[options.TokenIdsColumnName]);
        upstreamCols.Add(_input.Schema[options.AttentionMaskColumnName]);
        if (options.TokenTypeIdsColumnName != null)
            upstreamCols.Add(_input.Schema[options.TokenTypeIdsColumnName]);

        var inputCursor = _input.GetRowCursor(upstreamCols.Distinct(), rand);
        return new ScorerCursor(this, inputCursor, _scorer);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        // Single cursor — ONNX session handles internal threading
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor with lookahead batching for ONNX inference.
///
/// State machine:
///   1. When the cached batch is exhausted, read ahead BatchSize rows from upstream
///   2. Pack token arrays into flat batch tensors
///   3. Run a single session.Run() call
///   4. Cache the batch results
///   5. Serve cached results one at a time via MoveNext()
///
/// Memory: only one batch of ONNX output is alive at any time (~6 MB for batch=32,
/// seqLen=128, hiddenDim=384). Previous batches are eligible for GC.
/// </summary>
internal sealed class ScorerCursor : DataViewRowCursor
{
    private readonly ScorerDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly OnnxTextModelScorerTransformer _scorer;

    // Lookahead batch state
    private float[][]? _batchResults;
    private long[][]? _batchAttentionMasks;  // kept for downstream pooling passthrough
    private int _batchIndex = -1;
    private int _batchCount = 0;
    private long _position = -1;
    private bool _inputExhausted = false;

    // Cached upstream column values for current batch
    // (needed for passthrough — we read ahead, so we must cache)
    private readonly List<Dictionary<string, object>> _batchUpstreamValues = new();

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _position;
    public override long Batch => 0;

    internal ScorerCursor(
        ScorerDataView parent,
        DataViewRowCursor inputCursor,
        OnnxTextModelScorerTransformer scorer)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _scorer = scorer;
    }

    public override bool MoveNext()
    {
        _batchIndex++;

        if (_batchResults == null || _batchIndex >= _batchCount)
        {
            if (_inputExhausted)
                return false;

            // Read ahead BatchSize rows from upstream
            if (!FillNextBatch())
                return false;
        }

        _position++;
        return true;
    }

    private bool FillNextBatch()
    {
        var options = _scorer.Options;
        int seqLen = options.MaxTokenLength;
        int batchSize = options.BatchSize;

        var tokenIdsBatch = new List<long[]>();
        var attMaskBatch = new List<long[]>();
        var typeIdsBatch = new List<long[]>();
        _batchUpstreamValues.Clear();

        // Read ahead from upstream cursor
        var tokenIdsGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[options.TokenIdsColumnName]);
        var attMaskGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[options.AttentionMaskColumnName]);
        var typeIdsGetter = options.TokenTypeIdsColumnName != null
            ? _inputCursor.GetGetter<VBuffer<long>>(
                _inputCursor.Schema[options.TokenTypeIdsColumnName])
            : null;

        VBuffer<long> tokenIdsBuffer = default;
        VBuffer<long> attMaskBuffer = default;
        VBuffer<long> typeIdsBuffer = default;

        for (int i = 0; i < batchSize; i++)
        {
            if (!_inputCursor.MoveNext())
            {
                _inputExhausted = true;
                break;
            }

            tokenIdsGetter(ref tokenIdsBuffer);
            attMaskGetter(ref attMaskBuffer);
            tokenIdsBatch.Add(tokenIdsBuffer.DenseValues().ToArray());
            attMaskBatch.Add(attMaskBuffer.DenseValues().ToArray());

            if (typeIdsGetter != null)
            {
                typeIdsGetter(ref typeIdsBuffer);
                typeIdsBatch.Add(typeIdsBuffer.DenseValues().ToArray());
            }

            // Cache upstream column values for passthrough
            // (implementation detail: cache text and other columns for this row)
            CacheUpstreamValues();
        }

        if (tokenIdsBatch.Count == 0)
            return false;

        // Run ONNX inference on the batch
        _batchResults = _scorer.RunOnnxBatch(
            tokenIdsBatch.ToArray(),
            attMaskBatch.ToArray(),
            typeIdsBatch.Count > 0 ? typeIdsBatch.ToArray() : null,
            startIdx: 0,
            batchSize: tokenIdsBatch.Count,
            seqLen: seqLen);

        _batchAttentionMasks = attMaskBatch.ToArray();
        _batchIndex = 0;
        _batchCount = tokenIdsBatch.Count;
        return true;
    }

    private void CacheUpstreamValues()
    {
        // Cache all passthrough column values from the upstream cursor
        // for the current row, so they can be served when the downstream
        // consumer requests them via GetGetter.
        // Implementation stores per-row dictionaries keyed by column name.
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        // For the raw output column, return the cached ONNX result
        if (column.Name == _scorer.Options.OutputColumnName)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            {
                var data = _batchResults![_batchIndex];
                var editor = VBufferEditor.Create(ref value, data.Length);
                data.AsSpan().CopyTo(editor.Values);
                value = editor.Commit();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // For passthrough columns, return cached upstream values
        // (we can't delegate to _inputCursor because it's already advanced past this row)
        return GetCachedUpstreamGetter<TValue>(column);
    }

    private ValueGetter<TValue> GetCachedUpstreamGetter<TValue>(DataViewSchema.Column column)
    {
        // Return a getter that reads from the cached upstream values
        // for the current _batchIndex position.
        // Implementation depends on column type (text, VBuffer<long>, etc.)
        throw new NotImplementedException("Getter for cached upstream column");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
    {
        return (ref DataViewRowId value) =>
            value = new DataViewRowId((ulong)_position, 0);
    }

    public override bool IsColumnActive(DataViewSchema.Column column) => true;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _inputCursor.Dispose();
        base.Dispose(disposing);
    }
}
```

### Lookahead Batching: Why It Matters

| Approach | Memory (10K rows, unpooled) | ONNX throughput | Per-item latency |
|----------|---------------------------|-----------------|------------------|
| Eager (materialize all) | ~1.9 GB | ✅ Full batching | ~1ms |
| Lazy, no batching | ~200 KB | ❌ 15x slower | ~15ms |
| **Lazy, lookahead batch** | **~6 MB** | **✅ Full batching** | **~1ms** |

The lookahead cursor achieves the same throughput as the monolith by buffering one batch
at a time. Previous batches are eligible for GC, so peak memory is bounded by `BatchSize × rowSize`.

### Upstream Column Caching

The lookahead pattern creates a subtle problem: when the cursor reads ahead N rows from
upstream, the upstream cursor advances past those rows. When the downstream consumer later
asks for passthrough column values (e.g., the Text column), we can't delegate to the upstream
cursor because it's pointing at row N, not row 0.

Solution: the cursor caches all passthrough column values for the current batch. This costs
memory proportional to one batch's worth of upstream data (typically small — text strings
and token arrays for 32 rows).

## Code to Extract From Existing Files

| Source | What to Extract | Target |
|--------|----------------|--------|
| `OnnxTextEmbeddingEstimator.DiscoverModelMetadata()` | Tensor auto-discovery | `OnnxTextModelScorerEstimator.DiscoverModelMetadata()` |
| `OnnxTextEmbeddingEstimator.FindTensorName()` | Tensor name lookup | `OnnxTextModelScorerEstimator.FindTensorName()` |
| `OnnxTextEmbeddingEstimator.TryFindTensorName()` | Tensor name lookup | `OnnxTextModelScorerEstimator.TryFindTensorName()` |
| `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 156-189 | ONNX batch inference | `OnnxTextModelScorerTransformer.RunOnnxBatch()` |

## Key Design Decisions

### Raw output is always per-row

The scorer unpacks the batch ONNX output into per-row `float[]` arrays. With lazy evaluation,
only one batch of per-row arrays exists at a time. For unpooled models, each batch is
`BatchSize × float[seqLen × hiddenDim]` ≈ 6 MB. Previous batches are GC'd.

### No pooling inside the scorer

The scorer is task-agnostic. It doesn't know if the downstream consumer wants mean pooling, CLS token, softmax, or argmax. It outputs whatever the ONNX model outputs.

### Metadata is exposed for downstream transforms

`HiddenDim`, `HasPooledOutput`, and `Metadata` are exposed so downstream transforms (like `EmbeddingPoolingTransformer`) can auto-configure themselves. The facade uses this to wire up the pooling transform without requiring the user to specify dimensions manually.

### RunOnnxBatch is internal, shared between faces

The `RunOnnxBatch()` method is the core inference logic. It's used by:
- The direct face (`Score()`) for MEAI/facade usage
- The cursor (`FillNextBatch()`) for lazy ML.NET pipeline usage

This avoids code duplication between the two paths.

## Approach D Migration Notes

When migrating to ML.NET, this transform becomes:

```
OnnxTextModelScorerTransformer : RowToRowTransformerBase
  Mapper : MapperBase
    MakeGetter() → lookahead batching cache + RunOnnxBatch()
    GetOutputColumnsCore() → output column definition
    GetDependenciesCore() → token column dependencies
    SaveModel() → embed ONNX model bytes via ctx.SaveBinaryStream()
  [LoadableClass] attributes (4 variants for save/load registry)
  static Create() factory for loading
```

The `ScorerDataView` and `ScorerCursor` classes get **deleted** — `RowToRowTransformerBase`
provides cursor creation and schema propagation. The lookahead batching logic moves into
`MakeGetter()` — the getter maintains a batch cache and refills it when exhausted, identical
to the cursor's `FillNextBatch()` pattern.

**Key advantage of Approach D:** `MapperBase` automatically handles:
- `GetRowCursorSet()` for parallel cursor access
- Column activity tracking (only compute requested columns)
- Thread-safe cursor creation
- Schema propagation with proper metadata

## Acceptance Criteria

1. `OnnxTextModelScorerEstimator` can be created with a valid ONNX model path
2. `Fit()` validates that token columns exist in the input schema
3. `Fit()` auto-discovers ONNX tensor metadata (input/output names, dimensions)
4. `Transform()` returns a wrapping IDataView (no materialization)
5. Cursor uses lookahead batching (reads BatchSize rows, single session.Run())
6. Output shape is `float[hiddenDim]` for pre-pooled models, `float[seqLen × hiddenDim]` for unpooled
7. Passthrough columns are cached and served correctly
8. `Score()` (direct face) returns the same results without IDataView overhead
9. `Dispose()` disposes the `InferenceSession`
10. Manual tensor name overrides work for non-standard models
11. Peak memory is bounded by BatchSize × rowSize, not totalRows × rowSize

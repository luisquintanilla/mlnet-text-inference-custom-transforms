using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that pools raw model output into fixed-length embeddings.
/// Supports mean, CLS, and max pooling, plus optional L2 normalization.
///
/// Lazy evaluation: Transform() returns a wrapping IDataView. Pooling is computed
/// per-row as the cursor advances.
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

            embeddings[i] = EmbeddingPooling.Pool(
                hiddenStates, mask, 1, seqLen, hiddenDim,
                _options.Pooling, false)[0];

            if (_options.Normalize)
                L2Normalize(embeddings[i]);
        }

        return embeddings;
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
        {
            var maskCol = _input.Schema.GetColumnOrNull(_options.AttentionMaskColumnName);
            if (maskCol != null)
                upstreamCols.Add(maskCol.Value);
        }

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
/// Processes in lockstep — no lookahead, direct passthrough delegation.
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

        if (_options.IsPrePooled)
        {
            _currentEmbedding = rawOutputBuffer.DenseValues().ToArray();
            if (_options.Normalize)
            {
                float norm = TensorPrimitives.Norm(_currentEmbedding);
                if (norm > 0)
                    TensorPrimitives.Divide(_currentEmbedding, norm, _currentEmbedding);
            }
        }
        else
        {
            // Read attention mask
            long[] attentionMask;
            var maskCol = _inputCursor.Schema.GetColumnOrNull(_options.AttentionMaskColumnName);
            if (maskCol != null)
            {
                var maskGetter = _inputCursor.GetGetter<VBuffer<long>>(maskCol.Value);
                VBuffer<long> maskBuffer = default;
                maskGetter(ref maskBuffer);
                attentionMask = maskBuffer.DenseValues().ToArray();
            }
            else
            {
                // Default: all tokens active
                attentionMask = new long[_options.SequenceLength];
                Array.Fill(attentionMask, 1L);
            }

            ReadOnlySpan<float> rawOutput = rawOutputBuffer.DenseValues().ToArray();
            _currentEmbedding = EmbeddingPooling.Pool(
                rawOutput, attentionMask, 1, _options.SequenceLength, _options.HiddenDim,
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

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
        return new MappedDataView(input, GetRowToRowMapper(input.Schema));
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
                    EmbeddingPooling.L2Normalize(rawOutputs[i]);
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
                EmbeddingPooling.L2Normalize(embeddings[i]);
        }

        return embeddings;
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
        => new EmbeddingPoolingRowToRowMapper(inputSchema, this);

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

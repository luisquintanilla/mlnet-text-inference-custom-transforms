using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that applies sigmoid to raw model logits to produce a score.
/// Reads the first element of the raw output vector and applies sigmoid.
///
/// Lazy evaluation: Transform() returns a wrapping IDataView.
/// Sigmoid is computed per-row as the cursor advances.
/// </summary>
public sealed class SigmoidScorerTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly SigmoidScorerOptions _options;

    public bool IsRowToRowMapper => true;

    internal SigmoidScorerOptions Options => _options;

    internal SigmoidScorerTransformer(MLContext mlContext, SigmoidScorerOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    /// <summary>
    /// ML.NET face: returns a wrapping IDataView. No computation happens here.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        return new MappedDataView(input, GetRowToRowMapper(input.Schema));
    }

    /// <summary>
    /// Direct face: apply sigmoid to raw outputs without IDataView overhead.
    /// </summary>
    internal float[] Score(float[][] rawOutputs)
    {
        var scores = new float[rawOutputs.Length];
        for (int i = 0; i < rawOutputs.Length; i++)
        {
            float logit = rawOutputs[i][0];
            scores[i] = 1.0f / (1.0f + MathF.Exp(-logit));
        }
        return scores;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName, NumberDataViewType.Single);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => new SigmoidScorerRowToRowMapper(inputSchema, this);

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

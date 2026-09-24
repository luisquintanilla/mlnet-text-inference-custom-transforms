using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that applies softmax over raw logits to produce
/// class probabilities and a predicted label.
///
/// Lazy evaluation: Transform() returns a wrapping IDataView. Classification
/// is computed per-row as the cursor advances.
/// </summary>
public sealed class SoftmaxClassificationTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly SoftmaxClassificationOptions _options;

    public bool IsRowToRowMapper => true;

    internal SoftmaxClassificationOptions Options => _options;
    public int NumClasses => _options.NumClasses!.Value;
    public string[]? Labels => _options.Labels;

    internal SoftmaxClassificationTransformer(
        MLContext mlContext,
        SoftmaxClassificationOptions options)
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
    /// Direct face: classify raw outputs without IDataView overhead.
    /// </summary>
    internal ClassificationResult[] Classify(float[][] rawOutputs)
    {
        var results = new ClassificationResult[rawOutputs.Length];

        for (int i = 0; i < rawOutputs.Length; i++)
        {
            var logits = rawOutputs[i];
            var probabilities = StableSoftmax.Create(logits, _options.NumClasses!.Value);

            int predictedIndex = StableSoftmax.IndexOfMax(
                probabilities.AsSpan(0, _options.NumClasses.Value));
            string predictedLabel = _options.Labels != null && predictedIndex < _options.Labels.Length
                ? _options.Labels[predictedIndex]
                : predictedIndex.ToString();

            results[i] = new ClassificationResult
            {
                PredictedLabel = predictedLabel,
                Probabilities = probabilities
            };
        }

        return results;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.ProbabilitiesColumnName,
            new VectorDataViewType(NumberDataViewType.Single, _options.NumClasses!.Value));
        builder.AddColumn(_options.PredictedLabelColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => new SoftmaxClassificationRowToRowMapper(inputSchema, this);

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

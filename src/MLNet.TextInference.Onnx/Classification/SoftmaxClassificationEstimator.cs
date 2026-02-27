using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates a SoftmaxClassificationTransformer.
/// Validates schema and passes configuration through.
/// </summary>
public sealed class SoftmaxClassificationEstimator : IEstimator<SoftmaxClassificationTransformer>
{
    private readonly MLContext _mlContext;
    private readonly SoftmaxClassificationOptions _options;

    public SoftmaxClassificationEstimator(MLContext mlContext, SoftmaxClassificationOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public SoftmaxClassificationTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // Infer NumClasses from Labels if not explicitly set
        if (_options.NumClasses == null && _options.Labels != null)
            _options.NumClasses = _options.Labels.Length;

        if (_options.NumClasses is null or <= 0)
            throw new ArgumentException(
                "NumClasses must be positive. Provide NumClasses or Labels.");

        return new SoftmaxClassificationTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var result = inputSchema.ToDictionary(x => x.Name);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];

        var probCol = (SchemaShape.Column)colCtor.Invoke([
            _options.ProbabilitiesColumnName,
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);
        result[_options.ProbabilitiesColumnName] = probCol;

        var labelCol = (SchemaShape.Column)colCtor.Invoke([
            _options.PredictedLabelColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)TextDataViewType.Instance,
            false,
            (SchemaShape?)null
        ]);
        result[_options.PredictedLabelColumnName] = labelCol;

        return new SchemaShape(result.Values);
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates a NerDecodingTransformer.
/// Trivial estimator — validates input columns and label configuration.
/// </summary>
public sealed class NerDecodingEstimator : IEstimator<NerDecodingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly NerDecodingOptions _options;

    public NerDecodingEstimator(MLContext mlContext, NerDecodingOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (options.Labels == null || options.Labels.Length == 0)
            throw new ArgumentException("Labels must be provided.", nameof(options));

        options.NumLabels ??= options.Labels.Length;
    }

    public NerDecodingTransformer Fit(IDataView input)
    {
        ValidateColumn(input.Schema, _options.InputColumnName);
        ValidateColumn(input.Schema, _options.AttentionMaskColumnName);
        ValidateColumn(input.Schema, _options.TokenStartOffsetsColumnName);
        ValidateColumn(input.Schema, _options.TokenEndOffsetsColumnName);
        ValidateColumn(input.Schema, _options.TextColumnName);

        return new NerDecodingTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var result = inputSchema.ToDictionary(x => x.Name);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var outputCol = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)TextDataViewType.Instance,
            false,
            (SchemaShape?)null
        ]);
        result[_options.OutputColumnName] = outputCol;

        return new SchemaShape(result.Values);
    }

    private static void ValidateColumn(DataViewSchema schema, string columnName)
    {
        if (schema.GetColumnOrNull(columnName) == null)
            throw new ArgumentException(
                $"Input schema does not contain required column '{columnName}'.");
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates a QaSpanExtractionTransformer.
/// Validates that start/end logits, offsets, and text columns exist.
/// </summary>
public sealed class QaSpanExtractionEstimator : IEstimator<QaSpanExtractionTransformer>
{
    private readonly MLContext _mlContext;
    private readonly QaSpanExtractionOptions _options;

    public QaSpanExtractionEstimator(MLContext mlContext, QaSpanExtractionOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public QaSpanExtractionTransformer Fit(IDataView input)
    {
        ValidateColumn(input.Schema, _options.StartLogitsColumnName);
        ValidateColumn(input.Schema, _options.EndLogitsColumnName);
        ValidateColumn(input.Schema, _options.AttentionMaskColumnName);
        ValidateColumn(input.Schema, _options.TokenStartOffsetsColumnName);
        ValidateColumn(input.Schema, _options.TokenEndOffsetsColumnName);
        ValidateColumn(input.Schema, _options.TextColumnName);

        return new QaSpanExtractionTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var result = inputSchema.ToDictionary(x => x.Name);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];

        var answerCol = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)TextDataViewType.Instance,
            false,
            (SchemaShape?)null
        ]);
        result[_options.OutputColumnName] = answerCol;

        var scoreCol = (SchemaShape.Column)colCtor.Invoke([
            _options.ScoreColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);
        result[_options.ScoreColumnName] = scoreCol;

        return new SchemaShape(result.Values);
    }

    private static void ValidateColumn(DataViewSchema schema, string columnName)
    {
        if (schema.GetColumnOrNull(columnName) == null)
            throw new ArgumentException(
                $"Input schema does not contain required column '{columnName}'.");
    }
}

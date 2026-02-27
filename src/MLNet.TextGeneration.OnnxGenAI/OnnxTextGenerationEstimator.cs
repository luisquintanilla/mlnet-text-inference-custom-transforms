using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace MLNet.TextGeneration.OnnxGenAI;

/// <summary>
/// ML.NET IEstimator that loads an ONNX Runtime GenAI model for text generation.
/// </summary>
public sealed class OnnxTextGenerationEstimator : IEstimator<OnnxTextGenerationTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextGenerationOptions _options;

    public OnnxTextGenerationEstimator(MLContext mlContext, OnnxTextGenerationOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public OnnxTextGenerationTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        var model = new Model(_options.ModelPath);
        var tokenizer = new Tokenizer(model);
        return new OnnxTextGenerationTransformer(_mlContext, _options, model, tokenizer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var inputCol = inputSchema.FirstOrDefault(c => c.Name == _options.InputColumnName);
        if (inputCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

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
}

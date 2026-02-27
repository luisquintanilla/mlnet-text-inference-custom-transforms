using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the embedding pooling transform.
/// Reduces raw model output to a fixed-length embedding vector.
/// </summary>
public class EmbeddingPoolingOptions
{
    /// <summary>
    /// Name of the input column containing raw model output.
    /// Default: "RawOutput".
    /// </summary>
    public string InputColumnName { get; set; } = "RawOutput";

    /// <summary>
    /// Name of the attention mask column. Required for mean and max pooling.
    /// Default: "AttentionMask".
    /// </summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Name of the output embedding column. Default: "Embedding".</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>
    /// Pooling strategy for reducing per-token outputs to a single vector.
    /// Ignored when IsPrePooled is true. Default: MeanPooling.
    /// </summary>
    public PoolingStrategy Pooling { get; set; } = PoolingStrategy.MeanPooling;

    /// <summary>Whether to L2-normalize the output embeddings. Default: true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>
    /// Hidden dimension of the model output.
    /// When used via the facade, this is auto-set from scorer metadata.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Sequence length of the unpooled model output.
    /// Only needed for unpooled models.
    /// </summary>
    public int SequenceLength { get; set; }

    /// <summary>
    /// Whether the input is already pooled (e.g., sentence_embedding output).
    /// When true, only normalization is applied (pooling strategy is ignored).
    /// Default: false.
    /// </summary>
    public bool IsPrePooled { get; set; }
}

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

    public EmbeddingPoolingTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

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

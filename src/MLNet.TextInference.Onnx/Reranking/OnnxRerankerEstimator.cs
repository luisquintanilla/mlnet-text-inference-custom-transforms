using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an OnnxRerankerTransformer.
/// Internally composes TextTokenizerEstimator (text-pair) → OnnxTextModelScorerEstimator → SigmoidScorerEstimator.
/// </summary>
public sealed class OnnxRerankerEstimator : IEstimator<OnnxRerankerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxRerankerOptions _options;

    public OnnxRerankerEstimator(MLContext mlContext, OnnxRerankerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
        if (!File.Exists(options.TokenizerPath) && !Directory.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer path not found: {options.TokenizerPath}");
    }

    public OnnxRerankerTransformer Fit(IDataView input)
    {
        var queryCol = input.Schema.GetColumnOrNull(_options.QueryColumnName);
        if (queryCol == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.QueryColumnName}'.");

        var docCol = input.Schema.GetColumnOrNull(_options.DocumentColumnName);
        if (docCol == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.DocumentColumnName}'.");

        // 1. Create and fit the tokenizer (text-pair mode)
        var tokenizerOptions = new TextTokenizerOptions
        {
            TokenizerPath = _options.TokenizerPath,
            InputColumnName = _options.QueryColumnName,
            SecondInputColumnName = _options.DocumentColumnName,
            MaxTokenLength = _options.MaxTokenLength,
        };
        var tokenizerEstimator = new TextTokenizerEstimator(_mlContext, tokenizerOptions);
        var tokenizerTransformer = tokenizerEstimator.Fit(input);

        // 2. Create and fit the scorer
        var tokenizedData = tokenizerTransformer.Transform(input);

        var scorerOptions = new OnnxTextModelScorerOptions
        {
            ModelPath = _options.ModelPath,
            MaxTokenLength = _options.MaxTokenLength,
            BatchSize = _options.BatchSize,
            GpuDeviceId = _options.GpuDeviceId,
            FallbackToCpu = _options.FallbackToCpu,
            PreferredOutputNames = ["logits", "output"],
        };
        var scorerEstimator = new OnnxTextModelScorerEstimator(_mlContext, scorerOptions);
        var scorerTransformer = scorerEstimator.Fit(tokenizedData);

        // 3. Create and fit the sigmoid scorer
        var scoredData = scorerTransformer.Transform(tokenizedData);

        var sigmoidOptions = new SigmoidScorerOptions
        {
            OutputColumnName = _options.OutputColumnName,
        };
        var sigmoidEstimator = new SigmoidScorerEstimator(_mlContext, sigmoidOptions);
        var sigmoidTransformer = sigmoidEstimator.Fit(scoredData);

        return new OnnxRerankerTransformer(
            _mlContext, _options,
            tokenizerTransformer, scorerTransformer, sigmoidTransformer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var queryCol = inputSchema.FirstOrDefault(c => c.Name == _options.QueryColumnName);
        if (queryCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.QueryColumnName}'.");

        var docCol = inputSchema.FirstOrDefault(c => c.Name == _options.DocumentColumnName);
        if (docCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.DocumentColumnName}'.");

        var result = inputSchema.ToDictionary(x => x.Name);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var outputCol = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);
        result[_options.OutputColumnName] = outputCol;

        return new SchemaShape(result.Values);
    }
}

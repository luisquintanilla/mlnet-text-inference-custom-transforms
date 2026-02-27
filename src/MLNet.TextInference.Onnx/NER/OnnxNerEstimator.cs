using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Facade estimator that chains tokenizer → ONNX scorer → NER decoder.
/// </summary>
public sealed class OnnxNerEstimator : IEstimator<OnnxNerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxNerOptions _options;

    public OnnxNerEstimator(MLContext mlContext, OnnxNerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");

        if (!File.Exists(options.TokenizerPath) && !Directory.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer path not found: {options.TokenizerPath}");
    }

    public OnnxNerTransformer Fit(IDataView input)
    {
        // 1. Tokenizer with offset tracking
        var tokenizerOptions = new TextTokenizerOptions
        {
            TokenizerPath = _options.TokenizerPath,
            InputColumnName = _options.InputColumnName,
            MaxTokenLength = _options.MaxTokenLength,
            OutputOffsets = true,
            OutputTokenTypeIds = true
        };
        var tokenizerEstimator = new TextTokenizerEstimator(_mlContext, tokenizerOptions);
        var tokenizerTransformer = tokenizerEstimator.Fit(input);
        var tokenized = tokenizerTransformer.Transform(input);

        // 2. ONNX scorer (prefer logits output for NER)
        var scorerOptions = new OnnxTextModelScorerOptions
        {
            ModelPath = _options.ModelPath,
            MaxTokenLength = _options.MaxTokenLength,
            BatchSize = _options.BatchSize,
            GpuDeviceId = _options.GpuDeviceId,
            FallbackToCpu = _options.FallbackToCpu,
            PreferredOutputNames = ["logits", "output"]
        };
        var scorerEstimator = new OnnxTextModelScorerEstimator(_mlContext, scorerOptions);
        var scorerTransformer = scorerEstimator.Fit(tokenized);
        var scored = scorerTransformer.Transform(tokenized);

        // 3. NER decoder
        var nerOptions = new NerDecodingOptions
        {
            Labels = _options.Labels,
            OutputColumnName = _options.OutputColumnName,
            TextColumnName = _options.InputColumnName
        };
        var nerEstimator = new NerDecodingEstimator(_mlContext, nerOptions);
        var nerTransformer = nerEstimator.Fit(scored);

        return new OnnxNerTransformer(
            _mlContext, _options,
            tokenizerTransformer, scorerTransformer, nerTransformer);
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
}

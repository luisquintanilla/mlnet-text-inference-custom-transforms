using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an OnnxTextClassificationTransformer.
/// Internally composes TextTokenizerEstimator → OnnxTextModelScorerEstimator → SoftmaxClassificationEstimator.
/// </summary>
public sealed class OnnxTextClassificationEstimator : IEstimator<OnnxTextClassificationTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextClassificationOptions _options;

    public OnnxTextClassificationEstimator(MLContext mlContext, OnnxTextClassificationOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
        if (!File.Exists(options.TokenizerPath) && !Directory.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer path not found: {options.TokenizerPath}");
    }

    public OnnxTextClassificationTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // 1. Create and fit the tokenizer
        var tokenizerOptions = new TextTokenizerOptions
        {
            TokenizerPath = _options.TokenizerPath,
            InputColumnName = _options.InputColumnName,
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

        // 3. Auto-detect NumClasses from scorer output dimension
        int numClasses = scorerTransformer.HasPooledOutput
            ? scorerTransformer.HiddenDim
            : scorerTransformer.HiddenDim;

        // 4. Create and fit the softmax classifier
        var scoredData = scorerTransformer.Transform(tokenizedData);

        var classificationOptions = new SoftmaxClassificationOptions
        {
            ProbabilitiesColumnName = _options.ProbabilitiesColumnName,
            PredictedLabelColumnName = _options.PredictedLabelColumnName,
            Labels = _options.Labels,
            NumClasses = numClasses,
        };
        var classificationEstimator = new SoftmaxClassificationEstimator(_mlContext, classificationOptions);
        var classificationTransformer = classificationEstimator.Fit(scoredData);

        return new OnnxTextClassificationTransformer(
            _mlContext, _options,
            tokenizerTransformer, scorerTransformer, classificationTransformer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var inputCol = inputSchema.FirstOrDefault(c => c.Name == _options.InputColumnName);
        if (inputCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        if (inputCol.ItemType != TextDataViewType.Instance)
            throw new ArgumentException(
                $"Column '{_options.InputColumnName}' must be of type Text, but is {inputCol.ItemType}.");

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

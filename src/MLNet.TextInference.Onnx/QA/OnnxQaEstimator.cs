using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Facade estimator that chains text-pair tokenizer → multi-output ONNX scorer → QA span extractor.
/// </summary>
public sealed class OnnxQaEstimator : IEstimator<OnnxQaTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxQaOptions _options;

    public OnnxQaEstimator(MLContext mlContext, OnnxQaOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");

        if (!File.Exists(options.TokenizerPath) && !Directory.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer path not found: {options.TokenizerPath}");
    }

    public OnnxQaTransformer Fit(IDataView input)
    {
        var questionCol = input.Schema.GetColumnOrNull(_options.QuestionColumnName);
        if (questionCol == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.QuestionColumnName}'.");

        var contextCol = input.Schema.GetColumnOrNull(_options.ContextColumnName);
        if (contextCol == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.ContextColumnName}'.");

        // 1. Tokenizer with text-pair mode and offset tracking
        var tokenizerOptions = new TextTokenizerOptions
        {
            TokenizerPath = _options.TokenizerPath,
            InputColumnName = _options.QuestionColumnName,
            SecondInputColumnName = _options.ContextColumnName,
            OutputOffsets = true,
            MaxTokenLength = _options.MaxTokenLength,
        };
        var tokenizerEstimator = new TextTokenizerEstimator(_mlContext, tokenizerOptions);
        var tokenizerTransformer = tokenizerEstimator.Fit(input);
        var tokenized = tokenizerTransformer.Transform(input);

        // 2. Multi-output ONNX scorer (start_logits + end_logits)
        var scorerOptions = new OnnxTextModelScorerOptions
        {
            ModelPath = _options.ModelPath,
            MaxTokenLength = _options.MaxTokenLength,
            BatchSize = _options.BatchSize,
            OutputColumnName = "StartLogits",
            PreferredOutputNames = ["start_logits"],
            AdditionalOutputTensorNames = ["end_logits"],
            AdditionalOutputColumnNames = ["EndLogits"],
            GpuDeviceId = _options.GpuDeviceId,
            FallbackToCpu = _options.FallbackToCpu,
        };
        var scorerEstimator = new OnnxTextModelScorerEstimator(_mlContext, scorerOptions);
        var scorerTransformer = scorerEstimator.Fit(tokenized);
        var scored = scorerTransformer.Transform(tokenized);

        // 3. QA span extraction
        var qaOptions = new QaSpanExtractionOptions
        {
            TextColumnName = _options.ContextColumnName,
            OutputColumnName = _options.OutputColumnName,
            ScoreColumnName = _options.ScoreColumnName,
            MaxAnswerLength = _options.MaxAnswerLength,
        };
        var qaEstimator = new QaSpanExtractionEstimator(_mlContext, qaOptions);
        var qaTransformer = qaEstimator.Fit(scored);

        return new OnnxQaTransformer(
            _mlContext, _options,
            tokenizerTransformer, scorerTransformer, qaTransformer);
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
}

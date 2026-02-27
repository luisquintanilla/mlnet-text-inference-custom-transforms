using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an OnnxTextEmbeddingTransformer.
/// Internally composes TextTokenizerEstimator → OnnxTextModelScorerEstimator → EmbeddingPoolingEstimator.
/// </summary>
public sealed class OnnxTextEmbeddingEstimator : IEstimator<OnnxTextEmbeddingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextEmbeddingOptions _options;

    public OnnxTextEmbeddingEstimator(MLContext mlContext, OnnxTextEmbeddingOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
        if (!File.Exists(options.TokenizerPath) && !Directory.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer path not found: {options.TokenizerPath}");
    }

    public OnnxTextEmbeddingTransformer Fit(IDataView input)
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
            InputIdsTensorName = _options.InputIdsName,
            AttentionMaskTensorName = _options.AttentionMaskName,
            TokenTypeIdsTensorName = _options.TokenTypeIdsName,
            OutputTensorName = _options.OutputTensorName,
            GpuDeviceId = _options.GpuDeviceId,
            FallbackToCpu = _options.FallbackToCpu,
            PreferredOutputNames = ["sentence_embedding", "pooler_output", "last_hidden_state"],
        };
        var scorerEstimator = new OnnxTextModelScorerEstimator(_mlContext, scorerOptions);
        var scorerTransformer = scorerEstimator.Fit(tokenizedData);

        // 3. Create and fit the pooler (auto-configured from scorer metadata)
        var scoredData = scorerTransformer.Transform(tokenizedData);

        var poolingOptions = new EmbeddingPoolingOptions
        {
            OutputColumnName = _options.OutputColumnName,
            Pooling = _options.Pooling,
            Normalize = _options.Normalize,
            HiddenDim = scorerTransformer.HiddenDim,
            IsPrePooled = scorerTransformer.HasPooledOutput,
            SequenceLength = scorerTransformer.HasPooledOutput ? 1 : _options.MaxTokenLength,
        };
        var poolingEstimator = new EmbeddingPoolingEstimator(_mlContext, poolingOptions);
        var poolingTransformer = poolingEstimator.Fit(scoredData);

        return new OnnxTextEmbeddingTransformer(
            _mlContext, _options,
            tokenizerTransformer, scorerTransformer, poolingTransformer);
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

        // Probe the model to get embedding dimension
        int embeddingDim;
        var scorerEstimator = new OnnxTextModelScorerEstimator(_mlContext, new OnnxTextModelScorerOptions
        {
            ModelPath = _options.ModelPath,
            InputIdsTensorName = _options.InputIdsName,
            AttentionMaskTensorName = _options.AttentionMaskName,
            TokenTypeIdsTensorName = _options.TokenTypeIdsName,
            OutputTensorName = _options.OutputTensorName,
            GpuDeviceId = _options.GpuDeviceId,
            FallbackToCpu = _options.FallbackToCpu,
            PreferredOutputNames = ["sentence_embedding", "pooler_output", "last_hidden_state"],
        });
        var metadata = scorerEstimator.DiscoverModelMetadata();
        embeddingDim = metadata.HiddenDim;

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

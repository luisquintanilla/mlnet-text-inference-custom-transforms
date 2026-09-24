using Microsoft.Extensions.AI;
using Microsoft.ML;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Extension methods for MLContext to provide a convenient API for ONNX text embeddings.
/// </summary>
public static class MLContextExtensions
{
    /// <summary>
    /// Appends the compiled typed-decision facade to an existing ML.NET estimator chain.
    /// </summary>
    public static IEstimator<ITransformer> AppendOnnxTypedDecisions(
        this IEstimator<ITransformer> pipeline,
        MLContext mlContext,
        OnnxTypedDecisionsOptions options)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(mlContext);
        return Microsoft.ML.LearningPipelineExtensions.Append(
            pipeline, new OnnxTypedDecisionsEstimator(mlContext, options));
    }

    /// <summary>
    /// Creates a schema-aware, cursor-batched typed-decision transform backed by a local bundle.
    /// </summary>
    public static OnnxTypedDecisionsEstimator OnnxTypedDecisions(
        this TransformsCatalog catalog,
        OnnxTypedDecisionsOptions options)
    {
        return new OnnxTypedDecisionsEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>Creates the preparation stage of a typed-decision pipeline.</summary>
    public static DecisionInputPreparationEstimator PrepareDecisionInputs(
        this TransformsCatalog catalog,
        DecisionInputPreparationOptions options)
    {
        return new DecisionInputPreparationEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>Creates the ONNX scoring stage of a typed-decision pipeline.</summary>
    public static OnnxDecisionModelScorerEstimator ScoreOnnxDecisionModel(
        this TransformsCatalog catalog,
        OnnxDecisionModelScorerOptions options)
    {
        return new OnnxDecisionModelScorerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>Creates the typed-decoding stage of a typed-decision pipeline.</summary>
    public static DecisionDecodingEstimator DecodeDecisions(
        this TransformsCatalog catalog,
        DecisionDecodingOptions options)
    {
        return new DecisionDecodingEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates an estimator that generates text embeddings using a local ONNX model.
    /// Encapsulates tokenization, ONNX inference, pooling, and normalization.
    /// </summary>
    public static OnnxTextEmbeddingEstimator OnnxTextEmbedding(
        this TransformsCatalog catalog,
        OnnxTextEmbeddingOptions options)
    {
        return new OnnxTextEmbeddingEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a provider-agnostic embedding transform that wraps any IEmbeddingGenerator.
    /// </summary>
    public static EmbeddingGeneratorEstimator TextEmbedding(
        this TransformsCatalog catalog,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        EmbeddingGeneratorOptions? options = null)
    {
        return new EmbeddingGeneratorEstimator(catalog.GetMLContext(), generator, options);
    }

    /// <summary>
    /// Creates a text tokenizer transform for transformer-based models.
    /// </summary>
    public static TextTokenizerEstimator TokenizeText(
        this TransformsCatalog catalog,
        TextTokenizerOptions options)
    {
        return new TextTokenizerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates an ONNX text model scorer transform for transformer-based models.
    /// </summary>
    public static OnnxTextModelScorerEstimator ScoreOnnxTextModel(
        this TransformsCatalog catalog,
        OnnxTextModelScorerOptions options)
    {
        return new OnnxTextModelScorerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates an embedding pooling transform for reducing raw model output to embeddings.
    /// </summary>
    public static EmbeddingPoolingEstimator PoolEmbedding(
        this TransformsCatalog catalog,
        EmbeddingPoolingOptions options)
    {
        return new EmbeddingPoolingEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a softmax classification post-processing transform.
    /// </summary>
    public static SoftmaxClassificationEstimator SoftmaxClassify(
        this TransformsCatalog catalog, SoftmaxClassificationOptions options)
    {
        return new SoftmaxClassificationEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a full text classification pipeline using a local ONNX model.
    /// Encapsulates tokenization, ONNX inference, and softmax classification.
    /// </summary>
    public static OnnxTextClassificationEstimator OnnxTextClassification(
        this TransformsCatalog catalog, OnnxTextClassificationOptions options)
    {
        return new OnnxTextClassificationEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a provider-agnostic text generation transform that wraps any IChatClient.
    /// </summary>
    public static ChatClientEstimator TextGeneration(
        this TransformsCatalog catalog,
        IChatClient chatClient,
        TextGenerationOptions? options = null)
    {
        return new ChatClientEstimator(catalog.GetMLContext(), chatClient, options);
    }

    /// <summary>
    /// Creates a sigmoid scorer transform for converting raw logits to probabilities.
    /// </summary>
    public static SigmoidScorerEstimator SigmoidScore(
        this TransformsCatalog catalog,
        SigmoidScorerOptions options)
    {
        return new SigmoidScorerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a cross-encoder reranker transform using a local ONNX model.
    /// Encapsulates text-pair tokenization → ONNX inference → sigmoid scoring.
    /// </summary>
    public static OnnxRerankerEstimator OnnxRerank(
        this TransformsCatalog catalog,
        OnnxRerankerOptions options)
    {
        return new OnnxRerankerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a NER decoding transform that converts BIO-tagged model output into entity spans.
    /// </summary>
    public static NerDecodingEstimator NerDecode(
        this TransformsCatalog catalog, NerDecodingOptions options)
    {
        return new NerDecodingEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates an end-to-end ONNX NER transform (tokenizer → scorer → decoder).
    /// </summary>
    public static OnnxNerEstimator OnnxNer(
        this TransformsCatalog catalog, OnnxNerOptions options)
    {
        return new OnnxNerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a QA span extraction transform that finds answer spans from start/end logits.
    /// </summary>
    public static QaSpanExtractionEstimator QaExtract(
        this TransformsCatalog catalog, QaSpanExtractionOptions options)
    {
        return new QaSpanExtractionEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates an end-to-end ONNX extractive QA transform (text-pair tokenizer → multi-output scorer → QA extractor).
    /// </summary>
    public static OnnxQaEstimator OnnxQa(
        this TransformsCatalog catalog, OnnxQaOptions options)
    {
        return new OnnxQaEstimator(catalog.GetMLContext(), options);
    }

    // Gets the real MLContext from TransformsCatalog via reflection so that
    // context-level settings (e.g. GpuDeviceId) are preserved.
    private static MLContext GetMLContext(this TransformsCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        try
        {
            var environmentProperty = typeof(TransformsCatalog)
                .GetProperties(System.Reflection.BindingFlags.NonPublic |
                               System.Reflection.BindingFlags.Instance)
                .FirstOrDefault(static property =>
                    property.Name.EndsWith(".Environment", StringComparison.Ordinal));
            var environment = environmentProperty?.GetValue(catalog);
            if (environment is MLContext mlContext)
                return mlContext;
            if (environment is null)
                throw new InvalidOperationException(
                    "The ML.NET TransformsCatalog does not expose its host environment.");

            var environmentType = environment.GetType();
            var seed = ReadRequiredNullableInt(environment, environmentType, "Seed");
            var gpuDeviceId = ReadRequiredNullableInt(environment, environmentType, "GpuDeviceId");
            var fallbackToCpu = ReadRequiredBool(environment, environmentType, "FallbackToCpu");
            var tempFilePath = ReadRequiredString(environment, environmentType, "TempFilePath");
            var recovered = new MLContext(seed)
            {
                GpuDeviceId = gpuDeviceId,
                FallbackToCpu = fallbackToCpu,
                TempFilePath = tempFilePath
            };

            return recovered;
        }
        catch (InvalidOperationException)
        {
            throw;
        }
        catch (Exception exception)
        {
            throw new InvalidOperationException(
                "Could not recover MLContext settings from TransformsCatalog. " +
                "The transform cannot safely preserve provider and seed settings.",
                exception);
        }
    }

    private static int? ReadRequiredNullableInt(
        object environment,
        Type environmentType,
        string propertyName)
    {
        var property = environmentType.GetProperty(propertyName)
            ?? throw MissingEnvironmentProperty(environmentType, propertyName);
        if (property.PropertyType != typeof(int?) &&
            property.PropertyType != typeof(int))
        {
            throw new InvalidOperationException(
                $"ML.NET environment property '{propertyName}' has unexpected type " +
                $"'{property.PropertyType.FullName}'.");
        }

        var value = property.GetValue(environment);
        if (value is null)
            return null;
        if (value is int integer)
            return integer;
        throw new InvalidOperationException(
            $"ML.NET environment property '{propertyName}' returned an unexpected value.");
    }

    private static bool ReadRequiredBool(
        object environment,
        Type environmentType,
        string propertyName)
    {
        var property = environmentType.GetProperty(propertyName)
            ?? throw MissingEnvironmentProperty(environmentType, propertyName);
        if (property.PropertyType != typeof(bool))
            throw new InvalidOperationException(
                $"ML.NET environment property '{propertyName}' has unexpected type " +
                $"'{property.PropertyType.FullName}'.");
        return property.GetValue(environment) is bool value
            ? value
            : throw new InvalidOperationException(
                $"ML.NET environment property '{propertyName}' returned an unexpected value.");
    }

    private static string ReadRequiredString(
        object environment,
        Type environmentType,
        string propertyName)
    {
        var property = environmentType.GetProperty(propertyName)
            ?? throw MissingEnvironmentProperty(environmentType, propertyName);
        if (property.PropertyType != typeof(string))
            throw new InvalidOperationException(
                $"ML.NET environment property '{propertyName}' has unexpected type " +
                $"'{property.PropertyType.FullName}'.");
        return property.GetValue(environment) as string
            ?? throw new InvalidOperationException(
                $"ML.NET environment property '{propertyName}' returned null.");
    }

    private static InvalidOperationException MissingEnvironmentProperty(
        Type environmentType,
        string propertyName)
        => new(
            $"ML.NET environment type '{environmentType.FullName}' does not expose " +
            $"required property '{propertyName}'.");
}

using Microsoft.Extensions.AI;
using Microsoft.ML;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Extension methods for MLContext to provide a convenient API for ONNX text embeddings.
/// </summary>
public static class MLContextExtensions
{
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

    // Gets the real MLContext from TransformsCatalog via reflection so that
    // context-level settings (e.g. GpuDeviceId) are preserved.
    private static MLContext GetMLContext(this TransformsCatalog catalog)
    {
        var envProperty = typeof(TransformsCatalog)
            .GetProperty("Environment", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        // In ML.NET 5.0+, MLContext implements IHostEnvironment directly
        if (envProperty?.GetValue(catalog) is MLContext mlContext)
            return mlContext;

        // Fallback: return new MLContext (loses GpuDeviceId, but doesn't crash)
        return new MLContext();
    }
}

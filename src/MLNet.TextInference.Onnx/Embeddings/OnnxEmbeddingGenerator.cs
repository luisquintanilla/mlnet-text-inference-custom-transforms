using Microsoft.Extensions.AI;
using Microsoft.ML;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// MEAI IEmbeddingGenerator wrapper around OnnxTextEmbeddingTransformer.
/// Allows using the ML.NET embedding transform via the standard MEAI abstraction.
/// </summary>
public sealed class OnnxEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextEmbeddingTransformer _transformer;
    private readonly bool _ownsTransformer;

    public OnnxEmbeddingGenerator(MLContext mlContext, OnnxTextEmbeddingTransformer transformer, bool ownsTransformer = false)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _transformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _ownsTransformer = ownsTransformer;
    }

    /// <summary>
    /// Creates an OnnxEmbeddingGenerator from a saved model file.
    /// </summary>
    public OnnxEmbeddingGenerator(MLContext mlContext, string modelPath)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _transformer = OnnxTextEmbeddingTransformer.Load(mlContext, modelPath);
        _ownsTransformer = true;
    }

    public EmbeddingGeneratorMetadata Metadata => new(
        "MLNet.TextInference.Onnx",
        null,
        Path.GetFileNameWithoutExtension(_transformer.Options.ModelPath),
        _transformer.EmbeddingDimension);

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var textList = values as IReadOnlyList<string> ?? values.ToList();
        var embeddings = _transformer.GenerateEmbeddings(textList);

        var result = new GeneratedEmbeddings<Embedding<float>>(
            embeddings.Select(e => new Embedding<float>(e)));

        return Task.FromResult(result);
    }

    public TService? GetService<TService>(object? key = null) where TService : class
    {
        return GetService(typeof(TService), key) as TService;
    }

    public object? GetService(Type serviceType, object? key = null)
    {
        if (serviceType == typeof(OnnxTextEmbeddingTransformer))
            return _transformer;

        if (serviceType == typeof(OnnxEmbeddingGenerator) || serviceType == typeof(IEmbeddingGenerator<string, Embedding<float>>))
            return this;

        return null;
    }

    public void Dispose()
    {
        if (_ownsTransformer)
            _transformer.Dispose();
    }
}

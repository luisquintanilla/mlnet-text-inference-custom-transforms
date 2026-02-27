using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that generates text embeddings using a local ONNX model.
/// Internally composes tokenization → ONNX inference → pooling using three sub-transforms.
/// </summary>
public sealed class OnnxTextEmbeddingTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextEmbeddingOptions _options;

    private readonly TextTokenizerTransformer _tokenizer;
    private readonly OnnxTextModelScorerTransformer _scorer;
    private readonly EmbeddingPoolingTransformer _pooler;

    public bool IsRowToRowMapper => true;

    internal OnnxTextEmbeddingOptions Options => _options;
    public int EmbeddingDimension => _scorer.HiddenDim;

    internal TextTokenizerTransformer Tokenizer => _tokenizer;
    internal OnnxTextModelScorerTransformer Scorer => _scorer;
    internal EmbeddingPoolingTransformer Pooler => _pooler;

    internal OnnxTextEmbeddingTransformer(
        MLContext mlContext,
        OnnxTextEmbeddingOptions options,
        TextTokenizerTransformer tokenizer,
        OnnxTextModelScorerTransformer scorer,
        EmbeddingPoolingTransformer pooler)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
        _scorer = scorer;
        _pooler = pooler;
    }

    /// <summary>
    /// ML.NET face: chains the three sub-transforms via IDataView.
    /// All lazy — no materialization until a cursor iterates.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var tokenized = _tokenizer.Transform(input);
        var scored = _scorer.Transform(tokenized);
        var pooled = _pooler.Transform(scored);
        return pooled;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var tokSchema = _tokenizer.GetOutputSchema(inputSchema);
        var scorerSchema = _scorer.GetOutputSchema(tokSchema);
        return _pooler.GetOutputSchema(scorerSchema);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        throw new NotSupportedException(
            "Row-to-row mapping is not supported in this prototype. " +
            "Use Transform() for batch processing.");
    }

    void ICanSaveModel.Save(ModelSaveContext ctx)
    {
        throw new NotSupportedException(
            "ML.NET native save is not supported. Use transformer.Save(path) instead.");
    }

    /// <summary>
    /// Generates embeddings for a list of texts directly (bypasses IDataView).
    /// Chains the three sub-transforms' direct faces for zero-overhead batch processing.
    /// </summary>
    internal float[][] GenerateEmbeddings(IReadOnlyList<string> texts)
    {
        if (texts.Count == 0)
            return [];

        var allEmbeddings = new List<float[]>(texts.Count);
        int batchSize = _options.BatchSize;

        for (int start = 0; start < texts.Count; start += batchSize)
        {
            int count = Math.Min(batchSize, texts.Count - start);
            var batchTexts = new List<string>(count);
            for (int i = start; i < start + count; i++)
                batchTexts.Add(texts[i]);

            // Chain direct faces
            var tokenized = _tokenizer.Tokenize(batchTexts);
            var scored = _scorer.Score(tokenized);
            var embeddings = _pooler.Pool(scored, tokenized.AttentionMasks);

            allEmbeddings.AddRange(embeddings);
        }

        return [.. allEmbeddings];
    }

    /// <summary>
    /// Saves the transformer to a self-contained zip file.
    /// </summary>
    public void Save(string path) => ModelPackager.Save(this, path);

    /// <summary>
    /// Loads a transformer from a saved zip file.
    /// </summary>
    public static OnnxTextEmbeddingTransformer Load(MLContext mlContext, string path)
        => ModelPackager.Load(mlContext, path);

    public void Dispose()
    {
        _scorer.Dispose();
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that classifies text using a local ONNX model.
/// Internally composes tokenization → ONNX inference → softmax classification.
/// </summary>
public sealed class OnnxTextClassificationTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextClassificationOptions _options;

    private readonly TextTokenizerTransformer _tokenizer;
    private readonly OnnxTextModelScorerTransformer _scorer;
    private readonly SoftmaxClassificationTransformer _classifier;

    public bool IsRowToRowMapper => true;

    internal OnnxTextClassificationOptions Options => _options;
    public int NumClasses => _classifier.NumClasses;
    public string[]? Labels => _classifier.Labels;

    internal TextTokenizerTransformer Tokenizer => _tokenizer;
    internal OnnxTextModelScorerTransformer Scorer => _scorer;
    internal SoftmaxClassificationTransformer Classifier => _classifier;

    internal OnnxTextClassificationTransformer(
        MLContext mlContext,
        OnnxTextClassificationOptions options,
        TextTokenizerTransformer tokenizer,
        OnnxTextModelScorerTransformer scorer,
        SoftmaxClassificationTransformer classifier)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
        _scorer = scorer;
        _classifier = classifier;
    }

    /// <summary>
    /// ML.NET face: chains the three sub-transforms via IDataView.
    /// All lazy — no materialization until a cursor iterates.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var tokenized = _tokenizer.Transform(input);
        var scored = _scorer.Transform(tokenized);
        var classified = _classifier.Transform(scored);
        return classified;
    }

    /// <summary>
    /// Direct face: classify texts without IDataView overhead.
    /// Chains the three sub-transforms' direct faces for zero-overhead batch processing.
    /// </summary>
    public ClassificationResult[] Classify(IReadOnlyList<string> texts)
    {
        if (texts.Count == 0)
            return [];

        var allResults = new List<ClassificationResult>(texts.Count);
        int batchSize = _options.BatchSize;

        for (int start = 0; start < texts.Count; start += batchSize)
        {
            int count = Math.Min(batchSize, texts.Count - start);
            var batchTexts = new List<string>(count);
            for (int i = start; i < start + count; i++)
                batchTexts.Add(texts[i]);

            var tokenized = _tokenizer.Tokenize(batchTexts);
            var scored = _scorer.Score(tokenized);
            var results = _classifier.Classify(scored);

            allResults.AddRange(results);
        }

        return [.. allResults];
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var tokSchema = _tokenizer.GetOutputSchema(inputSchema);
        var scorerSchema = _scorer.GetOutputSchema(tokSchema);
        return _classifier.GetOutputSchema(scorerSchema);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        throw new NotSupportedException(
            "Row-to-row mapping is not supported. Use Transform() for batch processing.");
    }

    void ICanSaveModel.Save(ModelSaveContext ctx)
    {
        throw new NotSupportedException(
            "ML.NET native save is not supported for classification transformers.");
    }

    public void Dispose()
    {
        _scorer.Dispose();
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that reranks query-document pairs using a cross-encoder ONNX model.
/// Internally composes tokenization (text-pair) → ONNX inference → sigmoid scoring.
/// </summary>
public sealed class OnnxRerankerTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxRerankerOptions _options;

    private readonly TextTokenizerTransformer _tokenizer;
    private readonly OnnxTextModelScorerTransformer _scorer;
    private readonly SigmoidScorerTransformer _sigmoid;

    public bool IsRowToRowMapper => true;

    internal OnnxRerankerOptions Options => _options;

    internal OnnxRerankerTransformer(
        MLContext mlContext,
        OnnxRerankerOptions options,
        TextTokenizerTransformer tokenizer,
        OnnxTextModelScorerTransformer scorer,
        SigmoidScorerTransformer sigmoid)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
        _scorer = scorer;
        _sigmoid = sigmoid;
    }

    /// <summary>
    /// ML.NET face: chains the three sub-transforms via IDataView.
    /// All lazy — no materialization until a cursor iterates.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var tokenized = _tokenizer.Transform(input);
        var scored = _scorer.Transform(tokenized);
        var sigmoided = _sigmoid.Transform(scored);
        return sigmoided;
    }

    /// <summary>
    /// Direct face: rerank query-document pairs.
    /// Returns sigmoid scores for each pair.
    /// </summary>
    internal float[] Rerank(IReadOnlyList<string> queries, IReadOnlyList<string> documents)
    {
        if (queries.Count == 0)
            return [];

        var allScores = new List<float>(queries.Count);
        int batchSize = _options.BatchSize;

        for (int start = 0; start < queries.Count; start += batchSize)
        {
            int count = Math.Min(batchSize, queries.Count - start);
            var batchQueries = new List<string>(count);
            var batchDocuments = new List<string>(count);
            for (int i = start; i < start + count; i++)
            {
                batchQueries.Add(queries[i]);
                batchDocuments.Add(documents[i]);
            }

            var tokenized = _tokenizer.Tokenize(batchQueries, batchDocuments);
            var scored = _scorer.Score(tokenized);
            var sigmoidScores = _sigmoid.Score(scored);

            allScores.AddRange(sigmoidScores);
        }

        return [.. allScores];
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var tokSchema = _tokenizer.GetOutputSchema(inputSchema);
        var scorerSchema = _scorer.GetOutputSchema(tokSchema);
        return _sigmoid.GetOutputSchema(scorerSchema);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    public void Dispose() => _scorer.Dispose();
}

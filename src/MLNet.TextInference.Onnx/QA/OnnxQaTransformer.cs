using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// End-to-end QA transformer. Chains tokenizer → multi-output scorer → QA span extractor.
/// Provides both ML.NET IDataView face and a direct answer extraction API.
/// </summary>
public sealed class OnnxQaTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxQaOptions _options;
    private readonly TextTokenizerTransformer _tokenizer;
    private readonly OnnxTextModelScorerTransformer _scorer;
    private readonly QaSpanExtractionTransformer _qaExtractor;

    public bool IsRowToRowMapper => true;

    internal OnnxQaTransformer(
        MLContext mlContext,
        OnnxQaOptions options,
        TextTokenizerTransformer tokenizer,
        OnnxTextModelScorerTransformer scorer,
        QaSpanExtractionTransformer qaExtractor)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
        _scorer = scorer;
        _qaExtractor = qaExtractor;
    }

    /// <summary>
    /// ML.NET face: chains the three transforms.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var tokenized = _tokenizer.Transform(input);
        var scored = _scorer.Transform(tokenized);
        return _qaExtractor.Transform(scored);
    }

    /// <summary>
    /// Direct face: answer questions given context passages.
    /// Returns one QaResult per question-context pair.
    /// </summary>
    public QaResult[] Answer(IReadOnlyList<string> questions, IReadOnlyList<string> contexts)
    {
        if (questions.Count == 0)
            return [];

        if (questions.Count != contexts.Count)
            throw new ArgumentException("questions and contexts must have the same length.");

        var batch = _tokenizer.Tokenize(questions, contexts);
        var multiOutputs = _scorer.ScoreMulti(batch);

        // multiOutputs[0] = start_logits, multiOutputs[1] = end_logits
        var startLogits = multiOutputs[0];
        var endLogits = multiOutputs.Length > 1 ? multiOutputs[1] : multiOutputs[0];

        return _qaExtractor.ExtractAnswers(
            startLogits, endLogits,
            batch.AttentionMasks,
            batch.TokenStartOffsets!,
            batch.TokenEndOffsets!,
            contexts.ToArray());
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var tokenizedSchema = _tokenizer.GetOutputSchema(inputSchema);
        var scoredSchema = _scorer.GetOutputSchema(tokenizedSchema);
        return _qaExtractor.GetOutputSchema(scoredSchema);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    public void Dispose() => _scorer.Dispose();
}

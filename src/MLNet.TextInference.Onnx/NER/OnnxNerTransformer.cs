using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// End-to-end NER transformer. Chains tokenizer → scorer → NER decoder.
/// Provides both ML.NET IDataView face and a direct extraction API.
/// </summary>
public sealed class OnnxNerTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxNerOptions _options;
    private readonly TextTokenizerTransformer _tokenizer;
    private readonly OnnxTextModelScorerTransformer _scorer;
    private readonly NerDecodingTransformer _nerDecoder;

    public bool IsRowToRowMapper => true;

    internal OnnxNerTransformer(
        MLContext mlContext,
        OnnxNerOptions options,
        TextTokenizerTransformer tokenizer,
        OnnxTextModelScorerTransformer scorer,
        NerDecodingTransformer nerDecoder)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
        _scorer = scorer;
        _nerDecoder = nerDecoder;
    }

    /// <summary>
    /// ML.NET face: chains the three transforms.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        var tokenized = _tokenizer.Transform(input);
        var scored = _scorer.Transform(tokenized);
        return _nerDecoder.Transform(scored);
    }

    /// <summary>
    /// Direct face: extract entities from a list of texts.
    /// </summary>
    public NerEntity[][] ExtractEntities(IReadOnlyList<string> texts)
    {
        var batch = _tokenizer.Tokenize(texts);
        var rawOutputs = _scorer.Score(batch);
        return _nerDecoder.DecodeEntities(
            rawOutputs, batch.AttentionMasks,
            batch.TokenStartOffsets!, batch.TokenEndOffsets!,
            texts.ToArray());
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var tokenizedSchema = _tokenizer.GetOutputSchema(inputSchema);
        var scoredSchema = _scorer.GetOutputSchema(tokenizedSchema);
        return _nerDecoder.GetOutputSchema(scoredSchema);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    public void Dispose() => _scorer.Dispose();
}

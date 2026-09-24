using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that extracts answer spans from QA model start/end logits.
/// Finds the best (start, end) token pair, maps to character offsets, and extracts answer text.
/// </summary>
public sealed class QaSpanExtractionTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly QaSpanExtractionOptions _options;

    public bool IsRowToRowMapper => true;

    internal QaSpanExtractionOptions Options => _options;

    internal QaSpanExtractionTransformer(MLContext mlContext, QaSpanExtractionOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        return new MappedDataView(input, GetRowToRowMapper(input.Schema));
    }

    /// <summary>
    /// Direct face: extract answers from batched model outputs.
    /// </summary>
    internal QaResult[] ExtractAnswers(
        float[][] startLogits, float[][] endLogits,
        long[][] attentionMasks,
        long[][] startOffsets, long[][] endOffsets,
        string[] texts)
    {
        var results = new QaResult[startLogits.Length];
        for (int i = 0; i < startLogits.Length; i++)
        {
            var candidates = ExtractSpans(
                startLogits[i], endLogits[i],
                attentionMasks[i],
                startOffsets[i], endOffsets[i],
                texts[i],
                _options.MaxAnswerLength, _options.TopK);
            results[i] = candidates.Length > 0 ? candidates[0] : new QaResult();
        }
        return results;
    }

    internal static QaResult[] ExtractSpans(
        float[] startLogits, float[] endLogits,
        long[] attentionMask,
        long[] startOffsets, long[] endOffsets,
        string text,
        int maxAnswerLength, int topK)
    {
        float nullScore = startLogits[0] + endLogits[0];

        int seqLen = startLogits.Length;
        var candidates = new List<(float score, int start, int end)>();

        for (int s = 1; s < seqLen; s++)
        {
            if (attentionMask[s] != 1) continue;
            for (int e = s; e < seqLen && e - s < maxAnswerLength; e++)
            {
                if (attentionMask[e] != 1) continue;
                float score = startLogits[s] + endLogits[e];
                candidates.Add((score, s, e));
            }
        }

        var topCandidates = candidates
            .OrderByDescending(c => c.score)
            .Take(topK)
            .ToList();

        var results = new List<QaResult>();
        foreach (var (score, start, end) in topCandidates)
        {
            // SQuAD 2.0: if best span score < null score, question is unanswerable
            if (score < nullScore)
            {
                results.Add(new QaResult { Answer = "", Score = 0f });
                continue;
            }

            int startChar = (int)startOffsets[start];
            int endChar = (int)endOffsets[end];

            startChar = Math.Max(0, Math.Min(startChar, text.Length));
            endChar = Math.Max(startChar, Math.Min(endChar, text.Length));

            string answer = text[startChar..endChar];
            results.Add(new QaResult
            {
                Answer = answer,
                Score = score,
                StartChar = startChar,
                EndChar = endChar
            });
        }

        if (results.Count == 0)
            results.Add(new QaResult { Answer = "", Score = 0f });

        return [.. results];
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        builder.AddColumn(_options.ScoreColumnName, NumberDataViewType.Single);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => new QaSpanExtractionRowToRowMapper(inputSchema, this);

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

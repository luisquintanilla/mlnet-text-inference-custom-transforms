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

    internal QaSpanExtractionTransformer(MLContext mlContext, QaSpanExtractionOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        return new QaDataView(input, _options);
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
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

/// <summary>
/// Wrapping IDataView that adds Answer and AnswerScore columns.
/// </summary>
internal sealed class QaDataView : IDataView
{
    private readonly IDataView _input;
    private readonly QaSpanExtractionOptions _options;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal QaDataView(IDataView input, QaSpanExtractionOptions options)
    {
        _input = input;
        _options = options;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);
        builder.AddColumn(options.OutputColumnName, TextDataViewType.Instance);
        builder.AddColumn(options.ScoreColumnName, NumberDataViewType.Single);
        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var upstreamColumns = columnsNeeded
            .Where(c => _input.Schema.GetColumnOrNull(c.Name) != null)
            .Select(c => _input.Schema[c.Name]);

        var required = new[]
        {
            _options.StartLogitsColumnName,
            _options.EndLogitsColumnName,
            _options.AttentionMaskColumnName,
            _options.TokenStartOffsetsColumnName,
            _options.TokenEndOffsetsColumnName,
            _options.TextColumnName
        };

        var allUpstream = upstreamColumns.ToList();
        foreach (var name in required)
        {
            var col = _input.Schema.GetColumnOrNull(name);
            if (col != null) allUpstream.Add(col.Value);
        }

        var inputCursor = _input.GetRowCursor(allUpstream.Distinct(), rand);
        return new QaCursor(this, inputCursor, _options);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor that extracts QA answer spans for each row.
/// </summary>
internal sealed class QaCursor : DataViewRowCursor
{
    private readonly QaDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly QaSpanExtractionOptions _options;

    private string _currentAnswer = "";
    private float _currentScore;

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _inputCursor.Position;
    public override long Batch => _inputCursor.Batch;

    internal QaCursor(QaDataView parent, DataViewRowCursor inputCursor, QaSpanExtractionOptions options)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _options = options;
    }

    public override bool MoveNext()
    {
        if (!_inputCursor.MoveNext())
            return false;

        // Read start logits
        var startGetter = _inputCursor.GetGetter<VBuffer<float>>(
            _inputCursor.Schema[_options.StartLogitsColumnName]);
        VBuffer<float> startBuffer = default;
        startGetter(ref startBuffer);
        var startLogits = startBuffer.DenseValues().ToArray();

        // Read end logits
        var endGetter = _inputCursor.GetGetter<VBuffer<float>>(
            _inputCursor.Schema[_options.EndLogitsColumnName]);
        VBuffer<float> endBuffer = default;
        endGetter(ref endBuffer);
        var endLogits = endBuffer.DenseValues().ToArray();

        // Read attention mask
        var maskGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[_options.AttentionMaskColumnName]);
        VBuffer<long> maskBuffer = default;
        maskGetter(ref maskBuffer);
        var attentionMask = maskBuffer.DenseValues().ToArray();

        // Read offsets
        var startOffGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[_options.TokenStartOffsetsColumnName]);
        VBuffer<long> startOffBuffer = default;
        startOffGetter(ref startOffBuffer);
        var startOffsets = startOffBuffer.DenseValues().ToArray();

        var endOffGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[_options.TokenEndOffsetsColumnName]);
        VBuffer<long> endOffBuffer = default;
        endOffGetter(ref endOffBuffer);
        var endOffsets = endOffBuffer.DenseValues().ToArray();

        // Read text
        var textGetter = _inputCursor.GetGetter<ReadOnlyMemory<char>>(
            _inputCursor.Schema[_options.TextColumnName]);
        ReadOnlyMemory<char> textValue = default;
        textGetter(ref textValue);
        string text = textValue.ToString();

        // Extract best answer span
        var spans = QaSpanExtractionTransformer.ExtractSpans(
            startLogits, endLogits, attentionMask,
            startOffsets, endOffsets, text,
            _options.MaxAnswerLength, _options.TopK);

        var best = spans[0];
        _currentAnswer = best.Answer;
        _currentScore = best.Score;

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Name == _options.OutputColumnName)
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
            {
                value = _currentAnswer.AsMemory();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        if (column.Name == _options.ScoreColumnName)
        {
            ValueGetter<float> getter = (ref float value) =>
            {
                value = _currentScore;
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // Passthrough to upstream
        var inputCol = _inputCursor.Schema.GetColumnOrNull(column.Name);
        if (inputCol != null)
            return _inputCursor.GetGetter<TValue>(inputCol.Value);

        throw new InvalidOperationException($"Unknown column: {column.Name}");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => _inputCursor.GetIdGetter();

    public override bool IsColumnActive(DataViewSchema.Column column) => true;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _inputCursor.Dispose();
        base.Dispose(disposing);
    }
}

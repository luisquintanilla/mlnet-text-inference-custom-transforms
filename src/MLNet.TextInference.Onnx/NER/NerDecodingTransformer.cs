using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that decodes BIO-tagged NER model output into entity spans.
/// Reads raw logits, applies softmax + argmax, then merges BIO tags into entities.
/// </summary>
public sealed class NerDecodingTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly NerDecodingOptions _options;

    public bool IsRowToRowMapper => true;

    internal NerDecodingTransformer(MLContext mlContext, NerDecodingOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        return new NerDataView(input, _options);
    }

    /// <summary>
    /// Direct face: decode entities from raw model outputs.
    /// </summary>
    internal NerEntity[][] DecodeEntities(
        float[][] rawOutputs,
        long[][] attentionMasks,
        long[][] startOffsets,
        long[][] endOffsets,
        string[] texts)
    {
        var results = new NerEntity[rawOutputs.Length][];
        int numLabels = _options.NumLabels!.Value;

        for (int i = 0; i < rawOutputs.Length; i++)
        {
            results[i] = DecodeRow(
                rawOutputs[i], attentionMasks[i],
                startOffsets[i], endOffsets[i],
                texts[i], numLabels);
        }

        return results;
    }

    internal NerEntity[] DecodeRow(
        float[] rawOutput,
        long[] attentionMask,
        long[] startOffsets,
        long[] endOffsets,
        string text,
        int numLabels)
    {
        int seqLen = attentionMask.Length;
        var entities = new List<NerEntity>();

        // Find last real token index (for SEP detection)
        int lastRealToken = -1;
        for (int t = seqLen - 1; t >= 0; t--)
        {
            if (attentionMask[t] == 1) { lastRealToken = t; break; }
        }

        string? currentType = null;
        int currentStart = 0;
        int currentEnd = 0;
        float currentScoreSum = 0;
        int currentTokenCount = 0;

        // Pre-allocate probs array outside the loop to avoid stackalloc in loop (CA2014)
        var probs = new float[numLabels];

        for (int t = 0; t < seqLen; t++)
        {
            if (attentionMask[t] == 0) break;

            // Skip CLS (index 0) and SEP (last real token)
            if (t == 0 || t == lastRealToken) continue;

            // Extract logits for this token
            int offset = t * numLabels;
            if (offset + numLabels > rawOutput.Length) break;

            var logits = rawOutput.AsSpan(offset, numLabels);

            // Softmax + argmax
            float maxLogit = float.MinValue;
            for (int l = 0; l < numLabels; l++)
                if (logits[l] > maxLogit) maxLogit = logits[l];

            float expSum = 0;
            for (int l = 0; l < numLabels; l++)
            {
                probs[l] = MathF.Exp(logits[l] - maxLogit);
                expSum += probs[l];
            }

            int argmax = 0;
            float maxProb = 0;
            for (int l = 0; l < numLabels; l++)
            {
                probs[l] /= expSum;
                if (probs[l] > maxProb) { maxProb = probs[l]; argmax = l; }
            }

            string label = _options.Labels[argmax];
            string prefix = label.Length >= 2 && label[1] == '-' ? label[..2] : "";
            string entityType = prefix.Length > 0 ? label[2..] : "";

            if (prefix == "B-")
            {
                // Flush current entity
                FlushEntity(entities, currentType, currentStart, currentEnd, currentScoreSum, currentTokenCount, text);

                // Start new entity
                currentType = entityType;
                currentStart = (int)startOffsets[t];
                currentEnd = (int)endOffsets[t];
                currentScoreSum = maxProb;
                currentTokenCount = 1;
            }
            else if (prefix == "I-")
            {
                if (currentType == entityType)
                {
                    // Continue current entity
                    currentEnd = (int)endOffsets[t];
                    currentScoreSum += maxProb;
                    currentTokenCount++;
                }
                else
                {
                    // Type mismatch or orphan I- tag → start new entity
                    FlushEntity(entities, currentType, currentStart, currentEnd, currentScoreSum, currentTokenCount, text);
                    currentType = entityType;
                    currentStart = (int)startOffsets[t];
                    currentEnd = (int)endOffsets[t];
                    currentScoreSum = maxProb;
                    currentTokenCount = 1;
                }
            }
            else
            {
                // O tag — flush current entity
                FlushEntity(entities, currentType, currentStart, currentEnd, currentScoreSum, currentTokenCount, text);
                currentType = null;
                currentTokenCount = 0;
            }
        }

        // Flush any remaining entity
        FlushEntity(entities, currentType, currentStart, currentEnd, currentScoreSum, currentTokenCount, text);

        return [.. entities];
    }

    private static void FlushEntity(
        List<NerEntity> entities,
        string? entityType,
        int startChar,
        int endChar,
        float scoreSum,
        int tokenCount,
        string text)
    {
        if (entityType == null || tokenCount == 0) return;

        // Clamp offsets to text bounds
        startChar = Math.Max(0, Math.Min(startChar, text.Length));
        endChar = Math.Max(startChar, Math.Min(endChar, text.Length));

        entities.Add(new NerEntity
        {
            EntityType = entityType,
            Word = text[startChar..endChar],
            StartChar = startChar,
            EndChar = endChar,
            Score = scoreSum / tokenCount
        });
    }

    internal static string SerializeEntities(NerEntity[] entities)
    {
        var items = entities.Select(e => new
        {
            entity = e.EntityType,
            word = e.Word,
            start = e.StartChar,
            end = e.EndChar,
            score = MathF.Round(e.Score, 4)
        });
        return JsonSerializer.Serialize(items);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

/// <summary>
/// Wrapping IDataView that adds NER entity column.
/// </summary>
internal sealed class NerDataView : IDataView
{
    private readonly IDataView _input;
    private readonly NerDecodingOptions _options;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal NerDataView(IDataView input, NerDecodingOptions options)
    {
        _input = input;
        _options = options;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);
        builder.AddColumn(options.OutputColumnName, TextDataViewType.Instance);
        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var upstreamColumns = columnsNeeded
            .Where(c => _input.Schema.GetColumnOrNull(c.Name) != null)
            .Select(c => _input.Schema[c.Name]);

        // Always need input columns for NER decoding
        var required = new[]
        {
            _options.InputColumnName,
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
        return new NerCursor(this, inputCursor, _options);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor that decodes NER entities for each row.
/// </summary>
internal sealed class NerCursor : DataViewRowCursor
{
    private readonly NerDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly NerDecodingOptions _options;
    private readonly NerDecodingTransformer _decoder;

    private string? _currentEntitiesJson;

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _inputCursor.Position;
    public override long Batch => _inputCursor.Batch;

    internal NerCursor(NerDataView parent, DataViewRowCursor inputCursor, NerDecodingOptions options)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _options = options;
        _decoder = new NerDecodingTransformer(null!, options);
    }

    public override bool MoveNext()
    {
        if (!_inputCursor.MoveNext())
            return false;

        int numLabels = _options.NumLabels!.Value;

        // Read raw output
        var rawGetter = _inputCursor.GetGetter<VBuffer<float>>(
            _inputCursor.Schema[_options.InputColumnName]);
        VBuffer<float> rawBuffer = default;
        rawGetter(ref rawBuffer);
        var rawOutput = rawBuffer.DenseValues().ToArray();

        // Read attention mask
        var maskGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[_options.AttentionMaskColumnName]);
        VBuffer<long> maskBuffer = default;
        maskGetter(ref maskBuffer);
        var attentionMask = maskBuffer.DenseValues().ToArray();

        // Read offsets
        var startGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[_options.TokenStartOffsetsColumnName]);
        VBuffer<long> startBuffer = default;
        startGetter(ref startBuffer);
        var startOffsets = startBuffer.DenseValues().ToArray();

        var endGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[_options.TokenEndOffsetsColumnName]);
        VBuffer<long> endBuffer = default;
        endGetter(ref endBuffer);
        var endOffsets = endBuffer.DenseValues().ToArray();

        // Read text
        var textGetter = _inputCursor.GetGetter<ReadOnlyMemory<char>>(
            _inputCursor.Schema[_options.TextColumnName]);
        ReadOnlyMemory<char> textValue = default;
        textGetter(ref textValue);
        string text = textValue.ToString();

        // Decode entities
        var entities = _decoder.DecodeRow(rawOutput, attentionMask, startOffsets, endOffsets, text, numLabels);
        _currentEntitiesJson = NerDecodingTransformer.SerializeEntities(entities);

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Name == _options.OutputColumnName)
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
            {
                value = (_currentEntitiesJson ?? "[]").AsMemory();
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

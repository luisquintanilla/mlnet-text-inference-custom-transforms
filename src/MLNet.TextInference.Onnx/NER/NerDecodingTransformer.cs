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

    internal NerDecodingOptions Options => _options;

    internal NerDecodingTransformer(MLContext mlContext, NerDecodingOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        return new MappedDataView(input, GetRowToRowMapper(input.Schema));
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

            // Softmax + argmax. The stable helper keeps padded labels out of the
            // slice and rejects malformed finite-logit inputs explicitly.
            StableSoftmax.Apply(logits, probs, numLabels);

            int argmax = 0;
            for (int l = 1; l < numLabels; l++)
                if (probs[l] > probs[argmax]) argmax = l;
            float maxProb = probs[argmax];

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
        => new NerDecodingRowToRowMapper(inputSchema, this);

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

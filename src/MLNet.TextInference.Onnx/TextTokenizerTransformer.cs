using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Tokenizers;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Batch of tokenized text. Used by the direct face to pass data between transforms
/// without IDataView overhead.
/// </summary>
internal sealed class TokenizedBatch
{
    public long[][] TokenIds { get; }
    public long[][] AttentionMasks { get; }
    public long[][]? TokenTypeIds { get; }
    public long[][]? TokenStartOffsets { get; }
    public long[][]? TokenEndOffsets { get; }
    public int SequenceLength { get; }

    public TokenizedBatch(long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds, int seqLen,
        long[][]? tokenStartOffsets = null, long[][]? tokenEndOffsets = null)
    {
        TokenIds = tokenIds;
        AttentionMasks = attentionMasks;
        TokenTypeIds = tokenTypeIds;
        TokenStartOffsets = tokenStartOffsets;
        TokenEndOffsets = tokenEndOffsets;
        SequenceLength = seqLen;
    }

    public int Count => TokenIds.Length;
}

/// <summary>
/// ML.NET ITransformer that tokenizes text into token IDs, attention masks,
/// and token type IDs. Produces fixed-length padded/truncated output.
///
/// Lazy evaluation: Transform() returns a wrapping IDataView that tokenizes
/// rows on-demand as a cursor iterates. No data is materialized upfront.
/// </summary>
public sealed class TextTokenizerTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly TextTokenizerOptions _options;
    private readonly Tokenizer _tokenizer;

    public bool IsRowToRowMapper => true;

    internal TextTokenizerOptions Options => _options;

    internal TextTokenizerTransformer(
        MLContext mlContext,
        TextTokenizerOptions options,
        Tokenizer tokenizer)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
    }

    /// <summary>
    /// ML.NET face: returns a wrapping IDataView. No computation happens here.
    /// Tokenization occurs lazily when a cursor iterates the returned IDataView.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        return new MappedDataView(input, GetRowToRowMapper(input.Schema));
    }

    /// <summary>
    /// Direct face: tokenize a list of texts without IDataView overhead.
    /// Used by the facade and MEAI generator.
    /// </summary>
    internal TokenizedBatch Tokenize(IReadOnlyList<string> texts)
    {
        int seqLen = _options.MaxTokenLength;
        var allTokenIds = new long[texts.Count][];
        var allAttentionMasks = new long[texts.Count][];
        var allTokenTypeIds = _options.OutputTokenTypeIds ? new long[texts.Count][] : null;
        var allStartOffsets = _options.OutputOffsets ? new long[texts.Count][] : null;
        var allEndOffsets = _options.OutputOffsets ? new long[texts.Count][] : null;

        for (int i = 0; i < texts.Count; i++)
        {
            var tokenIds = new long[seqLen];
            var attentionMask = new long[seqLen];
            var tokenTypeIds = _options.OutputTokenTypeIds ? new long[seqLen] : null;
            var startOffsets = _options.OutputOffsets ? new long[seqLen] : null;
            var endOffsets = _options.OutputOffsets ? new long[seqLen] : null;

            TextTokenizerTransformer.TokenizeSingle(
                _tokenizer,
                _options,
                texts[i],
                tokenIds,
                attentionMask,
                tokenTypeIds,
                startOffsets,
                endOffsets);

            allTokenIds[i] = tokenIds;
            allAttentionMasks[i] = attentionMask;
            if (allTokenTypeIds != null)
                allTokenTypeIds[i] = tokenTypeIds!;
            if (allStartOffsets != null)
                allStartOffsets[i] = startOffsets!;
            if (allEndOffsets != null)
                allEndOffsets[i] = endOffsets!;
        }

        return new TokenizedBatch(allTokenIds, allAttentionMasks, allTokenTypeIds, seqLen,
            allStartOffsets, allEndOffsets);
    }

    /// <summary>
    /// Core single-text tokenization shared by the direct and cursor paths.
    /// Special-token placement remains the responsibility of task-specific pair/layout helpers.
    /// </summary>
    internal static void TokenizeSingle(
        Tokenizer tokenizer,
        TextTokenizerOptions options,
        string text,
        long[] tokenIds,
        long[] attentionMask,
        long[]? tokenTypeIds,
        long[]? startOffsets,
        long[]? endOffsets)
    {
        int sequenceLength = options.MaxTokenLength;
        if (tokenIds.Length != sequenceLength ||
            attentionMask.Length != sequenceLength ||
            (tokenTypeIds is not null && tokenTypeIds.Length != sequenceLength) ||
            (startOffsets is not null && startOffsets.Length != sequenceLength) ||
            (endOffsets is not null && endOffsets.Length != sequenceLength))
        {
            throw new ArgumentException(
                "Single-text tokenization buffers must match MaxTokenLength.");
        }

        Array.Clear(tokenIds);
        Array.Clear(attentionMask);
        if (tokenTypeIds is not null)
            Array.Clear(tokenTypeIds);
        if (startOffsets is not null)
            Array.Clear(startOffsets);
        if (endOffsets is not null)
            Array.Clear(endOffsets);

        var encodedTokens = TokenizerEncoding.EncodeTokens(tokenizer, text);
        int count = Math.Min(encodedTokens.Count, sequenceLength);
        for (int index = 0; index < count; index++)
        {
            var encoded = encodedTokens[index];
            tokenIds[index] = encoded.Id;
            attentionMask[index] = 1;
            if (startOffsets is not null && endOffsets is not null)
            {
                startOffsets[index] = encoded.Offset.Start.Value;
                endOffsets[index] = encoded.Offset.End.Value;
            }
        }
    }

    /// <summary>
    /// Direct face: tokenize text pairs for cross-encoder models.
    /// Produces [BOS] A [SEP] B [SEP] with proper token_type_ids.
    /// When OutputOffsets is true, records character offsets for B segment tokens.
    /// </summary>
    internal TokenizedBatch Tokenize(IReadOnlyList<string> textsA, IReadOnlyList<string> textsB)
    {
        if (textsA.Count != textsB.Count)
            throw new ArgumentException("textsA and textsB must have the same length.");

        int seqLen = _options.MaxTokenLength;
        var allTokenIds = new long[textsA.Count][];
        var allAttentionMasks = new long[textsA.Count][];
        var allTokenTypeIds = new long[textsA.Count][];
        var allStartOffsets = _options.OutputOffsets ? new long[textsA.Count][] : null;
        var allEndOffsets = _options.OutputOffsets ? new long[textsA.Count][] : null;

        for (int i = 0; i < textsA.Count; i++)
        {
            var tokenIds = new long[seqLen];
            var attentionMask = new long[seqLen];
            var tokenTypeIds = new long[seqLen];
            long[]? startOffsets = _options.OutputOffsets ? new long[seqLen] : null;
            long[]? endOffsets = _options.OutputOffsets ? new long[seqLen] : null;

            TokenizePair(_tokenizer, _options, textsA[i], textsB[i],
                tokenIds, attentionMask, tokenTypeIds, startOffsets, endOffsets);

            allTokenIds[i] = tokenIds;
            allAttentionMasks[i] = attentionMask;
            allTokenTypeIds[i] = tokenTypeIds;
            if (allStartOffsets != null) allStartOffsets[i] = startOffsets!;
            if (allEndOffsets != null) allEndOffsets[i] = endOffsets!;
        }

        return new TokenizedBatch(allTokenIds, allAttentionMasks, allTokenTypeIds, seqLen,
            allStartOffsets, allEndOffsets);
    }

    /// <summary>
    /// Core text-pair tokenization: [BOS] A [SEP] (SEP)? B [SEP].
    /// Uses EncodeToTokens (which never auto-injects special tokens for any tokenizer type)
    /// and manually injects BOS/SEP tokens for a uniform approach across BERT, BPE, and SentencePiece.
    /// </summary>
    internal static void TokenizePair(
        Tokenizer tokenizer, TextTokenizerOptions options,
        string textA, string textB,
        long[] tokenIds, long[] attentionMask, long[] tokenTypeIds,
        long[]? startOffsets, long[]? endOffsets)
    {
        int seqLen = options.MaxTokenLength;
        int bosId = options.BosTokenId
            ?? throw new InvalidOperationException(
                "Text-pair tokenization requires special token IDs (BOS/CLS and SEP). " +
                "Load the tokenizer from a directory containing tokenizer_config.json.");
        int sepId = options.SepTokenId
            ?? throw new InvalidOperationException(
                "Text-pair tokenization requires special token IDs (BOS/CLS and SEP). " +
                "Load the tokenizer from a directory containing tokenizer_config.json.");

        // EncodeToTokens never adds special tokens for any tokenizer type
        var encodedA = TokenizerEncoding.EncodeTokens(tokenizer, textA);
        var encodedB = TokenizerEncoding.EncodeTokens(tokenizer, textB);

        // Build: [BOS] A_tokens [SEP] (SEP if double) B_tokens [SEP]
        var combined = new List<int>(seqLen);
        combined.Add(bosId);

        for (int j = 0; j < encodedA.Count; j++)
            combined.Add(encodedA[j].Id);

        combined.Add(sepId);

        if (options.DoubleSeparator)
            combined.Add(sepId);

        // Boundary between segment A and segment B for token_type_ids.
        // Both separators (if double) belong to the A side.
        int firstSepIdx = combined.Count - 1;

        int bStartIdx = combined.Count;
        for (int j = 0; j < encodedB.Count; j++)
            combined.Add(encodedB[j].Id);

        combined.Add(sepId);

        if (combined.Count > seqLen)
            combined.RemoveRange(seqLen, combined.Count - seqLen);

        for (int s = 0; s < combined.Count; s++)
        {
            tokenIds[s] = combined[s];
            attentionMask[s] = 1;
            tokenTypeIds[s] = s <= firstSepIdx ? 0 : 1;
        }

        // Record B segment character offsets (for QA answer extraction)
        if (startOffsets != null && endOffsets != null)
        {
            for (int bIdx = 0; bIdx < encodedB.Count; bIdx++)
            {
                int combinedIdx = bStartIdx + bIdx;
                if (combinedIdx >= seqLen) break;
                startOffsets[combinedIdx] = encodedB[bIdx].Offset.Start.Value;
                endOffsets[combinedIdx] = encodedB[bIdx].Offset.End.Value;
            }
        }
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);

        var seqLen = _options.MaxTokenLength;
        builder.AddColumn(_options.TokenIdsColumnName,
            new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        builder.AddColumn(_options.AttentionMaskColumnName,
            new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        if (_options.OutputTokenTypeIds)
            builder.AddColumn(_options.TokenTypeIdsColumnName,
                new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        if (_options.OutputOffsets)
        {
            builder.AddColumn(_options.TokenStartOffsetsColumnName,
                new VectorDataViewType(NumberDataViewType.Int64, seqLen));
            builder.AddColumn(_options.TokenEndOffsetsColumnName,
                new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        }

        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => new TextTokenizerRowToRowMapper(inputSchema, this);

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

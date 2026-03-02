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
        return new TokenizerDataView(input, _tokenizer, _options);
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

            if (_options.OutputOffsets)
            {
                var startOffsets = new long[seqLen];
                var endOffsets = new long[seqLen];

                var encodedTokens = _tokenizer.EncodeToTokens(texts[i], out _);
                int count = Math.Min(encodedTokens.Count, seqLen);
                for (int s = 0; s < count; s++)
                {
                    tokenIds[s] = encodedTokens[s].Id;
                    attentionMask[s] = 1;
                    startOffsets[s] = encodedTokens[s].Offset.Start.Value;
                    endOffsets[s] = encodedTokens[s].Offset.End.Value;
                }

                allStartOffsets![i] = startOffsets;
                allEndOffsets![i] = endOffsets;
            }
            else
            {
                var tokens = _tokenizer.EncodeToIds(texts[i], seqLen, out _, out _);
                for (int s = 0; s < tokens.Count && s < seqLen; s++)
                {
                    tokenIds[s] = tokens[s];
                    attentionMask[s] = 1;
                }
            }

            allTokenIds[i] = tokenIds;
            allAttentionMasks[i] = attentionMask;
            if (allTokenTypeIds != null)
                allTokenTypeIds[i] = tokenTypeIds!;
        }

        return new TokenizedBatch(allTokenIds, allAttentionMasks, allTokenTypeIds, seqLen,
            allStartOffsets, allEndOffsets);
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
        var encodedA = tokenizer.EncodeToTokens(textA, out _);
        var encodedB = tokenizer.EncodeToTokens(textB, out _);

        // Build: [BOS] A_tokens [SEP] (SEP if double) B_tokens [SEP]
        var combined = new List<int>(seqLen);
        combined.Add(bosId);

        for (int j = 0; j < encodedA.Count; j++)
            combined.Add(encodedA[j].Id);

        combined.Add(sepId);
        int firstSepIdx = combined.Count - 1;

        if (options.DoubleSeparator)
            combined.Add(sepId);

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
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

/// <summary>
/// Wrapping IDataView that adds tokenized columns to the upstream schema.
/// No data is materialized — tokenization happens in the cursor.
/// </summary>
internal sealed class TokenizerDataView : IDataView
{
    private readonly IDataView _input;
    private readonly Tokenizer _tokenizer;
    private readonly TextTokenizerOptions _options;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal TokenizerDataView(IDataView input, Tokenizer tokenizer, TextTokenizerOptions options)
    {
        _input = input;
        _tokenizer = tokenizer;
        _options = options;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);

        int seqLen = options.MaxTokenLength;
        builder.AddColumn(options.TokenIdsColumnName,
            new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        builder.AddColumn(options.AttentionMaskColumnName,
            new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        if (options.OutputTokenTypeIds)
            builder.AddColumn(options.TokenTypeIdsColumnName,
                new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        if (options.OutputOffsets)
        {
            builder.AddColumn(options.TokenStartOffsetsColumnName,
                new VectorDataViewType(NumberDataViewType.Int64, seqLen));
            builder.AddColumn(options.TokenEndOffsetsColumnName,
                new VectorDataViewType(NumberDataViewType.Int64, seqLen));
        }

        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var upstreamColumns = columnsNeeded
            .Where(c => _input.Schema.GetColumnOrNull(c.Name) != null)
            .Select(c => _input.Schema[c.Name]);

        // Always need the text column(s) for tokenization
        var textCol = _input.Schema[_options.InputColumnName];
        var allUpstream = upstreamColumns.Append(textCol);

        if (_options.SecondInputColumnName != null)
        {
            var textCol2 = _input.Schema[_options.SecondInputColumnName];
            allUpstream = allUpstream.Append(textCol2);
        }

        var inputCursor = _input.GetRowCursor(allUpstream.Distinct(), rand);
        return new TokenizerCursor(this, inputCursor, _tokenizer, _options);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor that tokenizes one row at a time from the upstream input cursor.
/// Tokenization is cheap (~microseconds per row), so no batching is needed.
/// </summary>
internal sealed class TokenizerCursor : DataViewRowCursor
{
    private readonly TokenizerDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly Tokenizer _tokenizer;
    private readonly TextTokenizerOptions _options;

    private long[]? _currentTokenIds;
    private long[]? _currentAttentionMask;
    private long[]? _currentTokenTypeIds;
    private long[]? _currentStartOffsets;
    private long[]? _currentEndOffsets;

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _inputCursor.Position;
    public override long Batch => _inputCursor.Batch;

    internal TokenizerCursor(
        TokenizerDataView parent,
        DataViewRowCursor inputCursor,
        Tokenizer tokenizer,
        TextTokenizerOptions options)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _tokenizer = tokenizer;
        _options = options;
    }

    public override bool MoveNext()
    {
        if (!_inputCursor.MoveNext())
            return false;

        var textCol = _inputCursor.Schema[_options.InputColumnName];
        var getter = _inputCursor.GetGetter<ReadOnlyMemory<char>>(textCol);
        ReadOnlyMemory<char> textValue = default;
        getter(ref textValue);
        string text = textValue.ToString();

        int seqLen = _options.MaxTokenLength;
        _currentTokenIds = new long[seqLen];
        _currentAttentionMask = new long[seqLen];
        _currentTokenTypeIds = _options.OutputTokenTypeIds ? new long[seqLen] : null;

        if (_options.SecondInputColumnName != null)
        {
            // Text-pair tokenization via shared helper
            var textCol2 = _inputCursor.Schema[_options.SecondInputColumnName];
            var getter2 = _inputCursor.GetGetter<ReadOnlyMemory<char>>(textCol2);
            ReadOnlyMemory<char> textValue2 = default;
            getter2(ref textValue2);
            string text2 = textValue2.ToString();

            _currentTokenTypeIds ??= new long[seqLen];
            if (_options.OutputOffsets)
            {
                _currentStartOffsets = new long[seqLen];
                _currentEndOffsets = new long[seqLen];
            }

            TextTokenizerTransformer.TokenizePair(
                _tokenizer, _options, text, text2,
                _currentTokenIds, _currentAttentionMask, _currentTokenTypeIds,
                _currentStartOffsets, _currentEndOffsets);
        }
        else if (_options.OutputOffsets)
        {
            _currentStartOffsets = new long[seqLen];
            _currentEndOffsets = new long[seqLen];

            var encodedTokens = _tokenizer.EncodeToTokens(text, out _);
            int count = Math.Min(encodedTokens.Count, seqLen);
            for (int s = 0; s < count; s++)
            {
                _currentTokenIds[s] = encodedTokens[s].Id;
                _currentAttentionMask[s] = 1;
                _currentStartOffsets[s] = encodedTokens[s].Offset.Start.Value;
                _currentEndOffsets[s] = encodedTokens[s].Offset.End.Value;
            }
        }
        else
        {
            // Single-text tokenization (existing path)
            var tokens = _tokenizer.EncodeToIds(text, seqLen, out _, out _);
            for (int s = 0; s < tokens.Count && s < seqLen; s++)
            {
                _currentTokenIds[s] = tokens[s];
                _currentAttentionMask[s] = 1;
            }
        }

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        // For input passthrough columns, delegate to upstream cursor
        var inputCol = _inputCursor.Schema.GetColumnOrNull(column.Name);
        if (inputCol != null && column.Name != _options.TokenIdsColumnName
            && column.Name != _options.AttentionMaskColumnName
            && column.Name != _options.TokenTypeIdsColumnName
            && column.Name != _options.TokenStartOffsetsColumnName
            && column.Name != _options.TokenEndOffsetsColumnName)
        {
            return _inputCursor.GetGetter<TValue>(inputCol.Value);
        }

        // For tokenized output columns, return computed values
        if (column.Name == _options.TokenIdsColumnName)
            return MakeVBufferGetter<TValue>(() => _currentTokenIds!);
        if (column.Name == _options.AttentionMaskColumnName)
            return MakeVBufferGetter<TValue>(() => _currentAttentionMask!);
        if (column.Name == _options.TokenTypeIdsColumnName)
            return MakeVBufferGetter<TValue>(() => _currentTokenTypeIds ?? new long[_options.MaxTokenLength]);
        if (column.Name == _options.TokenStartOffsetsColumnName)
            return MakeVBufferGetter<TValue>(() => _currentStartOffsets ?? new long[_options.MaxTokenLength]);
        if (column.Name == _options.TokenEndOffsetsColumnName)
            return MakeVBufferGetter<TValue>(() => _currentEndOffsets ?? new long[_options.MaxTokenLength]);

        throw new InvalidOperationException($"Unknown column: {column.Name}");
    }

    private static ValueGetter<TValue> MakeVBufferGetter<TValue>(Func<long[]> dataSource)
    {
        ValueGetter<VBuffer<long>> getter = (ref VBuffer<long> value) =>
        {
            var data = dataSource();
            var editor = VBufferEditor.Create(ref value, data.Length);
            data.AsSpan().CopyTo(editor.Values);
            value = editor.Commit();
        };
        return (ValueGetter<TValue>)(object)getter;
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

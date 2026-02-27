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
    public int SequenceLength { get; }

    public TokenizedBatch(long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds, int seqLen)
    {
        TokenIds = tokenIds;
        AttentionMasks = attentionMasks;
        TokenTypeIds = tokenTypeIds;
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

        for (int i = 0; i < texts.Count; i++)
        {
            var tokenIds = new long[seqLen];
            var attentionMask = new long[seqLen];
            var tokenTypeIds = _options.OutputTokenTypeIds ? new long[seqLen] : null;

            var tokens = _tokenizer.EncodeToIds(texts[i], seqLen, out _, out _);

            for (int s = 0; s < tokens.Count && s < seqLen; s++)
            {
                tokenIds[s] = tokens[s];
                attentionMask[s] = 1;
            }

            allTokenIds[i] = tokenIds;
            allAttentionMasks[i] = attentionMask;
            if (allTokenTypeIds != null)
                allTokenTypeIds[i] = tokenTypeIds!;
        }

        return new TokenizedBatch(allTokenIds, allAttentionMasks, allTokenTypeIds, seqLen);
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

        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var upstreamColumns = columnsNeeded
            .Where(c => _input.Schema.GetColumnOrNull(c.Name) != null)
            .Select(c => _input.Schema[c.Name]);

        // Always need the text column for tokenization
        var textCol = _input.Schema[_options.InputColumnName];
        var allUpstream = upstreamColumns.Append(textCol).Distinct();

        var inputCursor = _input.GetRowCursor(allUpstream, rand);
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

        var tokens = _tokenizer.EncodeToIds(text, seqLen, out _, out _);
        for (int s = 0; s < tokens.Count && s < seqLen; s++)
        {
            _currentTokenIds[s] = tokens[s];
            _currentAttentionMask[s] = 1;
        }

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        // For input passthrough columns, delegate to upstream cursor
        var inputCol = _inputCursor.Schema.GetColumnOrNull(column.Name);
        if (inputCol != null && column.Name != _options.TokenIdsColumnName
            && column.Name != _options.AttentionMaskColumnName
            && column.Name != _options.TokenTypeIdsColumnName)
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

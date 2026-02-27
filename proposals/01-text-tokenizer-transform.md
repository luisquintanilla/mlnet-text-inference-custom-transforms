# Transform 1: TextTokenizerEstimator / TextTokenizerTransformer

## Purpose

Tokenizes text into token IDs, attention masks, and token type IDs suitable for any transformer-based ONNX model. This is the **universal first step** for any transformer task (embeddings, classification, NER, QA, etc.).

## Files to Create

| File | Contents |
|------|----------|
| `src/MLNet.TextInference.Onnx/TextTokenizerEstimator.cs` | Estimator + options class |
| `src/MLNet.TextInference.Onnx/TextTokenizerTransformer.cs` | Transformer |

## Options Class

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// Configuration for the text tokenizer transform.
/// Provide either <see cref="Tokenizer"/> (a pre-constructed instance) or
/// <see cref="TokenizerPath"/> (a file/directory to auto-load). If both are set,
/// <see cref="Tokenizer"/> takes precedence.
/// </summary>
public class TextTokenizerOptions
{
    /// <summary>
    /// A pre-constructed tokenizer instance. Use this when working with
    /// tokenizer formats that LoadTokenizer doesn't support, or when
    /// sharing a tokenizer across multiple estimators.
    /// Takes precedence over <see cref="TokenizerPath"/> if both are set.
    /// </summary>
    public Tokenizer? Tokenizer { get; set; }

    /// <summary>
    /// Path to tokenizer artifacts. Can be:
    /// - A directory containing tokenizer_config.json (HuggingFace auto-detect)
    /// - A tokenizer_config.json file directly
    /// - A vocab file: .txt (BERT/WordPiece), .model (SentencePiece)
    /// Used only when Tokenizer is not set.
    /// </summary>
    public string? TokenizerPath { get; set; }

    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output token IDs column. Default: "TokenIds".</summary>
    public string TokenIdsColumnName { get; set; } = "TokenIds";

    /// <summary>Name of the output attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Name of the output token type IDs column. Default: "TokenTypeIds".</summary>
    public string TokenTypeIdsColumnName { get; set; } = "TokenTypeIds";

    /// <summary>
    /// Maximum number of tokens per input text.
    /// Texts are truncated to this length; shorter texts are zero-padded.
    /// Default: 128.
    /// </summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>
    /// Whether to output the token type IDs column.
    /// Set to false for models that don't use segment embeddings.
    /// Default: true.
    /// </summary>
    public bool OutputTokenTypeIds { get; set; } = true;
}
```

## Estimator

```csharp
namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET IEstimator that creates a TextTokenizerTransformer.
/// Trivial estimator — nothing to learn from training data.
/// Fit() validates the input schema and loads the tokenizer.
/// </summary>
public sealed class TextTokenizerEstimator : IEstimator<TextTokenizerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly TextTokenizerOptions _options;

    public TextTokenizerEstimator(MLContext mlContext, TextTokenizerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (options.Tokenizer == null && options.TokenizerPath == null)
            throw new ArgumentException(
                "Either Tokenizer or TokenizerPath must be provided.", nameof(options));

        if (options.Tokenizer == null)
        {
            var path = options.TokenizerPath!;
            if (!File.Exists(path) && !Directory.Exists(path))
                throw new FileNotFoundException(
                    $"Tokenizer path not found: {path}");
        }
    }

    public TextTokenizerTransformer Fit(IDataView input)
    {
        // Validate input schema has the text column
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // Load tokenizer (smart resolution: directory, config, or vocab file)
        var tokenizer = _options.Tokenizer ?? LoadTokenizer(_options.TokenizerPath!);

        return new TextTokenizerTransformer(_mlContext, _options, tokenizer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Validate input column exists and is text
        var inputCol = inputSchema.FirstOrDefault(c => c.Name == _options.InputColumnName);
        if (inputCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        if (inputCol.ItemType != TextDataViewType.Instance)
            throw new ArgumentException(
                $"Column '{_options.InputColumnName}' must be of type Text.");

        // Build output schema: input columns + token columns
        var result = inputSchema.ToDictionary(x => x.Name);

        // Add TokenIds, AttentionMask, TokenTypeIds as Vector<Int64> columns
        // (uses same reflection workaround as current GetOutputSchema)
        AddVectorColumn(result, _options.TokenIdsColumnName, NumberDataViewType.Int64);
        AddVectorColumn(result, _options.AttentionMaskColumnName, NumberDataViewType.Int64);
        if (_options.OutputTokenTypeIds)
            AddVectorColumn(result, _options.TokenTypeIdsColumnName, NumberDataViewType.Int64);

        return new SchemaShape(result.Values);
    }

    // Smart tokenizer resolution: handles directories, config files, and vocab files.
    // See LoadTokenizer implementation for full resolution rules.
    internal static Tokenizer LoadTokenizer(string path) { /* smart resolution logic */ }

    private static void AddVectorColumn(
        Dictionary<string, SchemaShape.Column> schema,
        string name,
        DataViewType itemType)
    {
        // Same reflection workaround as current GetOutputSchema
        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var col = (SchemaShape.Column)colCtor.Invoke([
            name,
            SchemaShape.Column.VectorKind.Vector,
            itemType,
            false,
            (SchemaShape?)null
        ]);
        schema[name] = col;
    }
}
```

## Transformer

```csharp
namespace MLNet.TextInference.Onnx;

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
```

## Lazy IDataView and Cursor

```csharp
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

        // Build schema: input columns + token columns
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
        // Determine which upstream columns are needed
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
        // Single cursor — tokenization is cheap, parallelism not needed
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

    // Current row's tokenized output (computed on MoveNext)
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

        // Read text from upstream cursor
        var textCol = _inputCursor.Schema[_options.InputColumnName];
        var getter = _inputCursor.GetGetter<ReadOnlyMemory<char>>(textCol);
        ReadOnlyMemory<char> textValue = default;
        getter(ref textValue);
        string text = textValue.ToString();

        // Tokenize this single row
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
        // Cast to ValueGetter<VBuffer<long>> and reinterpret
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
```

## Direct Face Data Transfer Type

```csharp
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
```

## Code to Extract From Existing Files

| Source | What to Extract | Target |
|--------|----------------|--------|
| `OnnxTextEmbeddingEstimator.LoadTokenizer()` | Tokenizer loading logic (expanded with smart resolution) | `TextTokenizerEstimator.LoadTokenizer()` |
| `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 145-154 | Tokenization loop | `TextTokenizerTransformer.Tokenize()` (direct face) and `TokenizerCursor.MoveNext()` |
| `OnnxTextEmbeddingTransformer.ReadTextColumn()` | Text column reading pattern | `TokenizerCursor.MoveNext()` (cursor-based, per-row) |

## Column Passthrough Strategy

The lazy `TokenizerDataView` schema includes ALL input columns plus the new token columns. The cursor delegates getter calls for input columns to the upstream cursor — true passthrough with zero copy. Only the new token columns are computed by the cursor.

This is simpler and more correct than the eager approach (which had to explicitly enumerate and copy input columns).

## Approach D Migration Notes

When migrating to ML.NET, this transform becomes:

```
TextTokenizerTransformer : OneToOneTransformerBase
  Mapper : OneToOneMapperBase
    MakeGetter() → the tokenization logic from TokenizerCursor.MoveNext()
    GetOutputColumnsCore() → column definitions
    SaveModel() → serialize tokenizer vocab + options
```

The `TokenizerDataView`, `TokenizerCursor`, and all schema/getter boilerplate get **deleted** — the base class handles cursor creation, schema propagation, column passthrough, and threading automatically.

## Acceptance Criteria

1. `TextTokenizerEstimator` can be created with a valid tokenizer path (file, directory, or config)
2. `Fit()` validates the input schema has the text column
3. `Transform()` returns a wrapping IDataView (no materialization)
4. Iterating the cursor produces TokenIds, AttentionMask, TokenTypeIds columns
5. Token arrays are padded to MaxTokenLength with zeros
6. Attention masks are 1 for real tokens, 0 for padding
7. Input columns are passed through via cursor delegation (zero copy)
8. `Tokenize()` (direct face) returns the same results without IDataView overhead
9. Works with vocab.txt (BertTokenizer), SentencePiece (.model), BPE (vocab.json+merges.txt)
10. Works with HuggingFace model directories containing tokenizer_config.json
11. Memory usage is O(1) per row, not O(N) for N total rows
12. Pre-constructed `Tokenizer` instance can be provided as an escape hatch

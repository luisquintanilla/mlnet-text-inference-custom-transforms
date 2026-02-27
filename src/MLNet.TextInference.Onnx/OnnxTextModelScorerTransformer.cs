using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that runs ONNX inference on tokenized text inputs.
/// Task-agnostic — outputs the raw model tensor for downstream post-processing.
///
/// Lazy evaluation with lookahead batching: Transform() returns a wrapping IDataView.
/// The cursor reads ahead BatchSize rows from the upstream tokenizer cursor,
/// runs a single ONNX session.Run() call, then serves results one at a time.
/// </summary>
public sealed class OnnxTextModelScorerTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextModelScorerOptions _options;
    private readonly InferenceSession _session;
    private readonly OnnxModelMetadata _metadata;

    public bool IsRowToRowMapper => true;

    internal OnnxTextModelScorerOptions Options => _options;

    /// <summary>Hidden dimension of the model output.</summary>
    public int HiddenDim => _metadata.HiddenDim;

    /// <summary>Whether the model outputs pre-pooled embeddings (e.g., sentence_embedding).</summary>
    public bool HasPooledOutput => _metadata.HasPooledOutput;

    internal OnnxModelMetadata Metadata => _metadata;

    internal OnnxTextModelScorerTransformer(
        MLContext mlContext,
        OnnxTextModelScorerOptions options,
        InferenceSession session,
        OnnxModelMetadata metadata)
    {
        _mlContext = mlContext;
        _options = options;
        _session = session;
        _metadata = metadata;
    }

    /// <summary>
    /// ML.NET face: returns a wrapping IDataView. No computation happens here.
    /// ONNX inference occurs lazily in the cursor via lookahead batching.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        return new ScorerDataView(input, this);
    }

    /// <summary>
    /// Direct face: run ONNX inference on pre-tokenized input without IDataView overhead.
    /// </summary>
    internal float[][] Score(TokenizedBatch batch)
    {
        return Score(batch.TokenIds, batch.AttentionMasks, batch.TokenTypeIds);
    }

    /// <summary>
    /// Runs ONNX inference in batches.
    /// </summary>
    internal float[][] Score(long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds)
    {
        int totalRows = tokenIds.Length;
        int batchSize = _options.BatchSize;
        int seqLen = _options.MaxTokenLength;
        var allOutputs = new List<float[]>(totalRows);

        for (int start = 0; start < totalRows; start += batchSize)
        {
            int count = Math.Min(batchSize, totalRows - start);
            var batchOutputs = RunOnnxBatch(
                tokenIds, attentionMasks, tokenTypeIds,
                start, count, seqLen);
            allOutputs.AddRange(batchOutputs);
        }

        return [.. allOutputs];
    }

    /// <summary>
    /// Runs a single ONNX inference batch. Core inference logic shared by
    /// the direct face and the cursor's lookahead batching.
    /// </summary>
    internal float[][] RunOnnxBatch(
        long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds,
        int startIdx, int batchSize, int seqLen)
    {
        var idsArray = new long[batchSize * seqLen];
        var maskArray = new long[batchSize * seqLen];
        var typeIdsArray = _metadata.TokenTypeIdsName != null ? new long[batchSize * seqLen] : null;

        for (int b = 0; b < batchSize; b++)
        {
            Array.Copy(tokenIds[startIdx + b], 0, idsArray, b * seqLen, seqLen);
            Array.Copy(attentionMasks[startIdx + b], 0, maskArray, b * seqLen, seqLen);
            if (typeIdsArray != null && tokenTypeIds != null)
                Array.Copy(tokenTypeIds[startIdx + b], 0, typeIdsArray, b * seqLen, seqLen);
        }

        var inputs = new Dictionary<string, OrtValue>
        {
            [_metadata.InputIdsName] = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]),
            [_metadata.AttentionMaskName] = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen])
        };

        if (_metadata.TokenTypeIdsName != null && typeIdsArray != null)
            inputs[_metadata.TokenTypeIdsName] = OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);

        try
        {
            using var results = _session.Run(new RunOptions(), inputs, [_metadata.OutputTensorName]);
            var output = results[0];
            var outputSpan = output.GetTensorDataAsSpan<float>();

            var batchOutputs = new float[batchSize][];

            if (_metadata.OutputRank == 2)
            {
                for (int b = 0; b < batchSize; b++)
                    batchOutputs[b] = outputSpan.Slice(b * _metadata.HiddenDim, _metadata.HiddenDim).ToArray();
            }
            else
            {
                int rowSize = seqLen * _metadata.HiddenDim;
                for (int b = 0; b < batchSize; b++)
                    batchOutputs[b] = outputSpan.Slice(b * rowSize, rowSize).ToArray();
            }

            return batchOutputs;
        }
        finally
        {
            foreach (var ortValue in inputs.Values)
                ortValue.Dispose();
        }
    }

    /// <summary>
    /// Runs a single ONNX inference batch returning all configured outputs.
    /// Returns [numOutputs][batchSize][outputDim].
    /// </summary>
    internal float[][][] RunOnnxBatchMulti(
        long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds,
        int startIdx, int batchSize, int seqLen)
    {
        var idsArray = new long[batchSize * seqLen];
        var maskArray = new long[batchSize * seqLen];
        var typeIdsArray = _metadata.TokenTypeIdsName != null ? new long[batchSize * seqLen] : null;

        for (int b = 0; b < batchSize; b++)
        {
            Array.Copy(tokenIds[startIdx + b], 0, idsArray, b * seqLen, seqLen);
            Array.Copy(attentionMasks[startIdx + b], 0, maskArray, b * seqLen, seqLen);
            if (typeIdsArray != null && tokenTypeIds != null)
                Array.Copy(tokenTypeIds[startIdx + b], 0, typeIdsArray, b * seqLen, seqLen);
        }

        var inputs = new Dictionary<string, OrtValue>
        {
            [_metadata.InputIdsName] = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]),
            [_metadata.AttentionMaskName] = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen])
        };

        if (_metadata.TokenTypeIdsName != null && typeIdsArray != null)
            inputs[_metadata.TokenTypeIdsName] = OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);

        var outputNames = new List<string> { _metadata.OutputTensorName };
        if (_metadata.AdditionalOutputNames != null)
            outputNames.AddRange(_metadata.AdditionalOutputNames);

        try
        {
            using var results = _session.Run(new RunOptions(), inputs, outputNames);

            int numOutputs = outputNames.Count;
            var allOutputs = new float[numOutputs][][];

            for (int outIdx = 0; outIdx < numOutputs; outIdx++)
            {
                var output = results[outIdx];
                var outputSpan = output.GetTensorDataAsSpan<float>();
                int perRowSize = outputSpan.Length / batchSize;

                var batchOutputs = new float[batchSize][];
                for (int b = 0; b < batchSize; b++)
                    batchOutputs[b] = outputSpan.Slice(b * perRowSize, perRowSize).ToArray();

                allOutputs[outIdx] = batchOutputs;
            }

            return allOutputs;
        }
        finally
        {
            foreach (var ortValue in inputs.Values)
                ortValue.Dispose();
        }
    }

    /// <summary>
    /// Runs ONNX inference in batches, returning all configured outputs.
    /// Returns [numOutputs][totalRows][outputDim].
    /// </summary>
    internal float[][][] ScoreMulti(long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds)
    {
        int totalRows = tokenIds.Length;
        int batchSize = _options.BatchSize;
        int seqLen = _options.MaxTokenLength;
        int numOutputs = 1 + (_metadata.AdditionalOutputNames?.Length ?? 0);

        var allOutputs = new List<float[]>[numOutputs];
        for (int o = 0; o < numOutputs; o++)
            allOutputs[o] = new List<float[]>(totalRows);

        for (int start = 0; start < totalRows; start += batchSize)
        {
            int count = Math.Min(batchSize, totalRows - start);
            var batchOutputs = RunOnnxBatchMulti(
                tokenIds, attentionMasks, tokenTypeIds,
                start, count, seqLen);

            for (int o = 0; o < numOutputs; o++)
                allOutputs[o].AddRange(batchOutputs[o]);
        }

        var result = new float[numOutputs][][];
        for (int o = 0; o < numOutputs; o++)
            result[o] = [.. allOutputs[o]];

        return result;
    }

    /// <summary>
    /// Runs multi-output ONNX inference on a pre-tokenized batch.
    /// </summary>
    internal float[][][] ScoreMulti(TokenizedBatch batch)
    {
        return ScoreMulti(batch.TokenIds, batch.AttentionMasks, batch.TokenTypeIds);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);

        int outputSize = _metadata.OutputRank == 2
            ? _metadata.HiddenDim
            : _options.MaxTokenLength * _metadata.HiddenDim;

        builder.AddColumn(_options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single, outputSize));

        if (_options.AdditionalOutputColumnNames != null && _metadata.AdditionalOutputDims != null)
        {
            for (int i = 0; i < _options.AdditionalOutputColumnNames.Length; i++)
            {
                builder.AddColumn(_options.AdditionalOutputColumnNames[i],
                    new VectorDataViewType(NumberDataViewType.Single, _metadata.AdditionalOutputDims[i]));
            }
        }

        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    public void Dispose() => _session.Dispose();
}

/// <summary>
/// Wrapping IDataView that adds ONNX model output to the upstream schema.
/// No inference happens here — it's all in the cursor.
/// </summary>
internal sealed class ScorerDataView : IDataView
{
    private readonly IDataView _input;
    private readonly OnnxTextModelScorerTransformer _scorer;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal ScorerDataView(IDataView input, OnnxTextModelScorerTransformer scorer)
    {
        _input = input;
        _scorer = scorer;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);

        int outputSize = scorer.Metadata.OutputRank == 2
            ? scorer.HiddenDim
            : scorer.Options.MaxTokenLength * scorer.HiddenDim;

        builder.AddColumn(scorer.Options.OutputColumnName,
            new VectorDataViewType(NumberDataViewType.Single, outputSize));

        if (scorer.Options.AdditionalOutputColumnNames != null && scorer.Metadata.AdditionalOutputDims != null)
        {
            for (int i = 0; i < scorer.Options.AdditionalOutputColumnNames.Length; i++)
            {
                builder.AddColumn(scorer.Options.AdditionalOutputColumnNames[i],
                    new VectorDataViewType(NumberDataViewType.Single, scorer.Metadata.AdditionalOutputDims[i]));
            }
        }

        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var options = _scorer.Options;
        var upstreamCols = new List<DataViewSchema.Column>();

        foreach (var col in columnsNeeded)
        {
            var inputCol = _input.Schema.GetColumnOrNull(col.Name);
            if (inputCol != null)
                upstreamCols.Add(inputCol.Value);
        }

        // Always need token columns for ONNX inference
        upstreamCols.Add(_input.Schema[options.TokenIdsColumnName]);
        upstreamCols.Add(_input.Schema[options.AttentionMaskColumnName]);
        if (options.TokenTypeIdsColumnName != null)
        {
            var typeIdCol = _input.Schema.GetColumnOrNull(options.TokenTypeIdsColumnName);
            if (typeIdCol != null)
                upstreamCols.Add(typeIdCol.Value);
        }

        var inputCursor = _input.GetRowCursor(upstreamCols.Distinct(), rand);
        return new ScorerCursor(this, inputCursor, _scorer);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor with lookahead batching for ONNX inference.
/// Reads ahead BatchSize rows, runs a single session.Run(), caches results,
/// then serves them one at a time.
/// </summary>
internal sealed class ScorerCursor : DataViewRowCursor
{
    private readonly ScorerDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly OnnxTextModelScorerTransformer _scorer;

    // Lookahead batch state
    private float[][]? _batchResults;
    private float[][][]? _batchAdditionalResults;
    private int _batchIndex = -1;
    private int _batchCount = 0;
    private long _position = -1;
    private bool _inputExhausted;

    // Cached upstream column values for the current batch (needed for passthrough)
    private readonly List<CachedRow> _batchRows = new();

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _position;
    public override long Batch => 0;

    internal ScorerCursor(
        ScorerDataView parent,
        DataViewRowCursor inputCursor,
        OnnxTextModelScorerTransformer scorer)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _scorer = scorer;
    }

    public override bool MoveNext()
    {
        _batchIndex++;

        if (_batchResults == null || _batchIndex >= _batchCount)
        {
            if (_inputExhausted)
                return false;

            if (!FillNextBatch())
                return false;
        }

        _position++;
        return true;
    }

    private bool FillNextBatch()
    {
        var options = _scorer.Options;
        int seqLen = options.MaxTokenLength;
        int batchSize = options.BatchSize;

        var tokenIdsBatch = new List<long[]>();
        var attMaskBatch = new List<long[]>();
        var typeIdsBatch = new List<long[]>();
        _batchRows.Clear();

        var tokenIdsGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[options.TokenIdsColumnName]);
        var attMaskGetter = _inputCursor.GetGetter<VBuffer<long>>(
            _inputCursor.Schema[options.AttentionMaskColumnName]);

        ValueGetter<VBuffer<long>>? typeIdsGetter = null;
        if (options.TokenTypeIdsColumnName != null)
        {
            var typeIdCol = _inputCursor.Schema.GetColumnOrNull(options.TokenTypeIdsColumnName);
            if (typeIdCol != null)
                typeIdsGetter = _inputCursor.GetGetter<VBuffer<long>>(typeIdCol.Value);
        }

        VBuffer<long> tokenIdsBuffer = default;
        VBuffer<long> attMaskBuffer = default;
        VBuffer<long> typeIdsBuffer = default;

        for (int i = 0; i < batchSize; i++)
        {
            if (!_inputCursor.MoveNext())
            {
                _inputExhausted = true;
                break;
            }

            tokenIdsGetter(ref tokenIdsBuffer);
            attMaskGetter(ref attMaskBuffer);
            tokenIdsBatch.Add(tokenIdsBuffer.DenseValues().ToArray());
            attMaskBatch.Add(attMaskBuffer.DenseValues().ToArray());

            if (typeIdsGetter != null)
            {
                typeIdsGetter(ref typeIdsBuffer);
                typeIdsBatch.Add(typeIdsBuffer.DenseValues().ToArray());
            }

            // Cache all upstream column values for this row
            _batchRows.Add(CacheCurrentRow());
        }

        if (tokenIdsBatch.Count == 0)
            return false;

        if (_scorer.Options.AdditionalOutputTensorNames != null)
        {
            var multiResults = _scorer.RunOnnxBatchMulti(
                tokenIdsBatch.ToArray(),
                attMaskBatch.ToArray(),
                typeIdsBatch.Count > 0 ? typeIdsBatch.ToArray() : null,
                startIdx: 0,
                batchSize: tokenIdsBatch.Count,
                seqLen: seqLen);

            _batchResults = multiResults[0];
            _batchAdditionalResults = new float[multiResults.Length - 1][][];
            for (int i = 1; i < multiResults.Length; i++)
                _batchAdditionalResults[i - 1] = multiResults[i];
        }
        else
        {
            _batchResults = _scorer.RunOnnxBatch(
                tokenIdsBatch.ToArray(),
                attMaskBatch.ToArray(),
                typeIdsBatch.Count > 0 ? typeIdsBatch.ToArray() : null,
                startIdx: 0,
                batchSize: tokenIdsBatch.Count,
                seqLen: seqLen);
            _batchAdditionalResults = null;
        }

        _batchIndex = 0;
        _batchCount = tokenIdsBatch.Count;
        return true;
    }

    /// <summary>
    /// Caches all column values from the upstream cursor for the current row.
    /// Needed because lookahead advances the upstream cursor past these rows.
    /// </summary>
    private CachedRow CacheCurrentRow()
    {
        var cached = new CachedRow();

        foreach (var col in _inputCursor.Schema)
        {
            if (col.IsHidden) continue;

            try
            {
                if (col.Type is TextDataViewType)
                {
                    var getter = _inputCursor.GetGetter<ReadOnlyMemory<char>>(col);
                    ReadOnlyMemory<char> val = default;
                    getter(ref val);
                    cached.Values[col.Name] = val.ToString();
                }
                else if (col.Type is VectorDataViewType vecType && vecType.ItemType == NumberDataViewType.Int64)
                {
                    var getter = _inputCursor.GetGetter<VBuffer<long>>(col);
                    VBuffer<long> val = default;
                    getter(ref val);
                    cached.Values[col.Name] = val.DenseValues().ToArray();
                }
                else if (col.Type is VectorDataViewType vecTypeF && vecTypeF.ItemType == NumberDataViewType.Single)
                {
                    var getter = _inputCursor.GetGetter<VBuffer<float>>(col);
                    VBuffer<float> val = default;
                    getter(ref val);
                    cached.Values[col.Name] = val.DenseValues().ToArray();
                }
            }
            catch
            {
                // Skip columns that can't be cached
            }
        }

        return cached;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        // For the raw output column, return the cached ONNX result
        if (column.Name == _scorer.Options.OutputColumnName)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            {
                var data = _batchResults![_batchIndex];
                var editor = VBufferEditor.Create(ref value, data.Length);
                data.AsSpan().CopyTo(editor.Values);
                value = editor.Commit();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // For additional output columns
        if (_scorer.Options.AdditionalOutputColumnNames != null)
        {
            int additionalIdx = Array.IndexOf(_scorer.Options.AdditionalOutputColumnNames, column.Name);
            if (additionalIdx >= 0)
            {
                ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                {
                    var data = _batchAdditionalResults![additionalIdx][_batchIndex];
                    var editor = VBufferEditor.Create(ref value, data.Length);
                    data.AsSpan().CopyTo(editor.Values);
                    value = editor.Commit();
                };
                return (ValueGetter<TValue>)(object)getter;
            }
        }

        // For passthrough columns, return cached upstream values
        return GetCachedUpstreamGetter<TValue>(column);
    }

    private ValueGetter<TValue> GetCachedUpstreamGetter<TValue>(DataViewSchema.Column column)
    {
        if (typeof(TValue) == typeof(ReadOnlyMemory<char>))
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
            {
                var row = _batchRows[_batchIndex];
                if (row.Values.TryGetValue(column.Name, out var cached) && cached is string s)
                    value = s.AsMemory();
                else
                    value = ReadOnlyMemory<char>.Empty;
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        if (typeof(TValue) == typeof(VBuffer<long>))
        {
            ValueGetter<VBuffer<long>> getter = (ref VBuffer<long> value) =>
            {
                var row = _batchRows[_batchIndex];
                if (row.Values.TryGetValue(column.Name, out var cached) && cached is long[] arr)
                {
                    var editor = VBufferEditor.Create(ref value, arr.Length);
                    arr.AsSpan().CopyTo(editor.Values);
                    value = editor.Commit();
                }
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        if (typeof(TValue) == typeof(VBuffer<float>))
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            {
                var row = _batchRows[_batchIndex];
                if (row.Values.TryGetValue(column.Name, out var cached) && cached is float[] arr)
                {
                    var editor = VBufferEditor.Create(ref value, arr.Length);
                    arr.AsSpan().CopyTo(editor.Values);
                    value = editor.Commit();
                }
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        throw new InvalidOperationException(
            $"Unsupported column type for passthrough caching: {column.Name} ({typeof(TValue).Name})");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
    {
        return (ref DataViewRowId value) =>
            value = new DataViewRowId((ulong)_position, 0);
    }

    public override bool IsColumnActive(DataViewSchema.Column column) => true;

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _inputCursor.Dispose();
        base.Dispose(disposing);
    }

    /// <summary>
    /// Holds cached column values for a single upstream row.
    /// </summary>
    private sealed class CachedRow
    {
        public Dictionary<string, object> Values { get; } = new();
    }
}

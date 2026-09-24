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

        var inputs = new Dictionary<string, OrtValue>();
        try
        {
            inputs[_metadata.InputIdsName] =
                OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]);
            inputs[_metadata.AttentionMaskName] =
                OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen]);
            if (_metadata.TokenTypeIdsName != null && typeIdsArray != null)
            {
                inputs[_metadata.TokenTypeIdsName] =
                    OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);
            }

            using var runOptions = new RunOptions();
            using var results = _session.Run(runOptions, inputs, [_metadata.OutputTensorName]);
            var output = results[0];
            var outputSpan = GetValidatedOutputSpan(
                output,
                _metadata.OutputTensorName,
                batchSize,
                seqLen,
                _metadata.OutputRank,
                _metadata.OutputRank == 2
                    ? _metadata.HiddenDim
                    : checked(seqLen * _metadata.HiddenDim),
                _metadata.HiddenDim);

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

        var inputs = new Dictionary<string, OrtValue>();
        var outputNames = new List<string> { _metadata.OutputTensorName };
        if (_metadata.AdditionalOutputNames != null)
            outputNames.AddRange(_metadata.AdditionalOutputNames);

        try
        {
            inputs[_metadata.InputIdsName] =
                OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]);
            inputs[_metadata.AttentionMaskName] =
                OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen]);
            if (_metadata.TokenTypeIdsName != null && typeIdsArray != null)
            {
                inputs[_metadata.TokenTypeIdsName] =
                    OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);
            }

            using var runOptions = new RunOptions();
            using var results = _session.Run(runOptions, inputs, outputNames);

            int numOutputs = outputNames.Count;
            var allOutputs = new float[numOutputs][][];

            for (int outIdx = 0; outIdx < numOutputs; outIdx++)
            {
                var output = results[outIdx];
                int outputRank;
                int expectedWidth;
                if (outIdx == 0)
                {
                    outputRank = _metadata.OutputRank;
                    expectedWidth = outputRank == 2
                        ? _metadata.HiddenDim
                        : checked(seqLen * _metadata.HiddenDim);
                }
                else
                {
                    outputRank = _metadata.AdditionalOutputRanks?[outIdx - 1]
                        ?? throw new InvalidOperationException(
                            "Additional output rank metadata is missing.");
                    expectedWidth = _metadata.AdditionalOutputDims?[outIdx - 1]
                        ?? throw new InvalidOperationException(
                            "Additional output dimension metadata is missing.");
                }

                var outputSpan = GetValidatedOutputSpan(
                    output,
                    outputNames[outIdx],
                    batchSize,
                    seqLen,
                    outputRank,
                    expectedWidth,
                    outputRank == 3
                        ? _metadata.HiddenDim
                        : expectedWidth);
                int perRowSize = checked(outputSpan.Length / batchSize);

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

    private static ReadOnlySpan<float> GetValidatedOutputSpan(
        OrtValue output,
        string outputName,
        int batchSize,
        int sequenceLength,
        int expectedRank,
        int expectedRowWidth,
        int expectedLastDim)
    {
        var dimensions = output.GetTensorTypeAndShape().Shape;
        if (dimensions.Length != expectedRank ||
            dimensions.Length < 2 ||
            dimensions[0] != batchSize ||
            (dimensions.Length == 2 && dimensions[1] != expectedRowWidth) ||
            (dimensions.Length == 3 &&
             (dimensions[1] != sequenceLength ||
              dimensions[2] != expectedLastDim)))
        {
            throw new InvalidDataException(
                $"The ONNX output '{outputName}' has shape " +
                $"[{string.Join(",", dimensions)}]; expected " +
                (expectedRank == 2
                    ? $"[{batchSize},{expectedRowWidth}]"
                    : $"[{batchSize},{sequenceLength},{expectedLastDim}]."));
        }

        var outputSpan = output.GetTensorDataAsSpan<float>();
        var expectedElements = checked(batchSize * expectedRowWidth);
        if (outputSpan.Length != expectedElements)
            throw new InvalidDataException(
                $"The ONNX output '{outputName}' contains {outputSpan.Length} values, " +
                $"but its validated shape requires {expectedElements}.");

        return outputSpan;
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
        => new OnnxTextModelScorerRowToRowMapper(inputSchema, this);

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _session.Dispose();
    }
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
    internal int InputColumnCount => _input.Schema.Count;
    internal int OutputColumnIndex { get; }
    internal IReadOnlyDictionary<int, int> AdditionalOutputIndices { get; }

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
        OutputColumnIndex = input.Schema.Count;

        var additionalIndices = new Dictionary<int, int>();
        if (scorer.Options.AdditionalOutputColumnNames != null && scorer.Metadata.AdditionalOutputDims != null)
        {
            for (int i = 0; i < scorer.Options.AdditionalOutputColumnNames.Length; i++)
            {
                builder.AddColumn(scorer.Options.AdditionalOutputColumnNames[i],
                    new VectorDataViewType(NumberDataViewType.Single, scorer.Metadata.AdditionalOutputDims[i]));
                additionalIndices[input.Schema.Count + 1 + i] = i;
            }
        }

        AdditionalOutputIndices = additionalIndices;
        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var options = _scorer.Options;
        var upstreamCols = new List<DataViewSchema.Column>();
        var requestedColumns = columnsNeeded.ToArray();
        bool inferenceRequired = requestedColumns.Any(column =>
            column.Index == OutputColumnIndex || AdditionalOutputIndices.ContainsKey(column.Index));

        foreach (var col in requestedColumns)
        {
            if (col.Index < _input.Schema.Count)
                upstreamCols.Add(_input.Schema[col.Index]);
        }

        if (inferenceRequired)
        {
            upstreamCols.Add(_input.Schema[options.TokenIdsColumnName]);
            upstreamCols.Add(_input.Schema[options.AttentionMaskColumnName]);
            if (options.TokenTypeIdsColumnName != null)
            {
                var typeIdCol = _input.Schema.GetColumnOrNull(options.TokenTypeIdsColumnName);
                if (typeIdCol != null)
                    upstreamCols.Add(typeIdCol.Value);
            }
        }

        var inputCursor = _input.GetRowCursor(upstreamCols.Distinct(), rand);
        return new ScorerCursor(
            this,
            inputCursor,
            _scorer,
            inferenceRequired,
            upstreamCols.Select(column => column.Index).ToHashSet(),
            requestedColumns.Select(column => column.Index).ToHashSet());
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
    private readonly bool _inferenceRequired;
    private readonly HashSet<int> _cachedColumnIndices;
    private readonly HashSet<int> _requestedColumnIndices;
    private readonly ValueGetter<DataViewRowId> _inputIdGetter;

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
    public override long Batch =>
        _batchIndex >= 0 && _batchIndex < _batchRows.Count
            ? _batchRows[_batchIndex].Batch
            : -1;

    internal ScorerCursor(
        ScorerDataView parent,
        DataViewRowCursor inputCursor,
        OnnxTextModelScorerTransformer scorer,
        bool inferenceRequired,
        HashSet<int> cachedColumnIndices,
        HashSet<int> requestedColumnIndices)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _scorer = scorer;
        _inferenceRequired = inferenceRequired;
        _cachedColumnIndices = cachedColumnIndices;
        _requestedColumnIndices = requestedColumnIndices;
        _inputIdGetter = inputCursor.GetIdGetter();
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

        ValueGetter<VBuffer<long>>? tokenIdsGetter = null;
        ValueGetter<VBuffer<long>>? attMaskGetter = null;
        ValueGetter<VBuffer<long>>? typeIdsGetter = null;
        if (_inferenceRequired)
        {
            tokenIdsGetter = _inputCursor.GetGetter<VBuffer<long>>(
                _inputCursor.Schema[options.TokenIdsColumnName]);
            attMaskGetter = _inputCursor.GetGetter<VBuffer<long>>(
                _inputCursor.Schema[options.AttentionMaskColumnName]);
            if (options.TokenTypeIdsColumnName != null)
            {
                var typeIdCol = _inputCursor.Schema.GetColumnOrNull(options.TokenTypeIdsColumnName);
                if (typeIdCol != null)
                    typeIdsGetter = _inputCursor.GetGetter<VBuffer<long>>(typeIdCol.Value);
            }
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

            DataViewRowId rowId = default;
            _inputIdGetter(ref rowId);
            if (_inferenceRequired)
            {
                tokenIdsGetter!(ref tokenIdsBuffer);
                attMaskGetter!(ref attMaskBuffer);
                tokenIdsBatch.Add(tokenIdsBuffer.DenseValues().ToArray());
                attMaskBatch.Add(attMaskBuffer.DenseValues().ToArray());

                if (typeIdsGetter != null)
                {
                    typeIdsGetter(ref typeIdsBuffer);
                    typeIdsBatch.Add(typeIdsBuffer.DenseValues().ToArray());
                }
            }

            // Cache all upstream column values for this row
            _batchRows.Add(CacheCurrentRow(rowId, _inputCursor.Batch));
        }

        if (_batchRows.Count == 0)
            return false;

        if (!_inferenceRequired)
        {
            _batchResults = Array.Empty<float[]>();
            _batchAdditionalResults = null;
        }
        else if (_scorer.Options.AdditionalOutputTensorNames != null)
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
        _batchCount = _batchRows.Count;
        return true;
    }

    /// <summary>
    /// Caches requested upstream values from the current row. Lookahead advances
    /// the upstream cursor past these rows, so the mapped row must own snapshots.
    /// </summary>
    private CachedRow CacheCurrentRow(DataViewRowId rowId, long batch)
    {
        var cached = new CachedRow(rowId, batch);

        foreach (var col in _inputCursor.Schema)
        {
            if (!_cachedColumnIndices.Contains(col.Index))
                continue;

            if (col.Type is VectorDataViewType vectorType)
            {
                CacheVector(col, vectorType.ItemType.RawType, cached);
            }
            else
            {
                CacheScalar(col, col.Type.RawType, cached);
            }
        }

        return cached;
    }

    private void CacheScalar(DataViewSchema.Column column, Type rawType, CachedRow cached)
    {
        var method = GetType().GetMethod(
            nameof(CacheScalarValue),
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        method.MakeGenericMethod(rawType).Invoke(this, [column, cached]);
    }

    private void CacheScalarValue<T>(DataViewSchema.Column column, CachedRow cached)
    {
        var getter = _inputCursor.GetGetter<T>(column);
        T value = default!;
        getter(ref value);
        cached.Values[column.Index] = new CachedScalar<T>(
            DecisionDataViewUtils.CopyValue(value));
    }

    private void CacheVector(DataViewSchema.Column column, Type rawType, CachedRow cached)
    {
        var method = GetType().GetMethod(
            nameof(CacheVectorValue),
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        method.MakeGenericMethod(rawType).Invoke(this, [column, cached]);
    }

    private void CacheVectorValue<T>(DataViewSchema.Column column, CachedRow cached)
    {
        var getter = _inputCursor.GetGetter<VBuffer<T>>(column);
        VBuffer<T> value = default;
        getter(ref value);
        cached.Values[column.Index] = new CachedVector<T>(
            DecisionDataViewUtils.CopyValue(value));
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (!_requestedColumnIndices.Contains(column.Index))
            throw new InvalidOperationException(
                $"Column '{column.Name}' was not requested for this cursor.");

        // For the raw output column, return the cached ONNX result
        if (column.Index == _parent.OutputColumnIndex)
        {
            EnsureGetterType<TValue>(column, typeof(VBuffer<float>));
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            {
                var results = _batchResults
                    ?? throw new InvalidOperationException(
                        "The cursor has not advanced to a row.");
                var data = results[_batchIndex];
                var editor = VBufferEditor.Create(ref value, data.Length);
                data.AsSpan().CopyTo(editor.Values);
                value = editor.Commit();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // For additional output columns
        if (_parent.AdditionalOutputIndices.TryGetValue(column.Index, out int additionalIdx))
        {
            EnsureGetterType<TValue>(column, typeof(VBuffer<float>));

            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            {
                var additionalResults = _batchAdditionalResults
                    ?? throw new InvalidOperationException(
                        "The cursor has not advanced to a row.");
                var data = additionalResults[additionalIdx][_batchIndex];
                var editor = VBufferEditor.Create(ref value, data.Length);
                data.AsSpan().CopyTo(editor.Values);
                value = editor.Commit();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // For passthrough columns, return cached upstream values
        return GetCachedUpstreamGetter<TValue>(column);
    }

    private ValueGetter<TValue> GetCachedUpstreamGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index >= _parent.InputColumnCount)
            throw new InvalidOperationException(
                $"Column '{column.Name}' is not a passthrough input column.");

        var expectedType = column.Type is VectorDataViewType columnVectorType
            ? typeof(VBuffer<>).MakeGenericType(columnVectorType.ItemType.RawType)
            : column.Type.RawType;
        EnsureGetterType<TValue>(column, expectedType);

        if (column.Type is VectorDataViewType vectorType)
        {
            var method = GetType().GetMethod(
                nameof(CreateVectorGetter),
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
            return (ValueGetter<TValue>)method.MakeGenericMethod(vectorType.ItemType.RawType)
                .Invoke(this, [column])!;
        }

        var scalarMethod = GetType().GetMethod(
            nameof(CreateScalarGetter),
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        return (ValueGetter<TValue>)scalarMethod.MakeGenericMethod(typeof(TValue))
            .Invoke(this, [column])!;
    }

    private ValueGetter<T> CreateScalarGetter<T>(DataViewSchema.Column column)
    {
        return (ref T value) =>
        {
            var row = _batchRows[_batchIndex];
            value = ((CachedScalar<T>)row.Values[column.Index]).Value;
        };
    }

    private ValueGetter<VBuffer<T>> CreateVectorGetter<T>(DataViewSchema.Column column)
    {
        return (ref VBuffer<T> value) =>
        {
            var row = _batchRows[_batchIndex];
            value = DecisionDataViewUtils.CopyValue(
                ((CachedVector<T>)row.Values[column.Index]).Value);
        };
    }

    private static void EnsureGetterType<TValue>(DataViewSchema.Column column, Type expectedType)
    {
        if (typeof(TValue) != expectedType)
            throw new InvalidOperationException(
                $"Column '{column.Name}' has type {column.Type}, " +
                $"but getter requested {typeof(TValue).Name}; expected {expectedType.Name}.");
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
    {
        return (ref DataViewRowId value) =>
        {
            if (_batchIndex < 0 || _batchIndex >= _batchRows.Count)
                throw new InvalidOperationException("The cursor has not advanced to a row.");
            value = _batchRows[_batchIndex].Id;
        };
    }

    public override bool IsColumnActive(DataViewSchema.Column column)
        => _requestedColumnIndices.Contains(column.Index);

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
        public DataViewRowId Id { get; }
        public long Batch { get; }
        public Dictionary<int, object> Values { get; } = new();

        public CachedRow(DataViewRowId id, long batch)
        {
            Id = id;
            Batch = batch;
        }
    }

    private sealed class CachedScalar<T>
    {
        public T Value { get; }

        public CachedScalar(T value) => Value = value;
    }

    private sealed class CachedVector<T>
    {
        public VBuffer<T> Value { get; }

        public CachedVector(VBuffer<T> value) => Value = value;
    }
}

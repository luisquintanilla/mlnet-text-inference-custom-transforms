using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

internal sealed class TextTokenizerRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly TextTokenizerTransformer _transformer;
    private readonly TextTokenizerOptions _options;
    private readonly DataViewSchema.Column _textColumn;
    private readonly DataViewSchema.Column? _secondTextColumn;

    internal TextTokenizerRowToRowMapper(
        DataViewSchema inputSchema,
        TextTokenizerTransformer transformer)
        : base(inputSchema, transformer.GetOutputSchema(inputSchema))
    {
        _transformer = transformer;
        _options = transformer.Options;
        _textColumn = inputSchema[_options.InputColumnName];
        _secondTextColumn = _options.SecondInputColumnName is null
            ? null
            : inputSchema[_options.SecondInputColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _textColumn;
        if (_secondTextColumn is { } second)
            yield return second;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(static () => new TokenizerState());
        return DecisionDataViewUtils.VectorGetter<TValue, long>(
            () => GetVector(row, column, state));
    }

    private long[] GetVector(
        DecisionMappedRow row,
        DataViewSchema.Column column,
        TokenizerState state)
    {
        var batch = EnsureBatch(row, state);
        if (column.Name == _options.TokenIdsColumnName)
            return batch.TokenIds[0];
        if (column.Name == _options.AttentionMaskColumnName)
            return batch.AttentionMasks[0];
        if (_options.OutputTokenTypeIds && column.Name == _options.TokenTypeIdsColumnName)
            return batch.TokenTypeIds![0];
        if (_options.OutputOffsets && column.Name == _options.TokenStartOffsetsColumnName)
            return batch.TokenStartOffsets![0];
        if (_options.OutputOffsets && column.Name == _options.TokenEndOffsetsColumnName)
            return batch.TokenEndOffsets![0];
        throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");
    }

    private TokenizedBatch EnsureBatch(
        DecisionMappedRow row,
        TokenizerState state)
    {
        if (state.Batch is not null &&
            state.Position == row.Position &&
            state.BatchId == row.Batch)
            return state.Batch;

        var firstGetter = row.Input.GetGetter<ReadOnlyMemory<char>>(_textColumn);
        ReadOnlyMemory<char> first = default;
        firstGetter(ref first);

        TokenizedBatch batch;
        if (_secondTextColumn is { } secondColumn)
        {
            var secondGetter = row.Input.GetGetter<ReadOnlyMemory<char>>(secondColumn);
            ReadOnlyMemory<char> second = default;
            secondGetter(ref second);
            batch = _transformer.Tokenize([first.ToString()], [second.ToString()]);
        }
        else
        {
            batch = _transformer.Tokenize([first.ToString()]);
        }

        state.Batch = batch;
        state.Position = row.Position;
        state.BatchId = row.Batch;
        return batch;
    }

    private sealed class TokenizerState
    {
        internal TokenizedBatch? Batch { get; set; }
        internal long Position { get; set; }
        internal long BatchId { get; set; }
    }
}

internal sealed class SoftmaxClassificationRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly SoftmaxClassificationTransformer _transformer;
    private readonly SoftmaxClassificationOptions _options;
    private readonly DataViewSchema.Column _inputColumn;

    internal SoftmaxClassificationRowToRowMapper(
        DataViewSchema inputSchema,
        SoftmaxClassificationTransformer transformer)
        : base(inputSchema, transformer.GetOutputSchema(inputSchema))
    {
        _transformer = transformer;
        _options = transformer.Options;
        _inputColumn = inputSchema[_options.InputColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _inputColumn;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(static () => new ClassificationState());
        if (column.Name == _options.ProbabilitiesColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, float>(
                () => EnsureResult(row, state).Probabilities);
        if (column.Name == _options.PredictedLabelColumnName)
            return DecisionDataViewUtils.TextGetter<TValue>(
                () => EnsureResult(row, state).PredictedLabel);
        throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");
    }

    private ClassificationResult EnsureResult(
        DecisionMappedRow row,
        ClassificationState state)
    {
        if (state.Result is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Result;

        var getter = row.Input.GetGetter<VBuffer<float>>(_inputColumn);
        VBuffer<float> value = default;
        getter(ref value);
        state.Result = _transformer.Classify([value.DenseValues().ToArray()])[0];
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.Result;
    }

    private sealed class ClassificationState
    {
        internal ClassificationResult? Result { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal sealed class OnnxTextModelScorerRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly OnnxTextModelScorerTransformer _transformer;
    private readonly OnnxTextModelScorerOptions _options;
    private readonly DataViewSchema.Column _inputIds;
    private readonly DataViewSchema.Column _attentionMask;
    private readonly DataViewSchema.Column? _tokenTypeIds;

    internal OnnxTextModelScorerRowToRowMapper(
        DataViewSchema inputSchema,
        OnnxTextModelScorerTransformer transformer)
        : base(inputSchema, transformer.GetOutputSchema(inputSchema))
    {
        _transformer = transformer;
        _options = transformer.Options;
        _inputIds = inputSchema[_options.TokenIdsColumnName];
        _attentionMask = inputSchema[_options.AttentionMaskColumnName];
        _tokenTypeIds = _options.TokenTypeIdsColumnName is null
            ? null
            : inputSchema.GetColumnOrNull(_options.TokenTypeIdsColumnName);
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _inputIds;
        yield return _attentionMask;
        if (_tokenTypeIds is { } typeIds)
            yield return typeIds;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(static () => new ScorerState());
        var outputIndex = column.Name == _options.OutputColumnName
            ? 0
            : Array.IndexOf(_options.AdditionalOutputColumnNames ?? [], column.Name) + 1;
        if (outputIndex <= 0 && column.Name != _options.OutputColumnName)
            throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");
        return DecisionDataViewUtils.VectorGetter<TValue, float>(
            () => EnsureOutputs(row, state)[outputIndex]);
    }

    private float[][] EnsureOutputs(
        DecisionMappedRow row,
        ScorerState state)
    {
        if (state.Outputs is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Outputs;

        var ids = ReadVector(row, _inputIds);
        var attention = ReadVector(row, _attentionMask);
        long[][]? typeIds = _tokenTypeIds is null
            ? null
            : [ReadVector(row, _tokenTypeIds.Value)];
        var outputs = _transformer.RunOnnxBatchMulti(
            [ids], [attention], typeIds, 0, 1, _options.MaxTokenLength);
        state.Outputs = outputs.Select(static output => output[0]).ToArray();
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.Outputs;
    }

    private static long[] ReadVector(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var getter = row.Input.GetGetter<VBuffer<long>>(column);
        VBuffer<long> value = default;
        getter(ref value);
        return value.DenseValues().ToArray();
    }

    private sealed class ScorerState
    {
        internal float[][]? Outputs { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal sealed class EmbeddingPoolingRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly EmbeddingPoolingTransformer _transformer;
    private readonly EmbeddingPoolingOptions _options;
    private readonly DataViewSchema.Column _rawOutput;
    private readonly DataViewSchema.Column? _attentionMask;

    internal EmbeddingPoolingRowToRowMapper(
        DataViewSchema inputSchema,
        EmbeddingPoolingTransformer transformer)
        : base(inputSchema, transformer.GetOutputSchema(inputSchema))
    {
        _transformer = transformer;
        _options = transformer.Options;
        _rawOutput = inputSchema[_options.InputColumnName];
        _attentionMask = inputSchema.GetColumnOrNull(_options.AttentionMaskColumnName);
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _rawOutput;
        if (!_options.IsPrePooled &&
            _options.Pooling != PoolingStrategy.ClsToken &&
            _attentionMask is { } mask)
            yield return mask;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        if (column.Name != _options.OutputColumnName)
            throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");

        var state = row.GetState(static () => new PoolingState());
        return DecisionDataViewUtils.VectorGetter<TValue, float>(
            () => EnsureEmbedding(row, state));
    }

    private float[] EnsureEmbedding(DecisionMappedRow row, PoolingState state)
    {
        if (state.Embedding is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Embedding;

        var rawGetter = row.Input.GetGetter<VBuffer<float>>(_rawOutput);
        VBuffer<float> raw = default;
        rawGetter(ref raw);
        long[]? mask = null;
        if (!_options.IsPrePooled &&
            _options.Pooling != PoolingStrategy.ClsToken)
        {
            if (_attentionMask is { } maskColumn)
            {
                var maskGetter = row.Input.GetGetter<VBuffer<long>>(maskColumn);
                VBuffer<long> maskBuffer = default;
                maskGetter(ref maskBuffer);
                mask = maskBuffer.DenseValues().ToArray();
            }
            else
            {
                mask = new long[_options.SequenceLength];
                Array.Fill(mask, 1L);
            }
        }
        else if (!_options.IsPrePooled)
        {
            mask = new long[_options.SequenceLength];
            Array.Fill(mask, 1L);
        }

        state.Embedding = _transformer.Pool(
            [raw.DenseValues().ToArray()],
            mask is null ? null : [mask])[0];
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.Embedding;
    }

    private sealed class PoolingState
    {
        internal float[]? Embedding { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal sealed class SigmoidScorerRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly SigmoidScorerTransformer _transformer;
    private readonly SigmoidScorerOptions _options;
    private readonly DataViewSchema.Column _inputColumn;

    internal SigmoidScorerRowToRowMapper(
        DataViewSchema inputSchema,
        SigmoidScorerTransformer transformer)
        : base(inputSchema, transformer.GetOutputSchema(inputSchema))
    {
        _transformer = transformer;
        _options = transformer.Options;
        _inputColumn = inputSchema[_options.InputColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _inputColumn;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        if (column.Name != _options.OutputColumnName)
            throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");

        var state = row.GetState(static () => new SigmoidState());
        return DecisionDataViewUtils.ScalarGetter<TValue, float>(
            () => EnsureScore(row, state));
    }

    private float EnsureScore(DecisionMappedRow row, SigmoidState state)
    {
        if (state.HasScore &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Score;

        var getter = row.Input.GetGetter<VBuffer<float>>(_inputColumn);
        VBuffer<float> raw = default;
        getter(ref raw);
        state.Score = _transformer.Score([raw.DenseValues().ToArray()])[0];
        state.Position = row.Position;
        state.Batch = row.Batch;
        state.HasScore = true;
        return state.Score;
    }

    private sealed class SigmoidState
    {
        internal float Score { get; set; }
        internal bool HasScore { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal sealed class NerDecodingRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly NerDecodingTransformer _transformer;
    private readonly NerDecodingOptions _options;
    private readonly DataViewSchema.Column _rawOutput;
    private readonly DataViewSchema.Column _attentionMask;
    private readonly DataViewSchema.Column _startOffsets;
    private readonly DataViewSchema.Column _endOffsets;
    private readonly DataViewSchema.Column _text;

    internal NerDecodingRowToRowMapper(
        DataViewSchema inputSchema,
        NerDecodingTransformer transformer)
        : base(inputSchema, transformer.GetOutputSchema(inputSchema))
    {
        _transformer = transformer;
        _options = transformer.Options;
        _rawOutput = inputSchema[_options.InputColumnName];
        _attentionMask = inputSchema[_options.AttentionMaskColumnName];
        _startOffsets = inputSchema[_options.TokenStartOffsetsColumnName];
        _endOffsets = inputSchema[_options.TokenEndOffsetsColumnName];
        _text = inputSchema[_options.TextColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _rawOutput;
        yield return _attentionMask;
        yield return _startOffsets;
        yield return _endOffsets;
        yield return _text;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        if (column.Name != _options.OutputColumnName)
            throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");

        var state = row.GetState(static () => new NerState());
        return DecisionDataViewUtils.TextGetter<TValue>(
            () => EnsureEntities(row, state));
    }

    private string EnsureEntities(DecisionMappedRow row, NerState state)
    {
        if (state.Entities is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Entities;

        var raw = ReadVector<float>(row, _rawOutput);
        var mask = ReadVector<long>(row, _attentionMask);
        var starts = ReadVector<long>(row, _startOffsets);
        var ends = ReadVector<long>(row, _endOffsets);
        var textGetter = row.Input.GetGetter<ReadOnlyMemory<char>>(_text);
        ReadOnlyMemory<char> text = default;
        textGetter(ref text);
        state.Entities = NerDecodingTransformer.SerializeEntities(
            _transformer.DecodeRow(
                raw, mask, starts, ends, text.ToString(), _options.NumLabels!.Value));
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.Entities;
    }

    private static T[] ReadVector<T>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var getter = row.Input.GetGetter<VBuffer<T>>(column);
        VBuffer<T> value = default;
        getter(ref value);
        return value.DenseValues().ToArray();
    }

    private sealed class NerState
    {
        internal string? Entities { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal sealed class QaSpanExtractionRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly QaSpanExtractionTransformer _transformer;
    private readonly QaSpanExtractionOptions _options;
    private readonly DataViewSchema.Column _startLogits;
    private readonly DataViewSchema.Column _endLogits;
    private readonly DataViewSchema.Column _attentionMask;
    private readonly DataViewSchema.Column _startOffsets;
    private readonly DataViewSchema.Column _endOffsets;
    private readonly DataViewSchema.Column _text;

    internal QaSpanExtractionRowToRowMapper(
        DataViewSchema inputSchema,
        QaSpanExtractionTransformer transformer)
        : base(inputSchema, transformer.GetOutputSchema(inputSchema))
    {
        _transformer = transformer;
        _options = transformer.Options;
        _startLogits = inputSchema[_options.StartLogitsColumnName];
        _endLogits = inputSchema[_options.EndLogitsColumnName];
        _attentionMask = inputSchema[_options.AttentionMaskColumnName];
        _startOffsets = inputSchema[_options.TokenStartOffsetsColumnName];
        _endOffsets = inputSchema[_options.TokenEndOffsetsColumnName];
        _text = inputSchema[_options.TextColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _startLogits;
        yield return _endLogits;
        yield return _attentionMask;
        yield return _startOffsets;
        yield return _endOffsets;
        yield return _text;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(static () => new QaState());
        if (column.Name == _options.OutputColumnName)
            return DecisionDataViewUtils.TextGetter<TValue>(
                () => EnsureAnswer(row, state).Answer);
        if (column.Name == _options.ScoreColumnName)
            return DecisionDataViewUtils.ScalarGetter<TValue, float>(
                () => EnsureAnswer(row, state).Score);
        throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");
    }

    private QaResult EnsureAnswer(DecisionMappedRow row, QaState state)
    {
        if (state.Result is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Result;

        var start = ReadVector<float>(row, _startLogits);
        var end = ReadVector<float>(row, _endLogits);
        var mask = ReadVector<long>(row, _attentionMask);
        var starts = ReadVector<long>(row, _startOffsets);
        var ends = ReadVector<long>(row, _endOffsets);
        var textGetter = row.Input.GetGetter<ReadOnlyMemory<char>>(_text);
        ReadOnlyMemory<char> text = default;
        textGetter(ref text);
        state.Result = QaSpanExtractionTransformer.ExtractSpans(
            start, end, mask, starts, ends, text.ToString(),
            _options.MaxAnswerLength, _options.TopK)[0];
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.Result;
    }

    private static T[] ReadVector<T>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var getter = row.Input.GetGetter<VBuffer<T>>(column);
        VBuffer<T> value = default;
        getter(ref value);
        return value.DenseValues().ToArray();
    }

    private sealed class QaState
    {
        internal QaResult? Result { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

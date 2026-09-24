using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.TypedDecisions;

namespace MLNet.TextInference.Onnx;

internal abstract class DecisionRowToRowMapperBase : IRowToRowMapper
{
    protected DecisionRowToRowMapperBase(DataViewSchema inputSchema, DataViewSchema outputSchema)
    {
        InputSchema = inputSchema ?? throw new ArgumentNullException(nameof(inputSchema));
        OutputSchema = outputSchema ?? throw new ArgumentNullException(nameof(outputSchema));
    }

    public DataViewSchema InputSchema { get; }
    public DataViewSchema OutputSchema { get; }

    public IEnumerable<DataViewSchema.Column> GetDependencies(
        IEnumerable<DataViewSchema.Column> columnsNeeded)
    {
        ArgumentNullException.ThrowIfNull(columnsNeeded);
        var dependencies = new Dictionary<int, DataViewSchema.Column>();
        foreach (var column in columnsNeeded)
        {
            ValidateOutputColumn(column);
            if (IsPassthrough(column))
            {
                dependencies.TryAdd(column.Index, InputSchema[column.Index]);
            }
            else
            {
                foreach (var dependency in GetProducedDependencies(column))
                    dependencies.TryAdd(dependency.Index, dependency);
            }
        }

        return dependencies.Values;
    }

    public DataViewRow GetRow(
        DataViewRow input,
        IEnumerable<DataViewSchema.Column> activeColumns)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(activeColumns);
        if (!ReferenceEquals(input.Schema, InputSchema))
            throw new ArgumentException(
                "The input row schema must be the exact schema used to create this mapper.",
                nameof(input));

        var active = activeColumns.ToArray();
        foreach (var column in active)
            ValidateOutputColumn(column);
        return new DecisionMappedRow(this, input, active);
    }

    internal bool IsPassthrough(DataViewSchema.Column column)
        => column.Index < InputSchema.Count &&
            OutputSchema[column.Index].Name == InputSchema[column.Index].Name;

    internal ValueGetter<TValue> GetGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        if (IsPassthrough(column))
            return row.Input.GetGetter<TValue>(InputSchema[column.Index]);
        return GetProducedGetter<TValue>(row, column);
    }

    internal abstract IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column);

    internal abstract ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column);

    private void ValidateOutputColumn(DataViewSchema.Column column)
    {
        if (column.Index < 0 ||
            column.Index >= OutputSchema.Count ||
            OutputSchema[column.Index].Name != column.Name)
        {
            throw new ArgumentException(
                $"Column '{column.Name}' does not belong to this mapper's output schema.",
                nameof(column));
        }
    }

    internal sealed class DecisionMappedRow : DataViewRow
    {
        private readonly DecisionRowToRowMapperBase _mapper;
        private readonly HashSet<int> _activeColumns;
        private object? _state;
        private bool _disposed;

        internal DecisionMappedRow(
            DecisionRowToRowMapperBase mapper,
            DataViewRow input,
            IEnumerable<DataViewSchema.Column> activeColumns)
        {
            _mapper = mapper;
            Input = input;
            _activeColumns = activeColumns.Select(static column => column.Index).ToHashSet();
        }

        internal DataViewRow Input { get; }

        internal TState GetState<TState>(Func<TState> factory)
            where TState : class
            => (TState)(_state ??= factory());

        public override DataViewSchema Schema => _mapper.OutputSchema;
        public override long Position => Input.Position;
        public override long Batch => Input.Batch;

        public override ValueGetter<DataViewRowId> GetIdGetter()
            => Input.GetIdGetter();

        public override bool IsColumnActive(DataViewSchema.Column column)
            => _activeColumns.Contains(column.Index);

        public override ValueGetter<TValue> GetGetter<TValue>(
            DataViewSchema.Column column)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!IsColumnActive(column))
                throw new InvalidOperationException(
                    $"Column '{column.Name}' was not requested by this row.");
            return _mapper.GetGetter<TValue>(this, column);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && !_disposed)
            {
                _disposed = true;
                Input.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}

internal sealed class TypedDecisionRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly DecisionInferenceEngine _engine;
    private readonly OnnxTypedDecisionsOptions _options;
    private readonly DataViewSchema.Column _stateColumn;

    internal TypedDecisionRowToRowMapper(
        DataViewSchema inputSchema,
        DecisionInferenceEngine engine,
        OnnxTypedDecisionsOptions options)
        : base(inputSchema, DecisionSchema.AddResults(inputSchema, options))
    {
        _engine = engine;
        _options = options;
        _stateColumn = inputSchema[options.StateColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _stateColumn;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(() => new FacadeState());
        if (column.Name == _options.ResultsColumnName)
            return DecisionDataViewUtils.TextGetter<TValue>(
                () => DecisionJsonCodec.SerializeResponse(EnsureResponse(row, state)));

        return DecisionQuestionOutputProjection.CreateGetter<TValue>(
            column,
            _options.Questions,
            _options.OutputNames,
            () => EnsureResponse(row, state));
    }

    private DecisionResponse EnsureResponse(
        DecisionMappedRow row,
        FacadeState state)
    {
        if (state.HasResponse &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Response!;

        var getter = row.Input.GetGetter<ReadOnlyMemory<char>>(_stateColumn);
        ReadOnlyMemory<char> value = default;
        getter(ref value);
        state.Response = _engine.Infer([value.ToString()], _options.Questions)[0];
        state.Position = row.Position;
        state.Batch = row.Batch;
        state.HasResponse = true;
        return state.Response;
    }

    private sealed class FacadeState
    {
        internal DecisionResponse? Response { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
        internal bool HasResponse { get; set; }
    }
}

internal sealed class DecisionPreparationRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly PrepareDecisionInputs _preparer;
    private readonly DecisionInputPreparationOptions _options;
    private readonly DataViewSchema.Column _stateColumn;

    internal DecisionPreparationRowToRowMapper(
        DataViewSchema inputSchema,
        PrepareDecisionInputs preparer,
        DecisionInputPreparationOptions options)
        : base(inputSchema, DecisionSchema.AddPreparation(inputSchema, options))
    {
        _preparer = preparer;
        _options = options;
        _stateColumn = inputSchema[options.StateColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _stateColumn;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(() => new PreparationState());
        if (column.Name == _options.InputIdsColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, long>(
                () => EnsureBatch(row, state).InputIds);
        if (column.Name == _options.AttentionMaskColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, long>(
                () => EnsureBatch(row, state).AttentionMask);
        if (column.Name == _options.MarkerPositionsColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, long>(
                () => EnsureBatch(row, state).MarkerPositions);
        if (column.Name == _options.MarkerMaskColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, bool>(
                () => EnsureBatch(row, state).MarkerMask);
        if (column.Name == _options.QuestionTypesColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, long>(
                () => EnsureBatch(row, state).QuestionTypes);
        if (column.Name == _options.BatchSizeColumnName)
            return DecisionDataViewUtils.ScalarGetter<TValue, int>(
                () => EnsureBatch(row, state).BatchSize);
        if (column.Name == _options.SequenceLengthColumnName)
            return DecisionDataViewUtils.ScalarGetter<TValue, int>(
                () => EnsureBatch(row, state).SequenceLength);
        if (column.Name == _options.MarkerWidthColumnName)
            return DecisionDataViewUtils.ScalarGetter<TValue, int>(
                () => EnsureBatch(row, state).MarkerWidth);
        throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");
    }

    private DecisionInputBatch EnsureBatch(
        DecisionMappedRow row,
        PreparationState state)
    {
        if (state.BatchValue is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.BatchValue;

        var getter = row.Input.GetGetter<ReadOnlyMemory<char>>(_stateColumn);
        ReadOnlyMemory<char> value = default;
        getter(ref value);
        state.BatchValue = _preparer.Prepare(value.ToString(), _options.Questions);
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.BatchValue;
    }

    private sealed class PreparationState
    {
        internal DecisionInputBatch? BatchValue { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal sealed class DecisionScoringRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly ScoreOnnxDecisionModel _scorer;
    private readonly OnnxDecisionModelScorerOptions _options;
    private readonly DataViewSchema.Column _inputIds;
    private readonly DataViewSchema.Column _attention;
    private readonly DataViewSchema.Column _markerPositions;
    private readonly DataViewSchema.Column _markerMask;
    private readonly DataViewSchema.Column _questionTypes;
    private readonly DataViewSchema.Column _batchSize;
    private readonly DataViewSchema.Column _sequenceLength;
    private readonly DataViewSchema.Column _markerWidth;

    internal DecisionScoringRowToRowMapper(
        DataViewSchema inputSchema,
        ScoreOnnxDecisionModel scorer,
        OnnxDecisionModelScorerOptions options)
        : base(inputSchema, DecisionSchema.AddScoring(inputSchema, options))
    {
        _scorer = scorer;
        _options = options;
        _inputIds = inputSchema[options.InputIdsColumnName];
        _attention = inputSchema[options.AttentionMaskColumnName];
        _markerPositions = inputSchema[options.MarkerPositionsColumnName];
        _markerMask = inputSchema[options.MarkerMaskColumnName];
        _questionTypes = inputSchema[options.QuestionTypesColumnName];
        _batchSize = inputSchema[options.BatchSizeColumnName];
        _sequenceLength = inputSchema[options.SequenceLengthColumnName];
        _markerWidth = inputSchema[options.MarkerWidthColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _inputIds;
        yield return _attention;
        yield return _markerPositions;
        yield return _markerMask;
        yield return _questionTypes;
        yield return _batchSize;
        yield return _sequenceLength;
        yield return _markerWidth;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(() => new ScoringState());
        if (column.Name == _options.LogitsColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, float>(
                () => EnsureOutputs(row, state).Logits);
        if (column.Name == _options.ActionProbabilitiesColumnName)
            return DecisionDataViewUtils.VectorGetter<TValue, float>(
                () => EnsureOutputs(row, state).ActionProbabilities);
        throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");
    }

    private DecisionModelOutputs EnsureOutputs(
        DecisionMappedRow row,
        ScoringState state)
    {
        if (state.Outputs is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Outputs;

        var inputIds = GetVector<long>(row, _inputIds);
        var attention = GetVector<long>(row, _attention);
        var markerPositions = GetVector<long>(row, _markerPositions);
        var markerMask = GetVector<bool>(row, _markerMask);
        var questionTypes = GetVector<long>(row, _questionTypes);
        var batchSize = GetScalar<int>(row, _batchSize);
        var sequenceLength = GetScalar<int>(row, _sequenceLength);
        var markerWidth = GetScalar<int>(row, _markerWidth);
        state.Outputs = _scorer.Score(
            inputIds,
            attention,
            markerPositions,
            markerMask,
            questionTypes,
            batchSize,
            sequenceLength,
            markerWidth);
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.Outputs;
    }

    private static T[] GetVector<T>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var getter = row.Input.GetGetter<VBuffer<T>>(column);
        VBuffer<T> value = default;
        getter(ref value);
        return value.DenseValues().ToArray();
    }

    private static T GetScalar<T>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var getter = row.Input.GetGetter<T>(column);
        T value = default!;
        getter(ref value);
        return value;
    }

    private sealed class ScoringState
    {
        internal DecisionModelOutputs? Outputs { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal sealed class DecisionDecodingRowToRowMapper : DecisionRowToRowMapperBase
{
    private readonly DecodeDecisions _decoder;
    private readonly DecisionDecodingOptions _options;
    private readonly DataViewSchema.Column _inputIds;
    private readonly DataViewSchema.Column _attention;
    private readonly DataViewSchema.Column _markerPositions;
    private readonly DataViewSchema.Column _markerMask;
    private readonly DataViewSchema.Column _questionTypes;
    private readonly DataViewSchema.Column _batchSize;
    private readonly DataViewSchema.Column _sequenceLength;
    private readonly DataViewSchema.Column _markerWidth;
    private readonly DataViewSchema.Column _logits;
    private readonly DataViewSchema.Column _actions;

    internal DecisionDecodingRowToRowMapper(
        DataViewSchema inputSchema,
        DecodeDecisions decoder,
        DecisionDecodingOptions options)
        : base(inputSchema, DecisionSchema.AddResults(inputSchema, options))
    {
        _decoder = decoder;
        _options = options;
        _inputIds = inputSchema[options.InputIdsColumnName];
        _attention = inputSchema[options.AttentionMaskColumnName];
        _markerPositions = inputSchema[options.MarkerPositionsColumnName];
        _markerMask = inputSchema[options.MarkerMaskColumnName];
        _questionTypes = inputSchema[options.QuestionTypesColumnName];
        _batchSize = inputSchema[options.BatchSizeColumnName];
        _sequenceLength = inputSchema[options.SequenceLengthColumnName];
        _markerWidth = inputSchema[options.MarkerWidthColumnName];
        _logits = inputSchema[options.LogitsColumnName];
        _actions = inputSchema[options.ActionProbabilitiesColumnName];
    }

    internal override IEnumerable<DataViewSchema.Column> GetProducedDependencies(
        DataViewSchema.Column column)
    {
        yield return _inputIds;
        yield return _attention;
        yield return _markerPositions;
        yield return _markerMask;
        yield return _questionTypes;
        yield return _batchSize;
        yield return _sequenceLength;
        yield return _markerWidth;
        yield return _logits;
        yield return _actions;
    }

    internal override ValueGetter<TValue> GetProducedGetter<TValue>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var state = row.GetState(() => new DecodingState());
        if (column.Name == _options.ResultsColumnName)
            return DecisionDataViewUtils.TextGetter<TValue>(
                () => DecisionJsonCodec.SerializeResponse(EnsureResponse(row, state)));

        return DecisionQuestionOutputProjection.CreateGetter<TValue>(
            column,
            _options.Questions,
            _options.OutputNames,
            () => EnsureResponse(row, state));
    }

    private DecisionResponse EnsureResponse(
        DecisionMappedRow row,
        DecodingState state)
    {
        if (state.Response is not null &&
            state.Position == row.Position &&
            state.Batch == row.Batch)
            return state.Response;

        var inputIds = GetVector<long>(row, _inputIds);
        var attention = GetVector<long>(row, _attention);
        var markerPositions = GetVector<long>(row, _markerPositions);
        var markerMask = GetVector<bool>(row, _markerMask);
        var questionTypes = GetVector<long>(row, _questionTypes);
        var batchSize = GetScalar<int>(row, _batchSize);
        var sequenceLength = GetScalar<int>(row, _sequenceLength);
        var markerWidth = GetScalar<int>(row, _markerWidth);
        var inputs = DecisionDataViewUtils.CreateInputBatch(
            inputIds,
            attention,
            markerPositions,
            markerMask,
            questionTypes,
            batchSize,
            sequenceLength,
            markerWidth,
            _options.Questions);
        var outputs = new DecisionModelOutputs
        {
            BatchSize = batchSize,
            MarkerWidth = markerWidth,
            Logits = GetVector<float>(row, _logits),
            ActionProbabilities = GetVector<float>(row, _actions)
        };
        state.Response = _decoder.Decode(inputs, outputs);
        state.Position = row.Position;
        state.Batch = row.Batch;
        return state.Response;
    }

    private static T[] GetVector<T>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var getter = row.Input.GetGetter<VBuffer<T>>(column);
        VBuffer<T> value = default;
        getter(ref value);
        return value.DenseValues().ToArray();
    }

    private static T GetScalar<T>(
        DecisionMappedRow row,
        DataViewSchema.Column column)
    {
        var getter = row.Input.GetGetter<T>(column);
        T value = default!;
        getter(ref value);
        return value;
    }

    private sealed class DecodingState
    {
        internal DecisionResponse? Response { get; set; }
        internal long Position { get; set; }
        internal long Batch { get; set; }
    }
}

internal static class DecisionQuestionOutputProjection
{
    internal static ValueGetter<TValue> CreateGetter<TValue>(
        DataViewSchema.Column column,
        IReadOnlyList<DecisionQuestion> questions,
        DecisionOutputNames outputNames,
        Func<DecisionResponse> responseFactory)
    {
        var question = questions
            .Select((item, index) => (item, index))
            .FirstOrDefault(item => IsQuestionColumn(
                outputNames.ById[item.item.Id],
                column.Name));
        if (question.item is null)
            throw new InvalidOperationException($"Unknown produced column '{column.Name}'.");

        var output = outputNames.ById[question.item.Id];
        if (column.Name == output.PredictedLabel)
        {
            if (question.item.Type == DecisionQuestionType.Noul)
                return DecisionDataViewUtils.BoolGetter<TValue>(
                    () => ((NoulDecisionResult)responseFactory().Results[question.index]).Value);
            return DecisionDataViewUtils.TextGetter<TValue>(
                () => ((ChoiceDecisionResult)responseFactory().Results[question.index]).Choice);
        }
        if (column.Name == output.Score)
            return DecisionDataViewUtils.FloatGetter<TValue>(
                () => ((ScoreDecisionResult)responseFactory().Results[question.index]).Score);
        if (column.Name == output.Probability)
            return DecisionDataViewUtils.FloatGetter<TValue>(
                () => ((NoulDecisionResult)responseFactory().Results[question.index]).ProbabilityTrue);
        if (column.Name == output.Probabilities)
            return DecisionDataViewUtils.VectorGetter<TValue, float>(
                () => responseFactory().Results[question.index].Distribution.Probabilities.ToArray());
        if (column.Name == output.Confidence)
            return DecisionDataViewUtils.FloatGetter<TValue>(
                () => responseFactory().Results[question.index].Confidence);
        return DecisionDataViewUtils.FloatGetter<TValue>(
            () => responseFactory().Results[question.index].ActionProbability);
    }

    private static bool IsQuestionColumn(
        QuestionOutputNames output,
        string name)
        => name == output.PredictedLabel ||
            name == output.Score ||
            name == output.Probability ||
            name == output.Probabilities ||
            name == output.Confidence ||
            name == output.ActionProbability;
}

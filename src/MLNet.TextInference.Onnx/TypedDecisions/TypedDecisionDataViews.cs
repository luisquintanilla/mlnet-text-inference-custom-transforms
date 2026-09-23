using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.TypedDecisions;
using static MLNet.TextInference.Onnx.DecisionDataViewUtils;

namespace MLNet.TextInference.Onnx;

internal abstract class DecisionDataViewBase : IDataView
{
    protected readonly IDataView Input;

    protected DecisionDataViewBase(IDataView input, DataViewSchema schema)
    {
        Input = input;
        Schema = schema;
    }

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => Input.GetRowCount();

    protected DataViewRowCursor OpenInputCursor(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        IEnumerable<string> required,
        Random? rand)
    {
        var names = columnsNeeded
            .Where(column => Input.Schema.GetColumnOrNull(column.Name) is not null)
            .Select(column => Input.Schema[column.Name])
            .Concat(required
                .Select(name => Input.Schema.GetColumnOrNull(name))
                .Where(static column => column is not null)
                .Select(static column => column!.Value))
            .Distinct();
        return Input.GetRowCursor(names, rand);
    }

    public abstract DataViewRowCursor GetRowCursor(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        Random? rand = null);

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        int n,
        Random? rand = null)
        => [GetRowCursor(columnsNeeded, rand)];
}

internal abstract class DecisionCursorBase : DataViewRowCursor
{
    private readonly Dictionary<string, CachedColumn> _cachedColumns = new(StringComparer.Ordinal);
    private readonly HashSet<string> _activeColumns;
    private readonly ValueGetter<DataViewRowId> _idGetter;

    protected DecisionCursorBase(
        DecisionDataViewBase parent,
        DataViewRowCursor inputCursor,
        IEnumerable<DataViewSchema.Column> columnsNeeded)
    {
        Parent = parent;
        InputCursor = inputCursor;
        _activeColumns = columnsNeeded
            .Select(static column => column.Name)
            .ToHashSet(StringComparer.Ordinal);
        _idGetter = inputCursor.GetIdGetter();
    }

    protected DecisionDataViewBase Parent { get; }
    protected DataViewRowCursor InputCursor { get; }
    protected List<Action> Readers { get; } = [];
    protected DataViewRowId CurrentId;
    protected bool IsRequested(string columnName) => _activeColumns.Contains(columnName);

    public override DataViewSchema Schema => Parent.Schema;
    public override long Position => InputCursor.Position;
    public override long Batch => InputCursor.Batch;

    protected void ReadCachedColumns()
    {
        foreach (var reader in Readers)
            reader();
        _idGetter(ref CurrentId);
    }

    protected ValueGetter<TValue> GetCachedUpstreamGetter<TValue>(
        DataViewSchema.Column column)
    {
        if (!_cachedColumns.TryGetValue(column.Name, out var cached))
        {
            var value = new CachedValue<TValue>();
            var getter = InputCursor.GetGetter<TValue>(
                InputCursor.Schema[column.Name]);
            cached = new CachedColumn(
                value,
                () =>
                {
                    TValue current = default!;
                    getter(ref current);
                    value.Value = DecisionDataViewUtils.CopyValue(current);
                });
            _cachedColumns.Add(column.Name, cached);
            Readers.Add(cached.Reader);
        }

        if (cached.Value is not CachedValue<TValue> typed)
            throw new InvalidOperationException(
                $"Column '{column.Name}' was requested with inconsistent value types.");
        ValueGetter<TValue> result = (ref TValue value) =>
            value = typed.Value;
        return result;
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => (ref DataViewRowId value) => value = CurrentId;

    public override bool IsColumnActive(DataViewSchema.Column column)
        => _activeColumns.Contains(column.Name);

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            InputCursor.Dispose();
        base.Dispose(disposing);
    }

    private sealed record CachedColumn(object Value, Action Reader);
    private sealed class CachedValue<TValue>
    {
        public TValue Value = default!;
    }
}

internal sealed class DecisionPreparationDataView : DecisionDataViewBase
{
    private readonly PrepareDecisionInputs _preparer;
    private readonly DecisionInputPreparationOptions _options;

    internal DecisionPreparationDataView(
        IDataView input,
        PrepareDecisionInputs preparer,
        DecisionInputPreparationOptions options)
        : base(input, DecisionSchema.AddPreparation(input.Schema, options))
    {
        _preparer = preparer;
        _options = options;
    }

    public override DataViewRowCursor GetRowCursor(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        Random? rand = null)
    {
        var requested = columnsNeeded.ToArray();
        return new Cursor(
            this,
            OpenInputCursor(requested, [_options.StateColumnName], rand),
            requested);
    }

    private sealed class Cursor : DecisionCursorBase
    {
        private readonly DecisionPreparationDataView _parent;
        private readonly ValueGetter<ReadOnlyMemory<char>> _stateGetter;
        private DecisionInputBatch? _batch;
        private readonly bool _needsPreparation;

        internal Cursor(
            DecisionPreparationDataView parent,
            DataViewRowCursor inputCursor,
            IEnumerable<DataViewSchema.Column> columnsNeeded)
            : base(parent, inputCursor, columnsNeeded)
        {
            _parent = parent;
            _stateGetter = inputCursor.GetGetter<ReadOnlyMemory<char>>(
                inputCursor.Schema[parent._options.StateColumnName]);
            _needsPreparation = columnsNeeded.Any(column =>
                parent._options.ColumnNames().Any(item => item.Name == column.Name));
        }

        public override bool MoveNext()
        {
            if (!InputCursor.MoveNext())
                return false;
            ReadOnlyMemory<char> state = default;
            _stateGetter(ref state);
            _batch = _needsPreparation
                ? _parent._preparer.Prepare(state.ToString(), _parent._options.Questions)
                : null;
            ReadCachedColumns();
            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Name == _parent._options.InputIdsColumnName)
                return VectorGetter<TValue, long>(() => _batch?.InputIds ?? []);
            if (column.Name == _parent._options.AttentionMaskColumnName)
                return VectorGetter<TValue, long>(() => _batch?.AttentionMask ?? []);
            if (column.Name == _parent._options.MarkerPositionsColumnName)
                return VectorGetter<TValue, long>(() => _batch?.MarkerPositions ?? []);
            if (column.Name == _parent._options.MarkerMaskColumnName)
                return VectorGetter<TValue, bool>(() => _batch?.MarkerMask ?? []);
            if (column.Name == _parent._options.QuestionTypesColumnName)
                return VectorGetter<TValue, long>(() => _batch?.QuestionTypes ?? []);
            if (column.Name == _parent._options.BatchSizeColumnName)
                return ScalarGetter<TValue, int>(() => _batch?.BatchSize ?? 0);
            if (column.Name == _parent._options.SequenceLengthColumnName)
                return ScalarGetter<TValue, int>(() => _batch?.SequenceLength ?? 0);
            if (column.Name == _parent._options.MarkerWidthColumnName)
                return ScalarGetter<TValue, int>(() => _batch?.MarkerWidth ?? 0);

            var upstream = InputCursor.Schema.GetColumnOrNull(column.Name);
            if (upstream is null)
                throw new InvalidOperationException($"Unknown column '{column.Name}'.");
            return GetCachedUpstreamGetter<TValue>(upstream.Value);
        }
    }
}

internal sealed class DecisionScoringDataView : DecisionDataViewBase
{
    private readonly ScoreOnnxDecisionModel _scorer;
    private readonly OnnxDecisionModelScorerOptions _options;

    internal DecisionScoringDataView(
        IDataView input,
        ScoreOnnxDecisionModel scorer,
        OnnxDecisionModelScorerOptions options)
        : base(input, DecisionSchema.AddScoring(input.Schema, options))
    {
        _scorer = scorer;
        _options = options;
    }

    public override DataViewRowCursor GetRowCursor(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        Random? rand = null)
    {
        var requested = columnsNeeded.ToArray();
        return new Cursor(
            this,
            OpenInputCursor(
                requested,
                [
                    _options.InputIdsColumnName,
                    _options.AttentionMaskColumnName,
                    _options.MarkerPositionsColumnName,
                    _options.MarkerMaskColumnName,
                    _options.QuestionTypesColumnName,
                    _options.BatchSizeColumnName,
                    _options.SequenceLengthColumnName,
                    _options.MarkerWidthColumnName
                ],
                rand),
            requested);
    }

    private sealed class Cursor : DecisionCursorBase
    {
        private readonly DecisionScoringDataView _parent;
        private readonly ValueGetter<VBuffer<long>> _inputIdsGetter;
        private readonly ValueGetter<VBuffer<long>> _attentionGetter;
        private readonly ValueGetter<VBuffer<long>> _markerPositionsGetter;
        private readonly ValueGetter<VBuffer<bool>> _markerMaskGetter;
        private readonly ValueGetter<VBuffer<long>> _questionTypesGetter;
        private readonly ValueGetter<int> _batchSizeGetter;
        private readonly ValueGetter<int> _sequenceLengthGetter;
        private readonly ValueGetter<int> _markerWidthGetter;
        private readonly Dictionary<string, object> _cachedValues = new(StringComparer.Ordinal);
        private readonly List<Action<int>> _readers = [];
        private readonly List<DataViewRowId> _ids = [];
        private readonly List<PreparedRow> _preparedRows = [];
        private ScoredRow[] _scoredRows = [];
        private readonly int _lookahead;
        private readonly bool _needsScoring;
        private int _index = -1;
        private long _position = -1;

        internal Cursor(
            DecisionScoringDataView parent,
            DataViewRowCursor inputCursor,
            IEnumerable<DataViewSchema.Column> columnsNeeded)
            : base(parent, inputCursor, columnsNeeded)
        {
            _parent = parent;
            _lookahead = parent._options.BatchSize;
            _needsScoring = columnsNeeded.Any(column =>
                column.Name == parent._options.LogitsColumnName ||
                column.Name == parent._options.ActionProbabilitiesColumnName);
            _inputIdsGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.InputIdsColumnName]);
            _attentionGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.AttentionMaskColumnName]);
            _markerPositionsGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.MarkerPositionsColumnName]);
            _markerMaskGetter = inputCursor.GetGetter<VBuffer<bool>>(
                inputCursor.Schema[parent._options.MarkerMaskColumnName]);
            _questionTypesGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.QuestionTypesColumnName]);
            _batchSizeGetter = inputCursor.GetGetter<int>(
                inputCursor.Schema[parent._options.BatchSizeColumnName]);
            _sequenceLengthGetter = inputCursor.GetGetter<int>(
                inputCursor.Schema[parent._options.SequenceLengthColumnName]);
            _markerWidthGetter = inputCursor.GetGetter<int>(
                inputCursor.Schema[parent._options.MarkerWidthColumnName]);
        }

        public override long Position => _position;

        public override bool MoveNext()
        {
            if (_index + 1 < _scoredRows.Length)
            {
                _index++;
                _position++;
                return true;
            }

            _preparedRows.Clear();
            _scoredRows = [];
            _index = -1;
            _ids.Clear();
            while (_preparedRows.Count < _lookahead && InputCursor.MoveNext())
            {
                VBuffer<long> inputIds = default;
                VBuffer<long> attention = default;
                VBuffer<long> markerPositions = default;
                VBuffer<bool> markerMask = default;
                VBuffer<long> questionTypes = default;
                var batchSize = 0;
                var sequenceLength = 0;
                var markerWidth = 0;
                _inputIdsGetter(ref inputIds);
                _attentionGetter(ref attention);
                _markerPositionsGetter(ref markerPositions);
                _markerMaskGetter(ref markerMask);
                _questionTypesGetter(ref questionTypes);
                _batchSizeGetter(ref batchSize);
                _sequenceLengthGetter(ref sequenceLength);
                _markerWidthGetter(ref markerWidth);
                _preparedRows.Add(new PreparedRow(
                    inputIds.DenseValues().ToArray(),
                    attention.DenseValues().ToArray(),
                    markerPositions.DenseValues().ToArray(),
                    markerMask.DenseValues().ToArray(),
                    questionTypes.DenseValues().ToArray(),
                    batchSize,
                    sequenceLength,
                    markerWidth));
                var id = default(DataViewRowId);
                InputCursor.GetIdGetter()(ref id);
                _ids.Add(id);
                foreach (var reader in _readers)
                    reader(_preparedRows.Count - 1);
            }

            if (_preparedRows.Count == 0)
                return false;

            _scoredRows = _needsScoring
                ? ScoreRows(_preparedRows)
                : new ScoredRow[_preparedRows.Count];
            _index = 0;
            _position++;
            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Name == _parent._options.LogitsColumnName)
                return VectorGetter<TValue, float>(() => _scoredRows[_index].Logits);
            if (column.Name == _parent._options.ActionProbabilitiesColumnName)
                return VectorGetter<TValue, float>(() => _scoredRows[_index].ActionProbabilities);

            var upstream = InputCursor.Schema.GetColumnOrNull(column.Name);
            if (upstream is null)
                throw new InvalidOperationException($"Unknown column '{column.Name}'.");
            return GetCachedBatchGetter<TValue>(upstream.Value);
        }

        public override ValueGetter<DataViewRowId> GetIdGetter()
            => (ref DataViewRowId value) => value = _ids[_index];

        private ScoredRow[] ScoreRows(IReadOnlyList<PreparedRow> rows)
        {
            var maxSequenceLength = rows.Max(static row => row.SequenceLength);
            var maxMarkerWidth = rows.Max(static row => row.MarkerWidth);
            var totalBatchSize = rows.Sum(static row => row.BatchSize);
            var inputIds = new long[totalBatchSize * maxSequenceLength];
            Array.Fill(inputIds, (long)_parent._scorer.PadTokenId);
            var attention = new long[inputIds.Length];
            var markerPositions = new long[totalBatchSize * maxMarkerWidth];
            var markerMask = new bool[markerPositions.Length];
            var questionTypes = new long[totalBatchSize];
            var rowOffsets = new int[rows.Count];
            var batchOffset = 0;
            for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
            {
                var row = rows[rowIndex];
                rowOffsets[rowIndex] = batchOffset;
                for (var batchRow = 0; batchRow < row.BatchSize; batchRow++)
                {
                    Array.Copy(
                        row.InputIds,
                        batchRow * row.SequenceLength,
                        inputIds,
                        (batchOffset + batchRow) * maxSequenceLength,
                        row.SequenceLength);
                    Array.Copy(
                        row.AttentionMask,
                        batchRow * row.SequenceLength,
                        attention,
                        (batchOffset + batchRow) * maxSequenceLength,
                        row.SequenceLength);
                    Array.Copy(
                        row.MarkerPositions,
                        batchRow * row.MarkerWidth,
                        markerPositions,
                        (batchOffset + batchRow) * maxMarkerWidth,
                        row.MarkerWidth);
                    Array.Copy(
                        row.MarkerMask,
                        batchRow * row.MarkerWidth,
                        markerMask,
                        (batchOffset + batchRow) * maxMarkerWidth,
                        row.MarkerWidth);
                    questionTypes[batchOffset + batchRow] = row.QuestionTypes[batchRow];
                }

                batchOffset += row.BatchSize;
            }

            var outputs = _parent._scorer.Score(
                inputIds,
                attention,
                markerPositions,
                markerMask,
                questionTypes,
                totalBatchSize,
                maxSequenceLength,
                maxMarkerWidth);
            var result = new ScoredRow[rows.Count];
            for (var rowIndex = 0; rowIndex < rows.Count; rowIndex++)
            {
                var row = rows[rowIndex];
                var outputOffset = rowOffsets[rowIndex] * maxMarkerWidth;
                var logits = new float[row.BatchSize * row.MarkerWidth];
                for (var batchRow = 0; batchRow < row.BatchSize; batchRow++)
                {
                    Array.Copy(
                        outputs.Logits,
                        outputOffset + batchRow * maxMarkerWidth,
                        logits,
                        batchRow * row.MarkerWidth,
                        row.MarkerWidth);
                }

                var actions = new float[row.BatchSize * 2];
                Array.Copy(
                    outputs.ActionProbabilities,
                    rowOffsets[rowIndex] * 2,
                    actions,
                    0,
                    actions.Length);
                result[rowIndex] = new ScoredRow(logits, actions);
            }

            return result;
        }

        private ValueGetter<TValue> GetCachedBatchGetter<TValue>(DataViewSchema.Column column)
        {
            if (!_cachedValues.TryGetValue(column.Name, out var cachedObject))
            {
                var values = new TValue[_lookahead];
                var getter = InputCursor.GetGetter<TValue>(column);
                _readers.Add(row =>
                {
                    TValue value = default!;
                    getter(ref value);
                    values[row] = CopyValue(value);
                });
                cachedObject = values;
                _cachedValues.Add(column.Name, cachedObject);
            }

            return (ref TValue value) => value = ((TValue[])cachedObject)[_index];
        }

        private sealed record PreparedRow(
            long[] InputIds,
            long[] AttentionMask,
            long[] MarkerPositions,
            bool[] MarkerMask,
            long[] QuestionTypes,
            int BatchSize,
            int SequenceLength,
            int MarkerWidth);

        private sealed record ScoredRow(float[] Logits, float[] ActionProbabilities);
    }
}

internal sealed class DecisionDecodingDataView : DecisionDataViewBase
{
    private readonly DecodeDecisions _decoder;
    private readonly DecisionDecodingOptions _options;

    internal DecisionDecodingDataView(
        IDataView input,
        DecodeDecisions decoder,
        DecisionDecodingOptions options)
        : base(input, DecisionSchema.AddResults(input.Schema, options))
    {
        _decoder = decoder;
        _options = options;
    }

    public override DataViewRowCursor GetRowCursor(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        Random? rand = null)
    {
        var requested = columnsNeeded.ToArray();
        return new Cursor(
            this,
            OpenInputCursor(
                requested,
                [
                    _options.InputIdsColumnName,
                    _options.AttentionMaskColumnName,
                    _options.MarkerPositionsColumnName,
                    _options.MarkerMaskColumnName,
                    _options.QuestionTypesColumnName,
                    _options.BatchSizeColumnName,
                    _options.SequenceLengthColumnName,
                    _options.MarkerWidthColumnName,
                    _options.LogitsColumnName,
                    _options.ActionProbabilitiesColumnName
                ],
                rand),
            requested);
    }

    private sealed class Cursor : DecisionCursorBase
    {
        private readonly DecisionDecodingDataView _parent;
        private readonly ValueGetter<VBuffer<long>> _inputIdsGetter;
        private readonly ValueGetter<VBuffer<long>> _attentionGetter;
        private readonly ValueGetter<VBuffer<long>> _markerPositionsGetter;
        private readonly ValueGetter<VBuffer<bool>> _markerMaskGetter;
        private readonly ValueGetter<VBuffer<long>> _questionTypesGetter;
        private readonly ValueGetter<int> _batchSizeGetter;
        private readonly ValueGetter<int> _sequenceLengthGetter;
        private readonly ValueGetter<int> _markerWidthGetter;
        private readonly ValueGetter<VBuffer<float>> _logitsGetter;
        private readonly ValueGetter<VBuffer<float>> _actionProbabilitiesGetter;
        private DecisionResponse? _response;
        private readonly bool _needsDecoding;

        internal Cursor(
            DecisionDecodingDataView parent,
            DataViewRowCursor inputCursor,
            IEnumerable<DataViewSchema.Column> columnsNeeded)
            : base(parent, inputCursor, columnsNeeded)
        {
            _parent = parent;
            _inputIdsGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.InputIdsColumnName]);
            _attentionGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.AttentionMaskColumnName]);
            _markerPositionsGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.MarkerPositionsColumnName]);
            _markerMaskGetter = inputCursor.GetGetter<VBuffer<bool>>(
                inputCursor.Schema[parent._options.MarkerMaskColumnName]);
            _questionTypesGetter = inputCursor.GetGetter<VBuffer<long>>(
                inputCursor.Schema[parent._options.QuestionTypesColumnName]);
            _batchSizeGetter = inputCursor.GetGetter<int>(
                inputCursor.Schema[parent._options.BatchSizeColumnName]);
            _sequenceLengthGetter = inputCursor.GetGetter<int>(
                inputCursor.Schema[parent._options.SequenceLengthColumnName]);
            _markerWidthGetter = inputCursor.GetGetter<int>(
                inputCursor.Schema[parent._options.MarkerWidthColumnName]);
            _logitsGetter = inputCursor.GetGetter<VBuffer<float>>(
                inputCursor.Schema[parent._options.LogitsColumnName]);
            _actionProbabilitiesGetter = inputCursor.GetGetter<VBuffer<float>>(
                inputCursor.Schema[parent._options.ActionProbabilitiesColumnName]);
            _needsDecoding = columnsNeeded.Any(column =>
                column.Name == parent._options.ResultsColumnName ||
                parent._options.Questions
                    .Select((question, index) => (question, index))
                    .Any(item => IsQuestionColumn(item.question, item.index, column.Name)));
        }

        public override bool MoveNext()
        {
            if (!InputCursor.MoveNext())
                return false;

            VBuffer<long> inputIds = default;
            VBuffer<long> attention = default;
            VBuffer<long> markerPositions = default;
            VBuffer<bool> markerMask = default;
            VBuffer<long> questionTypes = default;
            VBuffer<float> logits = default;
            VBuffer<float> actionProbabilities = default;
            var batchSize = 0;
            var sequenceLength = 0;
            var markerWidth = 0;
            _inputIdsGetter(ref inputIds);
            _attentionGetter(ref attention);
            _markerPositionsGetter(ref markerPositions);
            _markerMaskGetter(ref markerMask);
            _questionTypesGetter(ref questionTypes);
            _batchSizeGetter(ref batchSize);
            _sequenceLengthGetter(ref sequenceLength);
            _markerWidthGetter(ref markerWidth);
            _logitsGetter(ref logits);
            _actionProbabilitiesGetter(ref actionProbabilities);

            if (!_needsDecoding)
            {
                _response = null;
                ReadCachedColumns();
                return true;
            }

            var inputs = DecisionDataViewUtils.CreateInputBatch(
                inputIds.DenseValues().ToArray(),
                attention.DenseValues().ToArray(),
                markerPositions.DenseValues().ToArray(),
                markerMask.DenseValues().ToArray(),
                questionTypes.DenseValues().ToArray(),
                batchSize,
                sequenceLength,
                markerWidth,
                _parent._options.Questions);
            var outputs = new DecisionModelOutputs
            {
                BatchSize = batchSize,
                MarkerWidth = markerWidth,
                Logits = logits.DenseValues().ToArray(),
                ActionProbabilities = actionProbabilities.DenseValues().ToArray()
            };
            _response = _parent._decoder.Decode(inputs, outputs);
            ReadCachedColumns();
            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Name == _parent._options.ResultsColumnName)
                return TextGetter<TValue>(() => DecisionJsonCodec.SerializeResponse(_response!));

            var questionIndex = _parent._options.Questions
                .Select((question, index) => (question, index))
                .FirstOrDefault(item =>
                    IsQuestionColumn(item.question, item.index, column.Name));
            if (questionIndex.question is not null)
                return QuestionGetter<TValue>(questionIndex.index, column.Name, () => _response!);

            var upstream = InputCursor.Schema.GetColumnOrNull(column.Name);
            if (upstream is null)
                throw new InvalidOperationException($"Unknown column '{column.Name}'.");
            return GetCachedUpstreamGetter<TValue>(upstream.Value);
        }

        private bool IsQuestionColumn(DecisionQuestion question, int index, string name)
        {
            var output = _parent._options.OutputNames.ById[question.Id];
            return name == output.PredictedLabel || name == output.Score || name == output.Probability ||
                name == output.Probabilities ||
                name == output.Confidence || name == output.ActionProbability;
        }

        private ValueGetter<TValue> QuestionGetter<TValue>(
            int questionIndex,
            string name,
            Func<DecisionResponse> responseFactory)
        {
            var output = _parent._options.OutputNames.ById[
                _parent._options.Questions[questionIndex].Id];

            if (name == output.PredictedLabel)
            {
                if (_parent._options.Questions[questionIndex].Type == DecisionQuestionType.Noul)
                    return BoolGetter<TValue>(() => responseFactory().Results[questionIndex] is NoulDecisionResult noul &&
                        noul.Value);
                return TextGetter<TValue>(() => responseFactory().Results[questionIndex] is ChoiceDecisionResult choice
                    ? choice.Choice
                    : string.Empty);
            }
            if (name == output.Score)
                return FloatGetter<TValue>(() => responseFactory().Results[questionIndex] is ScoreDecisionResult score
                    ? score.Score
                    : float.NaN);
            if (name == output.Probability)
                return FloatGetter<TValue>(() => responseFactory().Results[questionIndex] is NoulDecisionResult noul
                    ? noul.ProbabilityTrue
                    : float.NaN);
            if (name == output.Probabilities)
                return VectorGetter<TValue, float>(() =>
                    responseFactory().Results[questionIndex].Distribution.Probabilities.ToArray());
            if (name == output.Confidence)
                return FloatGetter<TValue>(() => responseFactory().Results[questionIndex].Confidence);
            return FloatGetter<TValue>(() => responseFactory().Results[questionIndex].ActionProbability);
        }
    }
}

internal sealed class TypedDecisionDataView : DecisionDataViewBase
{
    private readonly DecisionInferenceEngine _engine;
    private readonly OnnxTypedDecisionsOptions _options;

    internal TypedDecisionDataView(
        IDataView input,
        DecisionInferenceEngine engine,
        OnnxTypedDecisionsOptions options)
        : base(input, DecisionSchema.AddResults(input.Schema, options))
    {
        _engine = engine;
        _options = options;
    }

    public override DataViewRowCursor GetRowCursor(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        Random? rand = null)
    {
        var requested = columnsNeeded.ToArray();
        return new Cursor(
            this,
            OpenInputCursor(requested, [_options.StateColumnName], rand),
            _options.BatchSize,
            requested);
    }

    private sealed class Cursor : DecisionCursorBase
    {
        private readonly TypedDecisionDataView _parent;
        private readonly int _batchSize;
        private readonly ValueGetter<ReadOnlyMemory<char>> _stateGetter;
        private readonly bool _needsInference;
        private readonly List<Action<int>> _readers = [];
        private readonly Dictionary<string, object> _cachedValues = new(StringComparer.Ordinal);
        private readonly List<DataViewRowId> _ids = [];
        private string[] _states;
        private DecisionResponse[] _responses = [];
        private int _count;
        private int _index = -1;
        private long _position = -1;

        internal Cursor(
            TypedDecisionDataView parent,
            DataViewRowCursor inputCursor,
            int batchSize,
            IEnumerable<DataViewSchema.Column> columnsNeeded)
            : base(parent, inputCursor, columnsNeeded)
        {
            _parent = parent;
            _batchSize = batchSize;
            _states = new string[batchSize];
            _stateGetter = inputCursor.GetGetter<ReadOnlyMemory<char>>(
                inputCursor.Schema[parent._options.StateColumnName]);
            _needsInference = columnsNeeded.Any(column =>
                column.Name == parent._options.ResultsColumnName ||
                parent._options.Questions
                    .Select((question, index) => (question, index))
                    .Any(item => IsQuestionColumn(item.question, column.Name)));
        }

        public override long Position => _position;

        public override bool MoveNext()
        {
            if (_index + 1 < _count)
            {
                _index++;
                _position++;
                return true;
            }

            _count = 0;
            _index = -1;
            _ids.Clear();
            while (_count < _batchSize && InputCursor.MoveNext())
            {
                ReadOnlyMemory<char> state = default;
                _stateGetter(ref state);
                _states[_count] = state.ToString();
                var id = default(DataViewRowId);
                InputCursor.GetIdGetter()(ref id);
                _ids.Add(id);
                foreach (var reader in _readers)
                    reader(_count);
                _count++;
            }

            if (_count == 0)
                return false;

            _responses = _needsInference
                ? _parent._engine.Infer(
                    _states.AsSpan(0, _count).ToArray(),
                    _parent._options.Questions,
                    _parent._options.BatchSize).ToArray()
                : [];
            _index = 0;
            _position++;
            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Name == _parent._options.ResultsColumnName)
                return TextGetter<TValue>(() => DecisionJsonCodec.SerializeResponse(_responses[_index]));

            var questionIndex = _parent._options.Questions
                .Select((question, index) => (question, index))
                .FirstOrDefault(item => IsQuestionColumn(item.question, column.Name));
            if (questionIndex.question is not null)
                return QuestionGetter<TValue>(questionIndex.index, column.Name);

            var upstream = InputCursor.Schema.GetColumnOrNull(column.Name);
            if (upstream is null)
                throw new InvalidOperationException($"Unknown column '{column.Name}'.");
            return GetCachedGetter<TValue>(upstream.Value);
        }

        public override ValueGetter<DataViewRowId> GetIdGetter()
            => (ref DataViewRowId value) => value = _ids[_index];

        private bool IsQuestionColumn(DecisionQuestion question, string name)
        {
            var output = _parent._options.OutputNames.ById[question.Id];
            return name == output.PredictedLabel || name == output.Score || name == output.Probability ||
                name == output.Probabilities ||
                name == output.Confidence || name == output.ActionProbability;
        }

        private ValueGetter<TValue> GetCachedGetter<TValue>(DataViewSchema.Column column)
        {
            if (!_cachedValues.TryGetValue(column.Name, out var cachedObject))
            {
                var values = new TValue[_batchSize];
                var getter = InputCursor.GetGetter<TValue>(column);
                _readers.Add(row =>
                {
                    TValue value = default!;
                    getter(ref value);
                    values[row] = DecisionDataViewUtils.CopyValue(value);
                });
                cachedObject = values;
                _cachedValues.Add(column.Name, cachedObject);
            }

            var typedValues = (TValue[])cachedObject;
            return (ref TValue value) => value = typedValues[_index];
        }

        private ValueGetter<TValue> QuestionGetter<TValue>(int questionIndex, string name)
        {
            var output = _parent._options.OutputNames.ById[
                _parent._options.Questions[questionIndex].Id];
            if (name == output.PredictedLabel)
            {
                if (_parent._options.Questions[questionIndex].Type == DecisionQuestionType.Noul)
                    return BoolGetter<TValue>(() => _responses[_index].Results[questionIndex] is NoulDecisionResult noul &&
                        noul.Value);
                return TextGetter<TValue>(() => _responses[_index].Results[questionIndex] is ChoiceDecisionResult choice
                    ? choice.Choice
                    : string.Empty);
            }
            if (name == output.Score)
                return FloatGetter<TValue>(() => _responses[_index].Results[questionIndex] is ScoreDecisionResult score
                    ? score.Score
                    : float.NaN);
            if (name == output.Probability)
                return FloatGetter<TValue>(() => _responses[_index].Results[questionIndex] is NoulDecisionResult noul
                    ? noul.ProbabilityTrue
                    : float.NaN);
            if (name == output.Probabilities)
                return VectorGetter<TValue, float>(() =>
                    _responses[_index].Results[questionIndex].Distribution.Probabilities.ToArray());
            if (name == output.Confidence)
                return FloatGetter<TValue>(() => _responses[_index].Results[questionIndex].Confidence);
            return FloatGetter<TValue>(() => _responses[_index].Results[questionIndex].ActionProbability);
        }
    }
}

internal static class DecisionDataViewUtils
{
    internal static T CopyValue<T>(T value)
    {
        if (value is ReadOnlyMemory<char> memory)
            return (T)(object)memory.ToString().AsMemory();
        if (value is VBuffer<T> vector)
            return (T)(object)CopyVBuffer(vector);
        return value;
    }

    private static VBuffer<T> CopyVBuffer<T>(VBuffer<T> value)
    {
        var destination = default(VBuffer<T>);
        var values = value.GetValues();
        var editor = value.IsDense
            ? VBufferEditor.Create(ref destination, value.Length)
            : VBufferEditor.Create(ref destination, value.Length, values.Length);
        for (var index = 0; index < values.Length; index++)
            editor.Values[index] = CopyElement(values[index]);
        if (!value.IsDense)
            value.GetIndices().CopyTo(editor.Indices);
        return editor.Commit();
    }

    private static T CopyElement<T>(T value)
        => value is ReadOnlyMemory<char> memory
            ? (T)(object)memory.ToString().AsMemory()
            : value;

    internal static void SetDense<T>(ref VBuffer<T> target, T[] values)
    {
        var editor = VBufferEditor.Create(ref target, values.Length);
        values.AsSpan().CopyTo(editor.Values);
        target = editor.Commit();
    }

    internal static DecisionInputBatch CreateInputBatch(
        long[] inputIds,
        long[] attentionMask,
        long[] markerPositions,
        bool[] markerMask,
        long[] questionTypes,
        int batchSize,
        int sequenceLength,
        int markerWidth,
        IReadOnlyList<DecisionQuestion> questions)
    {
        if (questions.Count != batchSize)
            throw new InvalidDataException(
                $"The prepared row has {batchSize} question rows, but {questions.Count} questions are configured.");
        var items = new DecisionInputItem[batchSize];
        for (var row = 0; row < batchSize; row++)
        {
            var positions = new List<int>();
            var labels = questions[row].OptionLabels();
            for (var option = 0; option < markerWidth && option < labels.Count; option++)
            {
                if (markerMask[row * markerWidth + option])
                    positions.Add((int)markerPositions[row * markerWidth + option]);
            }

            items[row] = new DecisionInputItem(
                0,
                questions[row],
                positions.ToArray(),
                labels.ToArray(),
                questionTypes[row]);
        }

        var batch = new DecisionInputBatch
        {
            BatchSize = batchSize,
            SequenceLength = sequenceLength,
            MarkerWidth = markerWidth,
            InputIds = inputIds,
            AttentionMask = attentionMask,
            MarkerPositions = markerPositions,
            MarkerMask = markerMask,
            QuestionTypes = questionTypes,
            Items = items
        };
        batch.Validate();
        return batch;
    }

    internal static ValueGetter<TValue> TextGetter<TValue>(Func<string> factory)
    {
        ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
            value = factory().AsMemory();
        return (ValueGetter<TValue>)(object)getter;
    }

    internal static ValueGetter<TValue> FloatGetter<TValue>(Func<float> factory)
    {
        ValueGetter<float> getter = (ref float value) => value = factory();
        return (ValueGetter<TValue>)(object)getter;
    }

    internal static ValueGetter<TValue> BoolGetter<TValue>(Func<bool> factory)
    {
        ValueGetter<bool> getter = (ref bool value) => value = factory();
        return (ValueGetter<TValue>)(object)getter;
    }

    internal static ValueGetter<TValue> ScalarGetter<TValue, TScalar>(Func<TScalar> factory)
    {
        ValueGetter<TScalar> getter = (ref TScalar value) => value = factory();
        return (ValueGetter<TValue>)(object)getter;
    }

    internal static ValueGetter<TValue> VectorGetter<TValue, TElement>(Func<TElement[]> factory)
    {
        ValueGetter<VBuffer<TElement>> getter = (ref VBuffer<TElement> value) =>
            SetDense(ref value, factory());
        return (ValueGetter<TValue>)(object)getter;
    }
}

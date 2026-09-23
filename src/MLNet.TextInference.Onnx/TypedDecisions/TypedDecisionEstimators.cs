using Microsoft.ML;
using Microsoft.ML.Data;
using CoreDecoder = MLNet.TextInference.TypedDecisions.DecodeDecisions;
using CoreFacade = MLNet.TextInference.TypedDecisions.OnnxTypedDecisions;
using CorePreparer = MLNet.TextInference.TypedDecisions.PrepareDecisionInputs;
using CoreScorer = MLNet.TextInference.TypedDecisions.ScoreOnnxDecisionModel;
using CoreBundle = MLNet.TextInference.TypedDecisions.TypedDecisionBundle;
using CoreCodec = MLNet.TextInference.TypedDecisions.DecisionJsonCodec;
using CoreInputBatch = MLNet.TextInference.TypedDecisions.DecisionInputBatch;
using CoreModelOutputs = MLNet.TextInference.TypedDecisions.DecisionModelOutputs;
using CoreRequest = MLNet.TextInference.TypedDecisions.DecisionRequest;
using CoreResponse = MLNet.TextInference.TypedDecisions.DecisionResponse;
using static MLNet.TextInference.Onnx.TypedDecisionValidation;

namespace MLNet.TextInference.Onnx;

public sealed class OnnxTypedDecisionsEstimator : IEstimator<OnnxTypedDecisionsTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTypedDecisionsOptions _options;

    public OnnxTypedDecisionsEstimator(MLContext mlContext, OnnxTypedDecisionsOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();
    }

    public OnnxTypedDecisionsTransformer Fit(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.StateColumnName);
        return new OnnxTypedDecisionsTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return DecisionSchema.AddResults(inputSchema, _options);
    }
}

public sealed class PrepareDecisionInputsEstimator : IEstimator<PrepareDecisionInputsTransformer>
{
    private readonly MLContext _mlContext;
    private readonly PrepareDecisionInputsOptions _options;

    public PrepareDecisionInputsEstimator(MLContext mlContext, PrepareDecisionInputsOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();
    }

    public PrepareDecisionInputsTransformer Fit(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.StateColumnName);
        return new PrepareDecisionInputsTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return DecisionSchema.AddText(inputSchema, _options.OutputColumnName);
    }
}

public sealed class ScoreOnnxDecisionModelEstimator : IEstimator<ScoreOnnxDecisionModelTransformer>
{
    private readonly MLContext _mlContext;
    private readonly ScoreOnnxDecisionModelOptions _options;

    public ScoreOnnxDecisionModelEstimator(MLContext mlContext, ScoreOnnxDecisionModelOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();
    }

    public ScoreOnnxDecisionModelTransformer Fit(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.InputColumnName);
        return new ScoreOnnxDecisionModelTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.InputColumnName);
        return DecisionSchema.AddText(inputSchema, _options.OutputColumnName);
    }
}

public sealed class DecodeDecisionsEstimator : IEstimator<DecodeDecisionsTransformer>
{
    private readonly MLContext _mlContext;
    private readonly DecodeDecisionsOptions _options;

    public DecodeDecisionsEstimator(MLContext mlContext, DecodeDecisionsOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();
    }

    public DecodeDecisionsTransformer Fit(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.InputColumnName);
        return new DecodeDecisionsTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.InputColumnName);
        return DecisionSchema.AddResults(inputSchema, _options);
    }
}

public sealed class OnnxTypedDecisionsTransformer : ITransformer, IDisposable
{
    private readonly OnnxTypedDecisionsOptions _options;
    private readonly CoreFacade _facade;

    internal OnnxTypedDecisionsTransformer(MLContext mlContext, OnnxTypedDecisionsOptions options)
    {
        _options = options;
        _facade = new CoreFacade(options.BundlePath);
    }

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.StateColumnName);
        return new TypedDecisionDataView(input, _facade, _options);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return DecisionSchema.AddResults(inputSchema, _options);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException("Typed decisions use a lazy cursor-batched IDataView.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Typed decision bundles are referenced by path and are not embedded in ML.NET models.");

    public void Dispose() => _facade.Dispose();
}

public sealed class PrepareDecisionInputsTransformer : ITransformer, IDisposable
{
    private readonly PrepareDecisionInputsOptions _options;
    private readonly CoreBundle _bundle;
    private readonly CorePreparer _preparer;

    internal PrepareDecisionInputsTransformer(MLContext mlContext, PrepareDecisionInputsOptions options)
    {
        _options = options;
        _bundle = CoreBundle.Open(options.BundlePath);
        _preparer = new CorePreparer(_bundle.Profile, _bundle.Tokenizer);
    }

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.StateColumnName);
        return new JsonProjectionDataView(
            input, _options.StateColumnName, _options.OutputColumnName,
            state => CoreCodec.SerializeInputs(_preparer.Prepare(
                [CoreRequest.Create(state, _options.Questions)])));
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return DecisionSchema.AddText(inputSchema, _options.OutputColumnName);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx) => throw new NotSupportedException();

    public void Dispose() => _bundle.Dispose();
}

public sealed class ScoreOnnxDecisionModelTransformer : ITransformer, IDisposable
{
    private readonly ScoreOnnxDecisionModelOptions _options;
    private readonly CoreBundle _bundle;
    private readonly CoreScorer _scorer;

    internal ScoreOnnxDecisionModelTransformer(MLContext mlContext, ScoreOnnxDecisionModelOptions options)
    {
        _options = options;
        var bundle = CoreBundle.Open(options.BundlePath);
        try
        {
            _bundle = bundle;
            _scorer = new CoreScorer(bundle);
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.InputColumnName);
        return new JsonProjectionDataView(
            input, _options.InputColumnName, _options.OutputColumnName,
            json =>
            {
                var inputs = CoreCodec.DeserializeInputs(json);
                return CoreCodec.SerializeScored(inputs, _scorer.Score(inputs));
            });
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.InputColumnName);
        return DecisionSchema.AddText(inputSchema, _options.OutputColumnName);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx) => throw new NotSupportedException();

    public void Dispose()
    {
        _scorer.Dispose();
        _bundle.Dispose();
    }
}

public sealed class DecodeDecisionsTransformer : ITransformer, IDisposable
{
    private readonly DecodeDecisionsOptions _options;
    private readonly CoreBundle _bundle;
    private readonly CoreDecoder _decoder;

    internal DecodeDecisionsTransformer(MLContext mlContext, DecodeDecisionsOptions options)
    {
        _options = options;
        _bundle = CoreBundle.Open(options.BundlePath);
        _decoder = new CoreDecoder(_bundle.Profile.TemperaturePolicy, _bundle.Manifest.Decoder);
    }

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.InputColumnName);
        return new DecodeDecisionDataView(input, _options, _decoder);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.InputColumnName);
        return DecisionSchema.AddResults(inputSchema, _options);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx) => throw new NotSupportedException();

    public void Dispose() => _bundle.Dispose();
}

internal static class DecisionSchema
{
    internal static DataViewSchema AddText(DataViewSchema input, string name)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        builder.AddColumn(name, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    internal static SchemaShape AddResults(SchemaShape input, OnnxTypedDecisionsOptions options)
    {
        return AddColumns(
            input,
            (options.ResultsColumnName, (DataViewType)TextDataViewType.Instance),
            (options.ChoiceColumnName, (DataViewType)TextDataViewType.Instance),
            (options.ScoreColumnName, (DataViewType)NumberDataViewType.Single),
            (options.ProbabilityTrueColumnName, (DataViewType)NumberDataViewType.Single),
            (options.ConfidenceColumnName, (DataViewType)NumberDataViewType.Single),
            (options.ActionProbabilityColumnName, (DataViewType)NumberDataViewType.Single));
    }

    internal static SchemaShape AddResults(SchemaShape input, DecodeDecisionsOptions options)
    {
        return AddColumns(
            input,
            (options.ResultsColumnName, (DataViewType)TextDataViewType.Instance),
            (options.ChoiceColumnName, (DataViewType)TextDataViewType.Instance),
            (options.ScoreColumnName, (DataViewType)NumberDataViewType.Single),
            (options.ProbabilityTrueColumnName, (DataViewType)NumberDataViewType.Single),
            (options.ConfidenceColumnName, (DataViewType)NumberDataViewType.Single),
            (options.ActionProbabilityColumnName, (DataViewType)NumberDataViewType.Single));
    }

    internal static DataViewSchema AddResults(DataViewSchema input, OnnxTypedDecisionsOptions options)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        AddResultColumns(builder, options.ResultsColumnName, options.ChoiceColumnName,
            options.ScoreColumnName, options.ProbabilityTrueColumnName,
            options.ConfidenceColumnName, options.ActionProbabilityColumnName);
        return builder.ToSchema();
    }

    internal static DataViewSchema AddResults(DataViewSchema input, DecodeDecisionsOptions options)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        AddResultColumns(builder, options.ResultsColumnName, options.ChoiceColumnName,
            options.ScoreColumnName, options.ProbabilityTrueColumnName,
            options.ConfidenceColumnName, options.ActionProbabilityColumnName);
        return builder.ToSchema();
    }

    private static void AddResultColumns(DataViewSchema.Builder builder, params string[] names)
    {
        builder.AddColumn(names[0], TextDataViewType.Instance);
        builder.AddColumn(names[1], TextDataViewType.Instance);
        for (var i = 2; i < names.Length; i++)
            builder.AddColumn(names[i], NumberDataViewType.Single);
    }

    internal static SchemaShape AddText(SchemaShape input, string name)
        => AddColumns(input, (name, (DataViewType)TextDataViewType.Instance));

    private static SchemaShape AddColumns(
        SchemaShape input,
        params (string Name, DataViewType Type)[] columns)
    {
        var result = input.ToDictionary(column => column.Name, StringComparer.Ordinal);
        var columnConstructor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];

        foreach (var (name, type) in columns)
        {
            result[name] = (SchemaShape.Column)columnConstructor.Invoke([
                name,
                SchemaShape.Column.VectorKind.Scalar,
                type,
                false,
                (SchemaShape?)null
            ]);
        }

        return new SchemaShape(result.Values);
    }
}

internal static class TypedDecisionValidation
{
    internal static void ValidateTextColumn(DataViewSchema schema, string name)
    {
        var column = schema.GetColumnOrNull(name)
            ?? throw new ArgumentException($"Input schema does not contain column '{name}'.");
        if (column.Type != TextDataViewType.Instance)
            throw new ArgumentException($"Column '{name}' must be of type Text.");
    }

    internal static void ValidateTextColumn(SchemaShape schema, string name)
    {
        var column = schema.FirstOrDefault(c => c.Name == name);
        if (column.Name == null)
            throw new ArgumentException($"Input schema does not contain column '{name}'.");
        if (column.ItemType != TextDataViewType.Instance ||
            column.Kind != SchemaShape.Column.VectorKind.Scalar)
            throw new ArgumentException($"Column '{name}' must be a scalar Text column.");
    }
}

internal sealed class JsonProjectionDataView : IDataView
{
    private readonly IDataView _input;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly Func<string, string> _project;

    public JsonProjectionDataView(IDataView input, string inputName, string outputName, Func<string, string> project)
    {
        _input = input;
        _inputName = inputName;
        _outputName = outputName;
        _project = project;
        Schema = DecisionSchema.AddText(input.Schema, outputName);
    }

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var upstream = columnsNeeded
            .Where(c => _input.Schema.GetColumnOrNull(c.Name) != null)
            .Select(c => _input.Schema[c.Name])
            .Append(_input.Schema[_inputName])
            .Distinct();
        return new JsonProjectionCursor(this, _input.GetRowCursor(upstream, rand));
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
        => [GetRowCursor(columnsNeeded, rand)];

    private sealed class JsonProjectionCursor : DataViewRowCursor
    {
        private readonly JsonProjectionDataView _parent;
        private readonly DataViewRowCursor _inputCursor;
        private readonly ValueGetter<ReadOnlyMemory<char>> _inputGetter;
        private readonly ValueGetter<DataViewRowId> _idGetter;
        private string _output = string.Empty;

        internal JsonProjectionCursor(JsonProjectionDataView parent, DataViewRowCursor inputCursor)
        {
            _parent = parent;
            _inputCursor = inputCursor;
            _inputGetter = inputCursor.GetGetter<ReadOnlyMemory<char>>(
                inputCursor.Schema[parent._inputName]);
            _idGetter = inputCursor.GetIdGetter();
        }

        public override DataViewSchema Schema => _parent.Schema;
        public override long Position => _inputCursor.Position;
        public override long Batch => _inputCursor.Batch;

        public override bool MoveNext()
        {
            if (!_inputCursor.MoveNext())
                return false;
            ReadOnlyMemory<char> input = default;
            _inputGetter(ref input);
            _output = _parent._project(input.ToString());
            return true;
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Name == _parent._outputName)
            {
                ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
                    value = _output.AsMemory();
                return (ValueGetter<TValue>)(object)getter;
            }

            var upstream = _inputCursor.Schema.GetColumnOrNull(column.Name);
            if (upstream != null)
                return _inputCursor.GetGetter<TValue>(upstream.Value);
            throw new InvalidOperationException($"Unknown column '{column.Name}'.");
        }

        public override ValueGetter<DataViewRowId> GetIdGetter() => _idGetter;
        public override bool IsColumnActive(DataViewSchema.Column column) => true;

        protected override void Dispose(bool disposing)
        {
            if (disposing)
                _inputCursor.Dispose();
            base.Dispose(disposing);
        }
    }
}

internal sealed class DecodeDecisionDataView : IDataView
    {
        private readonly IDataView _input;
        private readonly DecodeDecisionsOptions _options;
        private readonly CoreDecoder _decoder;

        internal DecodeDecisionDataView(
            IDataView input,
            DecodeDecisionsOptions options,
            CoreDecoder decoder)
        {
            _input = input;
            _options = options;
            _decoder = decoder;
            Schema = DecisionSchema.AddResults(input.Schema, options);
        }

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _input.GetRowCount();

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        {
            var upstream = columnsNeeded
                .Where(c => _input.Schema.GetColumnOrNull(c.Name) != null)
                .Select(c => _input.Schema[c.Name])
                .Append(_input.Schema[_options.InputColumnName])
                .Distinct();
            return new DecodeCursor(this, _input.GetRowCursor(upstream, rand));
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class DecodeCursor : DataViewRowCursor
        {
            private readonly DecodeDecisionDataView _parent;
            private readonly DataViewRowCursor _inputCursor;
            private readonly ValueGetter<ReadOnlyMemory<char>> _inputGetter;
            private readonly ValueGetter<DataViewRowId> _idGetter;
            private CoreResponse _response = null!;

            internal DecodeCursor(DecodeDecisionDataView parent, DataViewRowCursor inputCursor)
            {
                _parent = parent;
                _inputCursor = inputCursor;
                _inputGetter = inputCursor.GetGetter<ReadOnlyMemory<char>>(
                    inputCursor.Schema[parent._options.InputColumnName]);
                _idGetter = inputCursor.GetIdGetter();
            }

            public override DataViewSchema Schema => _parent.Schema;
            public override long Position => _inputCursor.Position;
            public override long Batch => _inputCursor.Batch;

            public override bool MoveNext()
            {
                if (!_inputCursor.MoveNext())
                    return false;
                ReadOnlyMemory<char> json = default;
                _inputGetter(ref json);
                var (inputs, outputs) = CoreCodec.DeserializeScored(json.ToString());
                var decoded = _parent._decoder.Decode(inputs, outputs);
                _response = new CoreResponse
                {
                    InputTokenCount = decoded.InputTokenCount,
                    Results = decoded.Results
                };
                return true;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (column.Name == _parent._options.ResultsColumnName)
                    return TextGetter<TValue>(() => CoreCodec.SerializeResponse(_response));
                if (column.Name == _parent._options.ChoiceColumnName)
                    return TextGetter<TValue>(() => _response.Results.OfType<MLNet.TextInference.TypedDecisions.ChoiceDecisionResult>().FirstOrDefault() is { } choice
                        ? choice.Choice : string.Empty);
                if (column.Name == _parent._options.ScoreColumnName)
                    return FloatGetter<TValue>(() => _response.Results.OfType<MLNet.TextInference.TypedDecisions.ScoreDecisionResult>().FirstOrDefault() is { } score
                        ? score.Score : float.NaN);
                if (column.Name == _parent._options.ProbabilityTrueColumnName)
                    return FloatGetter<TValue>(() => _response.Results.OfType<MLNet.TextInference.TypedDecisions.NoulDecisionResult>().FirstOrDefault() is { } noul
                        ? noul.ProbabilityTrue : float.NaN);
                if (column.Name == _parent._options.ConfidenceColumnName)
                    return FloatGetter<TValue>(() => _response.Results.FirstOrDefault()?.Confidence ?? float.NaN);
                if (column.Name == _parent._options.ActionProbabilityColumnName)
                    return FloatGetter<TValue>(() => _response.Results.FirstOrDefault()?.ActionProbability ?? float.NaN);

                var upstream = _inputCursor.Schema.GetColumnOrNull(column.Name);
                if (upstream != null)
                    return _inputCursor.GetGetter<TValue>(upstream.Value);
                throw new InvalidOperationException($"Unknown column '{column.Name}'.");
            }

            public override ValueGetter<DataViewRowId> GetIdGetter() => _idGetter;
            public override bool IsColumnActive(DataViewSchema.Column column) => true;

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                base.Dispose(disposing);
            }

        private static ValueGetter<TValue> TextGetter<TValue>(Func<string> valueFactory)
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
                value = valueFactory().AsMemory();
            return (ValueGetter<TValue>)(object)getter;
        }

        private static ValueGetter<TValue> FloatGetter<TValue>(Func<float> valueFactory)
        {
            ValueGetter<float> getter = (ref float value) => value = valueFactory();
            return (ValueGetter<TValue>)(object)getter;
        }
    }
}

internal sealed class TypedDecisionDataView : IDataView
    {
        private readonly IDataView _input;
        private readonly CoreFacade _facade;
        private readonly OnnxTypedDecisionsOptions _options;

        internal TypedDecisionDataView(
            IDataView input,
            CoreFacade facade,
            OnnxTypedDecisionsOptions options)
        {
            _input = input;
            _facade = facade;
            _options = options;
            Schema = DecisionSchema.AddResults(input.Schema, options);
        }

        public DataViewSchema Schema { get; }
        public bool CanShuffle => false;
        public long? GetRowCount() => _input.GetRowCount();

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        {
            var upstream = columnsNeeded
                .Where(c => _input.Schema.GetColumnOrNull(c.Name) != null)
                .Select(c => _input.Schema[c.Name])
                .Append(_input.Schema[_options.StateColumnName])
                .Distinct();
            return new TypedDecisionCursor(
                this, _input.GetRowCursor(upstream, rand), _options.BatchSize);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => [GetRowCursor(columnsNeeded, rand)];

        private sealed class TypedDecisionCursor : DataViewRowCursor
        {
            private readonly TypedDecisionDataView _parent;
            private readonly DataViewRowCursor _inputCursor;
            private readonly int _batchSize;
            private readonly ValueGetter<ReadOnlyMemory<char>> _stateGetter;
            private readonly ValueGetter<DataViewRowId> _inputIdGetter;
            private readonly List<Action<int>> _cacheReaders = [];
            private readonly Dictionary<string, object> _cachedColumns = new(StringComparer.Ordinal);
            private readonly DataViewRowId[] _cachedIds;
            private string[] _states;
            private CoreResponse[] _responses = [];
            private int _batchCount;
            private int _currentIndex = -1;
            private long _position = -1;

            internal TypedDecisionCursor(
                TypedDecisionDataView parent,
                DataViewRowCursor inputCursor,
                int batchSize)
            {
                _parent = parent;
                _inputCursor = inputCursor;
                _batchSize = batchSize;
                _states = new string[batchSize];
                _cachedIds = new DataViewRowId[batchSize];
                _stateGetter = inputCursor.GetGetter<ReadOnlyMemory<char>>(
                    inputCursor.Schema[parent._options.StateColumnName]);
                _inputIdGetter = inputCursor.GetIdGetter();
            }

            public override DataViewSchema Schema => _parent.Schema;
            public override long Position => _position;
            public override long Batch => _inputCursor.Batch;

            public override bool MoveNext()
            {
                if (_currentIndex + 1 < _batchCount)
                {
                    _currentIndex++;
                    _position++;
                    return true;
                }

                _batchCount = 0;
                _currentIndex = -1;
                while (_batchCount < _batchSize && _inputCursor.MoveNext())
                {
                    ReadOnlyMemory<char> state = default;
                    _stateGetter(ref state);
                    _states[_batchCount] = state.ToString();
                    var id = default(DataViewRowId);
                    _inputIdGetter(ref id);
                    _cachedIds[_batchCount] = id;
                    foreach (var reader in _cacheReaders)
                        reader(_batchCount);
                    _batchCount++;
                }

                if (_batchCount == 0)
                    return false;

                var requests = new CoreRequest[_batchCount];
                for (var i = 0; i < _batchCount; i++)
                    requests[i] = CoreRequest.Create(_states[i], _parent._options.Questions);
                _responses = _parent._facade.Infer(requests).ToArray();
                _currentIndex = 0;
                _position++;
                return true;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (column.Name == _parent._options.ResultsColumnName)
                    return TextGetter<TValue>(() => CoreCodec.SerializeResponse(_responses[_currentIndex]));
                if (column.Name == _parent._options.ChoiceColumnName)
                    return TextGetter<TValue>(() => FirstChoice(_responses[_currentIndex]));
                if (column.Name == _parent._options.ScoreColumnName)
                    return FloatGetter<TValue>(() => FirstScore(_responses[_currentIndex]));
                if (column.Name == _parent._options.ProbabilityTrueColumnName)
                    return FloatGetter<TValue>(() => FirstNoulProbability(_responses[_currentIndex]));
                if (column.Name == _parent._options.ConfidenceColumnName)
                    return FloatGetter<TValue>(() => FirstConfidence(_responses[_currentIndex]));
                if (column.Name == _parent._options.ActionProbabilityColumnName)
                    return FloatGetter<TValue>(() => FirstActionProbability(_responses[_currentIndex]));

                var upstream = _inputCursor.Schema.GetColumnOrNull(column.Name);
                if (upstream == null)
                    throw new InvalidOperationException($"Unknown column '{column.Name}'.");

                if (!_cachedColumns.TryGetValue(column.Name, out var cached))
                {
                    var values = new TValue[_batchSize];
                    var getter = _inputCursor.GetGetter<TValue>(upstream.Value);
                    _cacheReaders.Add(row =>
                    {
                        TValue value = default!;
                        getter(ref value);
                        values[row] = value;
                    });
                    cached = values;
                    _cachedColumns[column.Name] = cached;
                }

                var typedValues = (TValue[])cached;
                ValueGetter<TValue> result = (ref TValue value) => value = typedValues[_currentIndex];
                return result;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                ValueGetter<DataViewRowId> getter = (ref DataViewRowId value) =>
                    value = _cachedIds[_currentIndex];
                return getter;
            }

            public override bool IsColumnActive(DataViewSchema.Column column) => true;

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                base.Dispose(disposing);
            }

            private static string FirstChoice(CoreResponse response) =>
                response.Results.OfType<MLNet.TextInference.TypedDecisions.ChoiceDecisionResult>().FirstOrDefault() is { } choice
                    ? choice.Choice : string.Empty;

            private static float FirstScore(CoreResponse response) =>
                response.Results.OfType<MLNet.TextInference.TypedDecisions.ScoreDecisionResult>().FirstOrDefault() is { } score
                    ? score.Score : float.NaN;

            private static float FirstNoulProbability(CoreResponse response) =>
                response.Results.OfType<MLNet.TextInference.TypedDecisions.NoulDecisionResult>().FirstOrDefault() is { } noul
                    ? noul.ProbabilityTrue : float.NaN;

            private static float FirstConfidence(CoreResponse response) =>
                response.Results.FirstOrDefault()?.Confidence ?? float.NaN;

            private static float FirstActionProbability(CoreResponse response) =>
                response.Results.FirstOrDefault()?.ActionProbability ?? float.NaN;
        }

        private static ValueGetter<TValue> TextGetter<TValue>(Func<string> valueFactory)
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
                value = valueFactory().AsMemory();
            return (ValueGetter<TValue>)(object)getter;
        }

        private static ValueGetter<TValue> FloatGetter<TValue>(Func<float> valueFactory)
        {
            ValueGetter<float> getter = (ref float value) => value = valueFactory();
            return (ValueGetter<TValue>)(object)getter;
        }
}

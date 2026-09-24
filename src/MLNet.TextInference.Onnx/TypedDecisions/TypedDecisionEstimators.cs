using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.TextInference.TypedDecisions;
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

public sealed class DecisionInputPreparationEstimator : IEstimator<DecisionInputPreparationTransformer>
{
    private readonly MLContext _mlContext;
    private readonly DecisionInputPreparationOptions _options;

    public DecisionInputPreparationEstimator(MLContext mlContext, DecisionInputPreparationOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();
    }

    public DecisionInputPreparationTransformer Fit(IDataView input)
    {
        ValidateTextColumn(input.Schema, _options.StateColumnName);
        return new DecisionInputPreparationTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return DecisionSchema.AddPreparation(inputSchema, _options);
    }
}

public sealed class OnnxDecisionModelScorerEstimator : IEstimator<OnnxDecisionModelScorerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxDecisionModelScorerOptions _options;

    public OnnxDecisionModelScorerEstimator(MLContext mlContext, OnnxDecisionModelScorerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();
    }

    public OnnxDecisionModelScorerTransformer Fit(IDataView input)
    {
        DecisionSchema.ValidatePreparationColumns(input.Schema, _options);
        return new OnnxDecisionModelScorerTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        DecisionSchema.ValidatePreparationColumns(inputSchema, _options);
        return DecisionSchema.AddScoring(inputSchema, _options);
    }
}

public sealed class DecisionDecodingEstimator : IEstimator<DecisionDecodingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly DecisionDecodingOptions _options;

    public DecisionDecodingEstimator(MLContext mlContext, DecisionDecodingOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();
    }

    public DecisionDecodingTransformer Fit(IDataView input)
    {
        DecisionSchema.ValidateScoringColumns(input.Schema, _options);
        return new DecisionDecodingTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        DecisionSchema.ValidateScoringColumns(inputSchema, _options);
        return DecisionSchema.AddResults(inputSchema, _options);
    }
}

public sealed class OnnxTypedDecisionsTransformer : ITransformer, IDisposable
{
    private readonly OnnxTypedDecisionsOptions _options;
    private readonly DecisionInferenceEngine _engine;
    private bool _disposed;

    internal OnnxTypedDecisionsTransformer(MLContext mlContext, OnnxTypedDecisionsOptions options)
    {
        _options = options;
        _engine = new DecisionInferenceEngine(mlContext, options.ModelAssetsPath);
    }

    internal OnnxTypedDecisionsOptions Options => _options;
    internal string AssetsRootPath => _engine.AssetsRootPath;

    /// <summary>Saves this transformer as a portable typed-decision artifact.</summary>
    public void Save(string path) => TypedDecisionPortableModel.Save(this, path);

    /// <summary>Loads a transformer from a portable typed-decision artifact.</summary>
    public static OnnxTypedDecisionsTransformer Load(MLContext mlContext, string path)
        => TypedDecisionPortableModel.LoadFacade(mlContext, path);

    /// <summary>
    /// Direct convenience inference using the same prepared/scored/decoded kernel as the IDataView path.
    /// </summary>
    public DecisionResponse Infer(string state)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _engine.Infer([state], _options.Questions)[0];
    }

    /// <summary>Runs configured questions over multiple states in one flattened ONNX batch.</summary>
    public IReadOnlyList<DecisionResponse> Infer(IReadOnlyList<string> states)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(states);
        return _engine.Infer(states, _options.Questions, _options.BatchSize);
    }

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateTextColumn(input.Schema, _options.StateColumnName);
        return new TypedDecisionDataView(input, _engine, _options);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return DecisionSchema.AddResults(inputSchema, _options);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return new TypedDecisionRowToRowMapper(inputSchema, _engine, _options);
    }

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "Native MLContext.Model.Save/Load is not supported for typed decisions; " +
            "use the portable typed-decision Save/Load API.");

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _engine.Dispose();
    }

    internal void AttachOwnedAssetDirectory(string rootPath)
        => _engine.AttachOwnedAssetDirectory(rootPath);

    internal void AttachOwnedAssetDirectory(
        string rootPath,
        TypedDecisionRootLease lease)
        => _engine.AttachOwnedAssetDirectory(rootPath, lease);
}

public sealed class DecisionInputPreparationTransformer : ITransformer, IDisposable
{
    private readonly DecisionInputPreparationOptions _options;
    private readonly TypedDecisionBundle _bundle;
    private readonly PrepareDecisionInputs _preparer;
    private bool _disposed;

    internal DecisionInputPreparationTransformer(MLContext mlContext, DecisionInputPreparationOptions options)
    {
        _options = options;
        _bundle = TypedDecisionBundle.Open(
            options.ModelAssetsPath,
            TypedDecisionBundleLoadRequirements.Model |
            TypedDecisionBundleLoadRequirements.Profile |
            TypedDecisionBundleLoadRequirements.Tokenizer);
        _preparer = new PrepareDecisionInputs(
            _bundle.Profile,
            _bundle.Tokenizer.Tokenizer,
            _bundle.Tokenizer.Metadata);
    }

    internal DecisionInputPreparationOptions Options => _options;
    internal string AssetsRootPath
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _bundle.RootPath;
        }
    }

    /// <summary>Saves this transformer as a portable typed-decision artifact.</summary>
    public void Save(string path) => TypedDecisionPortableModel.Save(this, path);

    /// <summary>Loads a transformer from a portable typed-decision artifact.</summary>
    public static DecisionInputPreparationTransformer Load(
        MLContext mlContext,
        string path)
        => TypedDecisionPortableModel.LoadPreparation(mlContext, path);

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateTextColumn(input.Schema, _options.StateColumnName);
        return new DecisionPreparationDataView(input, _preparer, _options);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return DecisionSchema.AddPreparation(inputSchema, _options);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateTextColumn(inputSchema, _options.StateColumnName);
        return new DecisionPreparationRowToRowMapper(inputSchema, _preparer, _options);
    }

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "Native MLContext.Model.Save/Load is not supported for typed decisions; " +
            "use the portable typed-decision Save/Load API.");

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _bundle.Dispose();
    }

    internal void AttachOwnedAssetDirectory(string rootPath)
        => _bundle.AttachOwnedRoot(rootPath);

    internal void AttachOwnedAssetDirectory(
        string rootPath,
        TypedDecisionRootLease lease)
        => _bundle.AttachOwnedRoot(rootPath, lease);
}

public sealed class OnnxDecisionModelScorerTransformer : ITransformer, IDisposable
{
    private readonly OnnxDecisionModelScorerOptions _options;
    private readonly TypedDecisionBundle _bundle;
    private readonly ScoreOnnxDecisionModel _scorer;
    private bool _disposed;

    internal OnnxDecisionModelScorerTransformer(MLContext mlContext, OnnxDecisionModelScorerOptions options)
    {
        _options = options;
        _bundle = TypedDecisionBundle.Open(
            options.ModelAssetsPath,
            TypedDecisionBundleLoadRequirements.Model |
            TypedDecisionBundleLoadRequirements.Tokenizer);
        try
        {
            _scorer = new ScoreOnnxDecisionModel(
                _bundle,
                new OnnxExecutionOptions(
                    mlContext.GpuDeviceId,
                    mlContext.FallbackToCpu,
                    static message =>
                        Console.Error.WriteLine($"[MLNet.TextInference.Onnx] {message}")));
        }
        catch
        {
            _bundle.Dispose();
            throw;
        }
    }

    internal OnnxDecisionModelScorerOptions Options => _options;
    internal string AssetsRootPath
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _bundle.RootPath;
        }
    }

    /// <summary>Saves this transformer as a portable typed-decision artifact.</summary>
    public void Save(string path) => TypedDecisionPortableModel.Save(this, path);

    /// <summary>Loads a transformer from a portable typed-decision artifact.</summary>
    public static OnnxDecisionModelScorerTransformer Load(
        MLContext mlContext,
        string path)
        => TypedDecisionPortableModel.LoadScoring(mlContext, path);

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        DecisionSchema.ValidatePreparationColumns(input.Schema, _options);
        return new DecisionScoringDataView(input, _scorer, _options);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        DecisionSchema.ValidatePreparationColumns(inputSchema, _options);
        return DecisionSchema.AddScoring(inputSchema, _options);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        DecisionSchema.ValidatePreparationColumns(inputSchema, _options);
        return new DecisionScoringRowToRowMapper(inputSchema, _scorer, _options);
    }

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "Native MLContext.Model.Save/Load is not supported for typed decisions; " +
            "use the portable typed-decision Save/Load API.");

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _scorer.Dispose();
        _bundle.Dispose();
    }

    internal void AttachOwnedAssetDirectory(string rootPath)
        => _bundle.AttachOwnedRoot(rootPath);

    internal void AttachOwnedAssetDirectory(
        string rootPath,
        TypedDecisionRootLease lease)
        => _bundle.AttachOwnedRoot(rootPath, lease);
}

public sealed class DecisionDecodingTransformer : ITransformer, IDisposable
{
    private readonly DecisionDecodingOptions _options;
    private readonly TypedDecisionBundle _bundle;
    private readonly DecodeDecisions _decoder;
    private bool _disposed;

    internal DecisionDecodingTransformer(MLContext mlContext, DecisionDecodingOptions options)
    {
        _options = options;
        _bundle = TypedDecisionBundle.Open(
            options.ModelAssetsPath,
            TypedDecisionBundleLoadRequirements.Profile);
        _decoder = new DecodeDecisions(_bundle.Profile.TemperaturePolicy, _bundle.Manifest.Decoder);
    }

    internal DecisionDecodingOptions Options => _options;
    internal string AssetsRootPath
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _bundle.RootPath;
        }
    }

    /// <summary>Saves this transformer as a portable typed-decision artifact.</summary>
    public void Save(string path) => TypedDecisionPortableModel.Save(this, path);

    /// <summary>Loads a transformer from a portable typed-decision artifact.</summary>
    public static DecisionDecodingTransformer Load(MLContext mlContext, string path)
        => TypedDecisionPortableModel.LoadDecoding(mlContext, path);

    public bool IsRowToRowMapper => true;

    public IDataView Transform(IDataView input)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        DecisionSchema.ValidateScoringColumns(input.Schema, _options);
        return new DecisionDecodingDataView(input, _decoder, _options);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        DecisionSchema.ValidateScoringColumns(inputSchema, _options);
        return DecisionSchema.AddResults(inputSchema, _options);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        DecisionSchema.ValidateScoringColumns(inputSchema, _options);
        return new DecisionDecodingRowToRowMapper(inputSchema, _decoder, _options);
    }

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "Native MLContext.Model.Save/Load is not supported for typed decisions; " +
            "use the portable typed-decision Save/Load API.");

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _bundle.Dispose();
    }

    internal void AttachOwnedAssetDirectory(string rootPath)
        => _bundle.AttachOwnedRoot(rootPath);

    internal void AttachOwnedAssetDirectory(
        string rootPath,
        TypedDecisionRootLease lease)
        => _bundle.AttachOwnedRoot(rootPath, lease);
}

internal sealed class DecisionInferenceEngine : IDisposable
{
    private readonly TypedDecisionBundle _bundle;
    private readonly PrepareDecisionInputs _preparer;
    private readonly ScoreOnnxDecisionModel _scorer;
    private readonly DecodeDecisions _decoder;
    private bool _disposed;

    internal DecisionInferenceEngine(MLContext mlContext, string bundlePath)
    {
        _bundle = TypedDecisionBundle.Open(
            bundlePath,
            TypedDecisionBundleLoadRequirements.Model |
            TypedDecisionBundleLoadRequirements.Profile |
            TypedDecisionBundleLoadRequirements.Tokenizer);
        try
        {
            _preparer = new PrepareDecisionInputs(
                _bundle.Profile,
                _bundle.Tokenizer.Tokenizer,
                _bundle.Tokenizer.Metadata);
            _scorer = new ScoreOnnxDecisionModel(
                _bundle,
                new OnnxExecutionOptions(
                    mlContext.GpuDeviceId,
                    mlContext.FallbackToCpu,
                    static message =>
                        Console.Error.WriteLine($"[MLNet.TextInference.Onnx] {message}")));
            _decoder = new DecodeDecisions(_bundle.Profile.TemperaturePolicy, _bundle.Manifest.Decoder);
        }
        catch
        {
            _bundle.Dispose();
            throw;
        }
    }

    internal void AttachOwnedAssetDirectory(string rootPath)
        => _bundle.AttachOwnedRoot(rootPath);

    internal void AttachOwnedAssetDirectory(
        string rootPath,
        TypedDecisionRootLease lease)
        => _bundle.AttachOwnedRoot(rootPath, lease);

    internal IReadOnlyList<DecisionResponse> Infer(
        IReadOnlyList<string> states,
        IReadOnlyList<DecisionQuestion> questions,
        int batchSize = int.MaxValue)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(states);
        if (states.Count == 0)
            return [];
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

        var responses = new List<DecisionResponse>(states.Count);
        for (var offset = 0; offset < states.Count; offset += batchSize)
        {
            var count = Math.Min(batchSize, states.Count - offset);
            var requests = new DecisionRequest[count];
            for (var index = 0; index < count; index++)
                requests[index] = DecisionRequest.Create(states[offset + index], questions);
            responses.AddRange(InferBatch(requests));
        }

        return responses;
    }

    internal string AssetsRootPath
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _bundle.RootPath;
        }
    }

    private IReadOnlyList<DecisionResponse> InferBatch(
        IReadOnlyList<DecisionRequest> requests)
    {
        var inputs = _preparer.Prepare(requests);
        var outputs = _scorer.Score(inputs);
        var decoded = _decoder.Decode(inputs, outputs);
        var builders = Enumerable.Range(0, requests.Count)
            .Select(static _ => new ResponseBuilder())
            .ToArray();
        for (var row = 0; row < inputs.Items.Count; row++)
        {
            var requestIndex = inputs.Items[row].RequestIndex;
            builders[requestIndex].Results.Add(decoded.Results[row]);
            var attention = inputs.AttentionMask.AsSpan(row * inputs.SequenceLength, inputs.SequenceLength);
            for (var token = 0; token < attention.Length; token++)
            {
                if (attention[token] != 0)
                    builders[requestIndex].InputTokenCount++;
            }
        }

        return builders.Select(static builder => builder.Build()).ToArray();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _scorer.Dispose();
        _bundle.Dispose();
    }

    private sealed class ResponseBuilder
    {
        internal List<DecisionResult> Results { get; } = [];
        internal int InputTokenCount { get; set; }

        internal DecisionResponse Build() => new()
        {
            Results = Results,
            InputTokenCount = InputTokenCount
        };
    }
}

internal static class DecisionSchema
{
    internal static DataViewSchema AddPreparation(
        DataViewSchema input,
        DecisionInputPreparationOptions options)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        AddPreparationColumns(builder, options);
        return builder.ToSchema();
    }

    internal static SchemaShape AddPreparation(
        SchemaShape input,
        DecisionInputPreparationOptions options)
    {
        var result = input.ToDictionary(column => column.Name, StringComparer.Ordinal);
        AddShapeColumn(result, options.InputIdsColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.Int64);
        AddShapeColumn(result, options.AttentionMaskColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.Int64);
        AddShapeColumn(result, options.MarkerPositionsColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.Int64);
        AddShapeColumn(result, options.MarkerMaskColumnName, SchemaShape.Column.VectorKind.VariableVector, BooleanDataViewType.Instance);
        AddShapeColumn(result, options.QuestionTypesColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.Int64);
        AddShapeColumn(result, options.BatchSizeColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Int32);
        AddShapeColumn(result, options.SequenceLengthColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Int32);
        AddShapeColumn(result, options.MarkerWidthColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Int32);
        return new SchemaShape(result.Values);
    }

    internal static DataViewSchema AddScoring(
        DataViewSchema input,
        OnnxDecisionModelScorerOptions options)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        builder.AddColumn(options.LogitsColumnName, new VectorDataViewType(NumberDataViewType.Single));
        builder.AddColumn(options.ActionProbabilitiesColumnName, new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    internal static SchemaShape AddScoring(
        SchemaShape input,
        OnnxDecisionModelScorerOptions options)
    {
        var result = input.ToDictionary(column => column.Name, StringComparer.Ordinal);
        AddShapeColumn(result, options.LogitsColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.Single);
        AddShapeColumn(result, options.ActionProbabilitiesColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.Single);
        return new SchemaShape(result.Values);
    }

    internal static DataViewSchema AddResults(
        DataViewSchema input,
        OnnxTypedDecisionsOptions options)
        => AddResults(input, options.OutputNames, options.ResultsColumnName, options.Questions);

    internal static DataViewSchema AddResults(
        DataViewSchema input,
        DecisionDecodingOptions options)
        => AddResults(input, options.OutputNames, options.ResultsColumnName, options.Questions);

    internal static SchemaShape AddResults(
        SchemaShape input,
        OnnxTypedDecisionsOptions options)
        => AddResults(input, options.OutputNames, options.ResultsColumnName, options.Questions);

    internal static SchemaShape AddResults(
        SchemaShape input,
        DecisionDecodingOptions options)
        => AddResults(input, options.OutputNames, options.ResultsColumnName, options.Questions);

    internal static void ValidatePreparationColumns(
        DataViewSchema schema,
        OnnxDecisionModelScorerOptions options)
    {
        foreach (var name in new[]
        {
            options.InputIdsColumnName, options.AttentionMaskColumnName,
            options.MarkerPositionsColumnName, options.QuestionTypesColumnName
        })
        {
            ValidateVectorColumn(schema, name, NumberDataViewType.Int64);
        }
        ValidateVectorColumn(schema, options.MarkerMaskColumnName, BooleanDataViewType.Instance);

        ValidateIntColumn(schema, options.BatchSizeColumnName);
        ValidateIntColumn(schema, options.SequenceLengthColumnName);
        ValidateIntColumn(schema, options.MarkerWidthColumnName);
    }

    internal static void ValidatePreparationColumns(
        SchemaShape schema,
        OnnxDecisionModelScorerOptions options)
    {
        foreach (var name in new[]
        {
            options.InputIdsColumnName, options.AttentionMaskColumnName,
            options.MarkerPositionsColumnName, options.QuestionTypesColumnName
        })
        {
            var column = schema.FirstOrDefault(c => c.Name == name);
            ValidateShapeVector(column, name, NumberDataViewType.Int64);
        }
        ValidateShapeVector(
            schema.FirstOrDefault(c => c.Name == options.MarkerMaskColumnName),
            options.MarkerMaskColumnName,
            BooleanDataViewType.Instance);
    }

    internal static void ValidateScoringColumns(
        DataViewSchema schema,
        DecisionDecodingOptions options)
    {
        var scorerOptions = new OnnxDecisionModelScorerOptions
        {
            ModelAssetsPath = options.ModelAssetsPath,
            InputIdsColumnName = options.InputIdsColumnName,
            AttentionMaskColumnName = options.AttentionMaskColumnName,
            MarkerPositionsColumnName = options.MarkerPositionsColumnName,
            MarkerMaskColumnName = options.MarkerMaskColumnName,
            QuestionTypesColumnName = options.QuestionTypesColumnName,
            BatchSizeColumnName = options.BatchSizeColumnName,
            SequenceLengthColumnName = options.SequenceLengthColumnName,
            MarkerWidthColumnName = options.MarkerWidthColumnName,
            LogitsColumnName = options.LogitsColumnName,
            ActionProbabilitiesColumnName = options.ActionProbabilitiesColumnName
        };
        ValidatePreparationColumns(schema, scorerOptions);
        ValidateVectorColumn(schema, options.LogitsColumnName, NumberDataViewType.Single);
        ValidateVectorColumn(schema, options.ActionProbabilitiesColumnName, NumberDataViewType.Single);
    }

    internal static void ValidateScoringColumns(
        SchemaShape schema,
        DecisionDecodingOptions options)
    {
        var scorerOptions = new OnnxDecisionModelScorerOptions
        {
            ModelAssetsPath = options.ModelAssetsPath,
            InputIdsColumnName = options.InputIdsColumnName,
            AttentionMaskColumnName = options.AttentionMaskColumnName,
            MarkerPositionsColumnName = options.MarkerPositionsColumnName,
            MarkerMaskColumnName = options.MarkerMaskColumnName,
            QuestionTypesColumnName = options.QuestionTypesColumnName,
            BatchSizeColumnName = options.BatchSizeColumnName,
            SequenceLengthColumnName = options.SequenceLengthColumnName,
            MarkerWidthColumnName = options.MarkerWidthColumnName,
            LogitsColumnName = options.LogitsColumnName,
            ActionProbabilitiesColumnName = options.ActionProbabilitiesColumnName
        };
        ValidatePreparationColumns(schema, scorerOptions);
        ValidateShapeVector(
            schema.FirstOrDefault(c => c.Name == options.LogitsColumnName),
            options.LogitsColumnName,
            NumberDataViewType.Single);
        ValidateShapeVector(
            schema.FirstOrDefault(c => c.Name == options.ActionProbabilitiesColumnName),
            options.ActionProbabilitiesColumnName,
            NumberDataViewType.Single);
    }

    private static DataViewSchema AddResults(
        DataViewSchema input,
        DecisionOutputNames names,
        string resultsColumn,
        IReadOnlyList<DecisionQuestion> questions)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        builder.AddColumn(resultsColumn, TextDataViewType.Instance);
        foreach (var question in questions)
            AddQuestionColumns(builder, names.ById[question.Id], question);
        return builder.ToSchema();
    }

    private static SchemaShape AddResults(
        SchemaShape input,
        DecisionOutputNames names,
        string resultsColumn,
        IReadOnlyList<DecisionQuestion> questions)
    {
        var result = input.ToDictionary(column => column.Name, StringComparer.Ordinal);
        AddShapeColumn(result, resultsColumn, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance);
        foreach (var question in questions)
        {
            var output = names.ById[question.Id];
            AddShapeColumnIfPresent(result, output.PredictedLabel, SchemaShape.Column.VectorKind.Scalar,
                question.Type == DecisionQuestionType.Noul ? BooleanDataViewType.Instance : TextDataViewType.Instance);
            AddShapeColumnIfPresent(result, output.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single);
            AddShapeColumnIfPresent(result, output.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single);
            if (output.Probabilities is not null)
            {
                AddShapeColumn(
                    result,
                    output.Probabilities,
                    SchemaShape.Column.VectorKind.Vector,
                    NumberDataViewType.Single,
                    CreateSlotMetadataShape(question.OptionLabels().Count));
            }
            AddShapeColumn(result, output.Confidence, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single);
            AddShapeColumn(result, output.ActionProbability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single);
        }

        return new SchemaShape(result.Values);
    }

    private static void AddPreparationColumns(
        DataViewSchema.Builder builder,
        DecisionInputPreparationOptions options)
    {
        builder.AddColumn(options.InputIdsColumnName, new VectorDataViewType(NumberDataViewType.Int64));
        builder.AddColumn(options.AttentionMaskColumnName, new VectorDataViewType(NumberDataViewType.Int64));
        builder.AddColumn(options.MarkerPositionsColumnName, new VectorDataViewType(NumberDataViewType.Int64));
        builder.AddColumn(options.MarkerMaskColumnName, new VectorDataViewType(BooleanDataViewType.Instance));
        builder.AddColumn(options.QuestionTypesColumnName, new VectorDataViewType(NumberDataViewType.Int64));
        builder.AddColumn(options.BatchSizeColumnName, NumberDataViewType.Int32);
        builder.AddColumn(options.SequenceLengthColumnName, NumberDataViewType.Int32);
        builder.AddColumn(options.MarkerWidthColumnName, NumberDataViewType.Int32);
    }

    private static void AddQuestionColumns(
        DataViewSchema.Builder builder,
        QuestionOutputNames output,
        DecisionQuestion question)
    {
        if (output.PredictedLabel is not null)
        {
            builder.AddColumn(
                output.PredictedLabel,
                question.Type == DecisionQuestionType.Noul
                    ? BooleanDataViewType.Instance
                    : TextDataViewType.Instance);
        }

        if (output.Score is not null)
            builder.AddColumn(output.Score, NumberDataViewType.Single);
        if (output.Probability is not null)
            builder.AddColumn(output.Probability, NumberDataViewType.Single);
        if (output.Probabilities is not null)
        {
            builder.AddColumn(
                output.Probabilities,
                new VectorDataViewType(NumberDataViewType.Single, question.OptionLabels().Count),
                CreateSlotMetadata(question.OptionLabels()));
        }
        builder.AddColumn(output.Confidence, NumberDataViewType.Single);
        builder.AddColumn(output.ActionProbability, NumberDataViewType.Single);
    }

    private static void AddShapeColumnIfPresent(
        IDictionary<string, SchemaShape.Column> columns,
        string? name,
        SchemaShape.Column.VectorKind kind,
        DataViewType type)
    {
        if (name is not null)
            AddShapeColumn(columns, name, kind, type);
    }

    private static DataViewSchema.Annotations CreateSlotMetadata(
        IReadOnlyList<string> optionLabels)
    {
        var builder = new DataViewSchema.Annotations.Builder();
        var values = optionLabels.Select(static label => label.AsMemory()).ToArray();
        builder.Add(
            "SlotNames",
            new VectorDataViewType(TextDataViewType.Instance, values.Length),
            (ref VBuffer<ReadOnlyMemory<char>> destination) =>
            {
                var editor = VBufferEditor.Create(ref destination, values.Length);
                values.AsSpan().CopyTo(editor.Values);
                destination = editor.Commit();
            });
        return builder.ToAnnotations();
    }

    private static SchemaShape CreateSlotMetadataShape(int slotCount)
    {
        var columns = new Dictionary<string, SchemaShape.Column>(StringComparer.Ordinal);
        AddShapeColumn(
            columns,
            "SlotNames",
            SchemaShape.Column.VectorKind.Vector,
            TextDataViewType.Instance);
        return new SchemaShape(columns.Values);
    }

    private static void ValidateIntColumn(DataViewSchema schema, string name)
    {
        var column = schema.GetColumnOrNull(name)
            ?? throw new ArgumentException($"Input schema does not contain column '{name}'.");
        if (column.Type != NumberDataViewType.Int32)
            throw new ArgumentException($"Column '{name}' must be Int32.");
    }

    private static void ValidateVectorColumn(
        DataViewSchema schema,
        string name,
        DataViewType itemType)
    {
        var column = schema.GetColumnOrNull(name)
            ?? throw new ArgumentException($"Input schema does not contain column '{name}'.");
        if (column.Type is not VectorDataViewType vector ||
            vector.ItemType != itemType)
            throw new ArgumentException(
                $"Column '{name}' must be a vector of {itemType}.");
    }

    private static void ValidateShapeVector(
        SchemaShape.Column column,
        string name,
        DataViewType itemType)
    {
        if (column.Name == null ||
            column.Kind is not SchemaShape.Column.VectorKind.Vector and
                not SchemaShape.Column.VectorKind.VariableVector ||
            column.ItemType != itemType)
            throw new ArgumentException(
                $"Column '{name}' must be a vector of {itemType}.");
    }

    private static void AddShapeColumn(
        IDictionary<string, SchemaShape.Column> columns,
        string name,
        SchemaShape.Column.VectorKind kind,
        DataViewType type)
        => AddShapeColumn(columns, name, kind, type, null);

    private static void AddShapeColumn(
        IDictionary<string, SchemaShape.Column> columns,
        string name,
        SchemaShape.Column.VectorKind kind,
        DataViewType type,
        SchemaShape? metadata)
    {
        var constructor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        columns[name] = (SchemaShape.Column)constructor.Invoke([name, kind, type, false, metadata]);
    }
}

internal static class TypedDecisionValidation
{
    internal static void ValidateTextColumn(DataViewSchema schema, string name)
    {
        var column = schema.GetColumnOrNull(name)
            ?? throw new ArgumentException($"Input schema does not contain column '{name}'.");
        if (column.Type != TextDataViewType.Instance)
            throw new ArgumentException($"Column '{name}' must be a scalar Text column.");
    }

    internal static void ValidateTextColumn(SchemaShape schema, string name)
    {
        var column = schema.FirstOrDefault(c => c.Name == name);
        if (column.Name == null ||
            column.ItemType != TextDataViewType.Instance ||
            column.Kind != SchemaShape.Column.VectorKind.Scalar)
            throw new ArgumentException($"Column '{name}' must be a scalar Text column.");
    }
}

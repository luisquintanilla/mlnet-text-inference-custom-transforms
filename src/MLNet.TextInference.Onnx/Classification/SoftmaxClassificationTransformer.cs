using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that applies softmax over raw logits to produce
/// class probabilities and a predicted label.
///
/// Lazy evaluation: Transform() returns a wrapping IDataView. Classification
/// is computed per-row as the cursor advances.
/// </summary>
public sealed class SoftmaxClassificationTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly SoftmaxClassificationOptions _options;

    public bool IsRowToRowMapper => true;

    internal SoftmaxClassificationOptions Options => _options;
    public int NumClasses => _options.NumClasses!.Value;
    public string[]? Labels => _options.Labels;

    internal SoftmaxClassificationTransformer(
        MLContext mlContext,
        SoftmaxClassificationOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    /// <summary>
    /// ML.NET face: returns a wrapping IDataView. No computation happens here.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        return new ClassificationDataView(input, _options);
    }

    /// <summary>
    /// Direct face: classify raw outputs without IDataView overhead.
    /// </summary>
    internal ClassificationResult[] Classify(float[][] rawOutputs)
    {
        var results = new ClassificationResult[rawOutputs.Length];

        for (int i = 0; i < rawOutputs.Length; i++)
        {
            var logits = rawOutputs[i];
            var probabilities = new float[logits.Length];
            TensorPrimitives.SoftMax(logits, probabilities);

            int predictedIndex = TensorPrimitives.IndexOfMax(probabilities);
            string predictedLabel = _options.Labels != null && predictedIndex < _options.Labels.Length
                ? _options.Labels[predictedIndex]
                : predictedIndex.ToString();

            results[i] = new ClassificationResult
            {
                PredictedLabel = predictedLabel,
                Probabilities = probabilities
            };
        }

        return results;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.ProbabilitiesColumnName,
            new VectorDataViewType(NumberDataViewType.Single, _options.NumClasses!.Value));
        builder.AddColumn(_options.PredictedLabelColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

/// <summary>
/// Wrapping IDataView that adds probabilities and predicted label columns.
/// </summary>
internal sealed class ClassificationDataView : IDataView
{
    private readonly IDataView _input;
    private readonly SoftmaxClassificationOptions _options;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal ClassificationDataView(IDataView input, SoftmaxClassificationOptions options)
    {
        _input = input;
        _options = options;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);
        builder.AddColumn(options.ProbabilitiesColumnName,
            new VectorDataViewType(NumberDataViewType.Single, options.NumClasses!.Value));
        builder.AddColumn(options.PredictedLabelColumnName, TextDataViewType.Instance);
        Schema = builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var upstreamCols = new List<DataViewSchema.Column>();
        foreach (var col in columnsNeeded)
        {
            var inputCol = _input.Schema.GetColumnOrNull(col.Name);
            if (inputCol != null)
                upstreamCols.Add(inputCol.Value);
        }

        // Always need raw output for classification
        upstreamCols.Add(_input.Schema[_options.InputColumnName]);

        var inputCursor = _input.GetRowCursor(upstreamCols.Distinct(), rand);
        return new ClassificationCursor(this, inputCursor, _options);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor that applies softmax classification one row at a time.
/// </summary>
internal sealed class ClassificationCursor : DataViewRowCursor
{
    private readonly ClassificationDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly SoftmaxClassificationOptions _options;

    private float[]? _currentProbabilities;
    private string? _currentLabel;

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _inputCursor.Position;
    public override long Batch => _inputCursor.Batch;

    internal ClassificationCursor(
        ClassificationDataView parent,
        DataViewRowCursor inputCursor,
        SoftmaxClassificationOptions options)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _options = options;
    }

    public override bool MoveNext()
    {
        if (!_inputCursor.MoveNext())
            return false;

        // Read raw logits from upstream
        var rawOutputCol = _inputCursor.Schema[_options.InputColumnName];
        var rawOutputGetter = _inputCursor.GetGetter<VBuffer<float>>(rawOutputCol);
        VBuffer<float> rawOutputBuffer = default;
        rawOutputGetter(ref rawOutputBuffer);

        var logits = rawOutputBuffer.DenseValues().ToArray();
        _currentProbabilities = new float[logits.Length];
        TensorPrimitives.SoftMax(logits, _currentProbabilities);

        int predictedIndex = TensorPrimitives.IndexOfMax(_currentProbabilities);
        _currentLabel = _options.Labels != null && predictedIndex < _options.Labels.Length
            ? _options.Labels[predictedIndex]
            : predictedIndex.ToString();

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        // Probabilities column
        if (column.Name == _options.ProbabilitiesColumnName)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            {
                var editor = VBufferEditor.Create(ref value, _currentProbabilities!.Length);
                _currentProbabilities.AsSpan().CopyTo(editor.Values);
                value = editor.Commit();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // PredictedLabel column
        if (column.Name == _options.PredictedLabelColumnName)
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
            {
                value = _currentLabel.AsMemory();
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // Passthrough columns — delegate to upstream cursor
        var inputCol = _inputCursor.Schema.GetColumnOrNull(column.Name);
        if (inputCol != null)
            return _inputCursor.GetGetter<TValue>(inputCol.Value);

        throw new InvalidOperationException($"Unknown column: {column.Name}");
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

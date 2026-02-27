using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// ML.NET ITransformer that applies sigmoid to raw model logits to produce a score.
/// Reads the first element of the raw output vector and applies sigmoid.
///
/// Lazy evaluation: Transform() returns a wrapping IDataView.
/// Sigmoid is computed per-row as the cursor advances.
/// </summary>
public sealed class SigmoidScorerTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly SigmoidScorerOptions _options;

    public bool IsRowToRowMapper => true;

    internal SigmoidScorerOptions Options => _options;

    internal SigmoidScorerTransformer(MLContext mlContext, SigmoidScorerOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    /// <summary>
    /// ML.NET face: returns a wrapping IDataView. No computation happens here.
    /// </summary>
    public IDataView Transform(IDataView input)
    {
        return new SigmoidDataView(input, _options);
    }

    /// <summary>
    /// Direct face: apply sigmoid to raw outputs without IDataView overhead.
    /// </summary>
    internal float[] Score(float[][] rawOutputs)
    {
        var scores = new float[rawOutputs.Length];
        for (int i = 0; i < rawOutputs.Length; i++)
        {
            float logit = rawOutputs[i][0];
            scores[i] = 1.0f / (1.0f + MathF.Exp(-logit));
        }
        return scores;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);
        builder.AddColumn(_options.OutputColumnName, NumberDataViewType.Single);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

/// <summary>
/// Wrapping IDataView that adds the sigmoid score column to the upstream schema.
/// </summary>
internal sealed class SigmoidDataView : IDataView
{
    private readonly IDataView _input;
    private readonly SigmoidScorerOptions _options;

    public DataViewSchema Schema { get; }
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    internal SigmoidDataView(IDataView input, SigmoidScorerOptions options)
    {
        _input = input;
        _options = options;

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input.Schema);
        builder.AddColumn(options.OutputColumnName, NumberDataViewType.Single);
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

        // Always need the raw output column for sigmoid computation
        upstreamCols.Add(_input.Schema[_options.InputColumnName]);

        var inputCursor = _input.GetRowCursor(upstreamCols.Distinct(), rand);
        return new SigmoidCursor(this, inputCursor, _options);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }
}

/// <summary>
/// Cursor that applies sigmoid to the raw output one row at a time.
/// </summary>
internal sealed class SigmoidCursor : DataViewRowCursor
{
    private readonly SigmoidDataView _parent;
    private readonly DataViewRowCursor _inputCursor;
    private readonly SigmoidScorerOptions _options;

    private float _currentScore;

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _inputCursor.Position;
    public override long Batch => _inputCursor.Batch;

    internal SigmoidCursor(
        SigmoidDataView parent,
        DataViewRowCursor inputCursor,
        SigmoidScorerOptions options)
    {
        _parent = parent;
        _inputCursor = inputCursor;
        _options = options;
    }

    public override bool MoveNext()
    {
        if (!_inputCursor.MoveNext())
            return false;

        var rawOutputCol = _inputCursor.Schema[_options.InputColumnName];
        var rawOutputGetter = _inputCursor.GetGetter<VBuffer<float>>(rawOutputCol);
        VBuffer<float> rawOutputBuffer = default;
        rawOutputGetter(ref rawOutputBuffer);

        float logit = rawOutputBuffer.DenseValues().First();
        _currentScore = 1.0f / (1.0f + MathF.Exp(-logit));

        return true;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Name == _options.OutputColumnName)
        {
            ValueGetter<float> getter = (ref float value) =>
            {
                value = _currentScore;
            };
            return (ValueGetter<TValue>)(object)getter;
        }

        // For all passthrough columns, delegate directly to upstream cursor
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

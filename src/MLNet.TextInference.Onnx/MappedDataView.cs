using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.TextInference.Onnx;

/// <summary>
/// Adapts a row mapper to an IDataView while preserving ML.NET's requested-column
/// and cursor ownership semantics. Cross-row batching remains implemented by the
/// dedicated ONNX scorer data view.
/// </summary>
internal sealed class MappedDataView : IDataView
{
    private readonly IDataView _input;
    private readonly IRowToRowMapper _mapper;

    internal MappedDataView(IDataView input, IRowToRowMapper mapper)
    {
        _input = input ?? throw new ArgumentNullException(nameof(input));
        _mapper = mapper ?? throw new ArgumentNullException(nameof(mapper));
        if (!ReferenceEquals(input.Schema, mapper.InputSchema))
            throw new ArgumentException(
                "The mapper must be created for the exact input schema.",
                nameof(mapper));
    }

    public DataViewSchema Schema => _mapper.OutputSchema;
    public bool CanShuffle => _input.CanShuffle;
    public long? GetRowCount() => _input.GetRowCount();

    public DataViewRowCursor GetRowCursor(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        Random? rand = null)
    {
        ArgumentNullException.ThrowIfNull(columnsNeeded);
        var requested = columnsNeeded.ToArray();
        var dependencies = _mapper.GetDependencies(requested).ToArray();
        var inputCursor = _input.GetRowCursor(dependencies, rand);
        return new MappedCursor(this, inputCursor, requested);
    }

    public DataViewRowCursor[] GetRowCursorSet(
        IEnumerable<DataViewSchema.Column> columnsNeeded,
        int n,
        Random? rand = null)
        => [GetRowCursor(columnsNeeded, rand)];

    private sealed class MappedCursor : DataViewRowCursor
    {
        private readonly MappedDataView _parent;
        private readonly DataViewRowCursor _inputCursor;
        private readonly DataViewSchema.Column[] _requested;
        private DataViewRow? _mappedRow;
        private bool _disposed;

        internal MappedCursor(
            MappedDataView parent,
            DataViewRowCursor inputCursor,
            DataViewSchema.Column[] requested)
        {
            _parent = parent;
            _inputCursor = inputCursor;
            _requested = requested;
            try
            {
                _mappedRow = _parent._mapper.GetRow(_inputCursor, _requested);
            }
            catch
            {
                _inputCursor.Dispose();
                throw;
            }
        }

        public override DataViewSchema Schema => _parent.Schema;
        public override long Position => _inputCursor.Position;
        public override long Batch => _inputCursor.Batch;

        public override bool MoveNext()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _inputCursor.MoveNext();
        }

        public override ValueGetter<TValue> GetGetter<TValue>(
            DataViewSchema.Column column)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_requested.Any(requested => requested.Index == column.Index))
                throw new InvalidOperationException(
                    $"Column '{column.Name}' was not requested by this cursor.");
            return _mappedRow!.GetGetter<TValue>(column);
        }

        public override ValueGetter<DataViewRowId> GetIdGetter()
            => _inputCursor.GetIdGetter();

        public override bool IsColumnActive(DataViewSchema.Column column)
            => _requested.Any(requested => requested.Index == column.Index);

        protected override void Dispose(bool disposing)
        {
            if (disposing && !_disposed)
            {
                _disposed = true;
                _mappedRow?.Dispose();
                _mappedRow = null;
            }

            base.Dispose(disposing);
        }
    }
}

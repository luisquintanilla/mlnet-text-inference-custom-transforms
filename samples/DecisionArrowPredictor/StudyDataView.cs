using Microsoft.ML;
using Microsoft.ML.Data;

namespace DecisionArrowPredictor;

public sealed class StudyDataView : IDataView
{
    private readonly RowSelection selection;
    public static DataViewSchema SharedSchema { get; } = CreateSchema();
    public DataViewSchema Schema => SharedSchema;
    // ML.NET 5 LoadFromEnumerable uses StreamingDataView, not the shuffling ListDataView.
    public bool CanShuffle => false;
    public long? GetRowCount() { selection.Owner.RequireOpen(); return selection.Count; }

    public StudyDataView(RowSelection selection)
    {
        selection.Owner.RequireOpen();
        this.selection = selection;
    }

    private static DataViewSchema CreateSchema()
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(nameof(LearningRow.RowId), NumberDataViewType.Int64);
        builder.AddColumn(nameof(LearningRow.GroupId), NumberDataViewType.Int64);
        builder.AddColumn(nameof(LearningRow.Label), BooleanDataViewType.Instance);
        builder.AddColumn(nameof(LearningRow.Text), TextDataViewType.Instance);
        builder.AddColumn(nameof(LearningRow.Semantic), new VectorDataViewType(NumberDataViewType.Single, FeatureContract.Width));
        builder.AddColumn(nameof(LearningRow.SpamBaseline), NumberDataViewType.Double);
        return builder.ToSchema();
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null) =>
        new Cursor(selection, columnsNeeded);

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        return [GetRowCursor(columnsNeeded, rand)];
    }

    private sealed class Cursor : DataViewRowCursor
    {
        private readonly RowSelection selection;
        private readonly bool[] active = new bool[SharedSchema.Count];
        private readonly Delegate?[] getters = new Delegate?[SharedSchema.Count];
        private bool disposed;
        private long position = -1;
        public override long Position => position;
        public override long Batch => 0;
        public override DataViewSchema Schema => SharedSchema;

        public Cursor(RowSelection selection, IEnumerable<DataViewSchema.Column> columns)
        {
            this.selection = selection;
            foreach (var column in columns)
            {
                CheckColumn(column);
                active[column.Index] = true;
            }
            if (active[0]) getters[0] = (ValueGetter<long>)((ref long value) => value = Current.RowId);
            if (active[1]) getters[1] = (ValueGetter<long>)((ref long value) => value = Current.GroupId);
            if (active[2]) getters[2] = (ValueGetter<bool>)((ref bool value) => value = Current.Label);
            if (active[3]) getters[3] = (ValueGetter<ReadOnlyMemory<char>>)((ref ReadOnlyMemory<char> value) => value = Current.Text.AsMemory());
            if (active[4]) getters[4] = (ValueGetter<VBuffer<float>>)Semantic;
            if (active[5]) getters[5] = (ValueGetter<double>)((ref double value) => value = selection.Owner.Probabilities.Direct(Ordinal));
            selection.Owner.AcquireCursor();
        }

        private int Ordinal
        {
            get
            {
                if (disposed || position < 0 || position >= selection.Count)
                    throw new InvalidOperationException("Cursor is not positioned on a live row.");
                return selection[(int)position];
            }
        }

        private StudyRowMetadata Current => selection.Owner.Metadata[Ordinal];

        private void Semantic(ref VBuffer<float> value)
        {
            int ordinal = Ordinal;
            var editor = VBufferEditor.Create(ref value, FeatureContract.Width);
            selection.Owner.Probabilities.CopySemantic(ordinal, editor.Values);
            value = editor.Commit();
        }

        public override ValueGetter<DataViewRowId> GetIdGetter() => Id;
        private void Id(ref DataViewRowId value)
        {
            _ = Ordinal;
            value = new DataViewRowId((ulong)position, 0);
        }

        private static void CheckColumn(DataViewSchema.Column column)
        {
            if ((uint)column.Index >= (uint)SharedSchema.Count ||
                column.Name != SharedSchema[column.Index].Name ||
                !column.Type.Equals(SharedSchema[column.Index].Type))
                throw new ArgumentException("Column is not from the study schema.", nameof(column));
        }

        public override bool IsColumnActive(DataViewSchema.Column column)
        {
            CheckColumn(column);
            return active[column.Index];
        }

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            CheckColumn(column);
            if (!active[column.Index]) throw new InvalidOperationException("Requested column is inactive.");
            return getters[column.Index] as ValueGetter<TValue> ??
                throw new InvalidOperationException("Requested getter type differs from the column type.");
        }

        public override bool MoveNext()
        {
            if (disposed) return false;
            if (position + 1 < selection.Count) { position++; return true; }
            position = -1;
            Dispose();
            return false;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && !disposed)
            {
                disposed = true;
                position = -1;
                selection.Owner.ReleaseCursor();
            }
            base.Dispose(disposing);
        }
    }
}

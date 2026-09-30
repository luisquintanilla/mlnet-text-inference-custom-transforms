using System.Buffers.Binary;
using System.Security.Cryptography;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DecisionArrowPredictor;

public sealed record CursorTrace(int Request, int Cursor, string Method, int RequestedCount, bool RandomSupplied,
    string[] ActiveColumns, long Rows, bool Complete, string RowOrderSha256, long FeatureGetterCalls,
    string FeatureBitsSha256, long LabelGetterCalls, string LabelBitsSha256);

// Diagnostic wrapper only: it forwards cursor policy and never advances the supplied Random.
public sealed class DataViewTrace : IDataView
{
    private readonly IDataView source;
    private readonly long[] sourceIds;
    private readonly List<CursorTrace> completed = [];
    private readonly object gate = new();
    private int request;
    public DataViewSchema Schema => source.Schema;
    public bool CanShuffle => source.CanShuffle;
    public string SourceType => source.GetType().FullName ?? source.GetType().Name;
    public long? GetRowCount() => source.GetRowCount();

    public DataViewTrace(IDataView source, IReadOnlyList<long> sourceIds)
    {
        this.source = source;
        this.sourceIds = [.. sourceIds];
    }

    public CursorTrace[] Snapshot()
    {
        lock (gate) return [.. completed.OrderBy(t => t.Request).ThenBy(t => t.Cursor)];
    }

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var columns = columnsNeeded.ToArray();
        return Wrap(source.GetRowCursor(columns, rand), columns, Interlocked.Increment(ref request), 0, "single", 1, rand);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        var columns = columnsNeeded.ToArray();
        int current = Interlocked.Increment(ref request);
        var cursors = source.GetRowCursorSet(columns, n, rand);
        return cursors.Select((cursor, i) => Wrap(cursor, columns, current, i, "set", n, rand)).ToArray();
    }

    private Cursor Wrap(DataViewRowCursor cursor, DataViewSchema.Column[] columns, int current, int index,
        string method, int count, Random? random) => new(this, cursor, current, index, method, count,
            random is not null, columns.Select(c => c.Name).Order(StringComparer.Ordinal).ToArray());

    private void Add(CursorTrace trace) { lock (gate) completed.Add(trace); }

    private sealed class Cursor : DataViewRowCursor
    {
        private readonly DataViewTrace owner;
        private readonly DataViewRowCursor source;
        private readonly ValueGetter<DataViewRowId> getId;
        private readonly IncrementalHash order = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        private readonly IncrementalHash features = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        private readonly IncrementalHash labels = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        private readonly int request, index, requestedCount;
        private readonly string method;
        private readonly bool randomSupplied;
        private readonly string[] active;
        private DataViewRowId currentId;
        private long rows, featureCalls, labelCalls;
        private bool complete, disposed;
        public override DataViewSchema Schema => source.Schema;
        public override long Position => source.Position;
        public override long Batch => source.Batch;

        public Cursor(DataViewTrace owner, DataViewRowCursor source, int request, int index, string method,
            int requestedCount, bool randomSupplied, string[] active)
        {
            this.owner = owner; this.source = source; this.request = request; this.index = index;
            this.method = method; this.requestedCount = requestedCount;
            this.randomSupplied = randomSupplied; this.active = active;
            getId = source.GetIdGetter();
        }

        public override ValueGetter<DataViewRowId> GetIdGetter() => source.GetIdGetter();
        public override bool IsColumnActive(DataViewSchema.Column column) => source.IsColumnActive(column);

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            var getter = source.GetGetter<TValue>(column);
            if (column.Name is "Semantic" or "Features" && getter is ValueGetter<VBuffer<float>> vector)
            {
                ValueGetter<VBuffer<float>> traced = (ref VBuffer<float> value) =>
                {
                    vector(ref value);
                    AppendIdentity(features);
                    AppendInt(features, value.Length);
                    AppendInt(features, value.GetValues().Length);
                    foreach (int item in value.GetIndices()) AppendInt(features, item);
                    foreach (float item in value.GetValues()) AppendInt(features, BitConverter.SingleToInt32Bits(item));
                    featureCalls++;
                };
                return (ValueGetter<TValue>)(Delegate)traced;
            }
            if (column.Name == "Label" && getter is ValueGetter<bool> label)
            {
                ValueGetter<bool> traced = (ref bool value) =>
                {
                    label(ref value);
                    AppendIdentity(labels);
                    AppendInt(labels, value ? 1 : 0);
                    labelCalls++;
                };
                return (ValueGetter<TValue>)(Delegate)traced;
            }
            return getter;
        }

        public override bool MoveNext()
        {
            if (disposed) return false;
            if (!source.MoveNext()) { complete = true; Dispose(); return false; }
            getId(ref currentId);
            AppendIdentity(order);
            rows++;
            return true;
        }

        private void AppendIdentity(IncrementalHash hash)
        {
            if (currentId.High != 0 || currentId.Low >= (ulong)owner.sourceIds.Length)
                throw new InvalidDataException("Diagnostic requires the declared selection-local stable DataViewRowId.");
            Span<byte> bytes = stackalloc byte[32];
            BinaryPrimitives.WriteUInt64LittleEndian(bytes, currentId.Low);
            BinaryPrimitives.WriteUInt64LittleEndian(bytes[8..], currentId.High);
            BinaryPrimitives.WriteInt64LittleEndian(bytes[16..], Position);
            BinaryPrimitives.WriteInt64LittleEndian(bytes[24..], owner.sourceIds[(int)currentId.Low]);
            hash.AppendData(bytes);
        }

        private static void AppendInt(IncrementalHash hash, int value)
        {
            Span<byte> bytes = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
            hash.AppendData(bytes);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && !disposed)
            {
                disposed = true;
                owner.Add(new(request, index, method, requestedCount, randomSupplied, active, rows, complete,
                    Convert.ToHexStringLower(order.GetHashAndReset()), featureCalls,
                    Convert.ToHexStringLower(features.GetHashAndReset()), labelCalls,
                    Convert.ToHexStringLower(labels.GetHashAndReset())));
                order.Dispose(); features.Dispose(); labels.Dispose(); source.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}

internal sealed class TracedEstimator(IEstimator<ITransformer> inner, Func<IDataView, IDataView> observe) : IEstimator<ITransformer>
{
    public SchemaShape GetOutputSchema(SchemaShape inputSchema) => inner.GetOutputSchema(inputSchema);
    public ITransformer Fit(IDataView input)
    {
        return inner.Fit(observe(input));
    }
}

namespace DecisionArrowPredictor;

public sealed class ProbabilityStore : IDisposable
{
    public const int BlockRows = 1024;
    public const long DefaultNumericCapBytes = 64L * 1024 * 1024;
    private float[][] semantic = [];
    private double[][] direct = [];
    private bool disposed;
    public int Count { get; }
    public long NumericCapacityBytes { get; private set; }
    public long NumericCapBytes { get; }

    public ProbabilityStore(int count, long numericCapBytes = DefaultNumericCapBytes)
    {
        if (count < 0 || numericCapBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Row count and numeric cap must be nonnegative.");
        Count = count;
        NumericCapBytes = numericCapBytes;
        long blocks = checked(((long)count + BlockRows - 1) / BlockRows);
        long required = checked((long)count * (FeatureContract.Width * sizeof(float) + sizeof(double)));
        RequireCapacity(required);
        try
        {
            semantic = new float[checked((int)blocks)][];
            direct = new double[checked((int)blocks)][];
            for (int i = 0; i < blocks; i++)
            {
                int capacity = Math.Min(BlockRows, checked(count - i * BlockRows));
                long reservation = checked(NumericCapacityBytes + capacity *
                    (FeatureContract.Width * sizeof(float) + sizeof(double)));
                RequireCapacity(reservation);
                semantic[i] = new float[checked(capacity * FeatureContract.Width)];
                direct[i] = new double[capacity];
                NumericCapacityBytes = checked(NumericCapacityBytes +
                    (long)semantic[i].Length * sizeof(float) + (long)direct[i].Length * sizeof(double));
                RequireCapacity(NumericCapacityBytes);
                if (NumericCapacityBytes != reservation)
                    throw new InvalidDataException("Numeric cache allocation differs from its reserved capacity.");
            }
            if (NumericCapacityBytes != required)
                throw new InvalidDataException("Numeric cache final capacity mismatch.");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private void RequireCapacity(long bytes)
    {
        if (bytes > NumericCapBytes)
            throw new InvalidDataException($"Compact numeric cache needs {bytes} bytes (including final block capacity), " +
                $"exceeding cap {NumericCapBytes}. Supply a larger explicit --numeric-cap-bytes or a smaller dataset; " +
                "no unlimited fallback is available. Text, metadata and trainer memory are separate.");
    }

    private void Check(int ordinal)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if ((uint)ordinal >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(ordinal));
    }

    internal Span<float> WritableSemantic(int ordinal)
    {
        Check(ordinal);
        return semantic[ordinal / BlockRows].AsSpan((ordinal % BlockRows) * FeatureContract.Width, FeatureContract.Width);
    }

    internal void SetDirect(int ordinal, double value)
    {
        Check(ordinal);
        RequireProbability(value);
        direct[ordinal / BlockRows][ordinal % BlockRows] = value;
    }

    internal void CopySemantic(int ordinal, Span<float> destination) => WritableSemantic(ordinal).CopyTo(destination);

    internal double Direct(int ordinal)
    {
        Check(ordinal);
        return direct[ordinal / BlockRows][ordinal % BlockRows];
    }

    internal static void RequireProbability(double value)
    {
        if (!double.IsFinite(value) || value < 0 || value > 1)
            throw new InvalidDataException("Expected a finite probability in [0,1]; no renormalization is permitted.");
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        semantic = [];
        direct = [];
    }
}

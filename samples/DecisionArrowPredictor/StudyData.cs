namespace DecisionArrowPredictor;

public sealed class StudyData : IDisposable
{
    private readonly object gate = new();
    private int cursors;
    private bool disposed;
    internal ProbabilityStore Probabilities { get; }
    public StudyMetadata Metadata { get; }
    public string FeatureFingerprint { get; }
    public string DatasetManifestSha256 { get; }
    public string SplitSha256 { get; }
    public string QuestionsSha256 { get; }
    public ExtractionCost? Extraction { get; }
    public long NumericCapacityBytes => Probabilities.NumericCapacityBytes;
    public long NumericCapBytes => Probabilities.NumericCapBytes;
    public int ActiveCursors { get { lock (gate) return cursors; } }

    internal StudyData(StudyMetadata metadata, ProbabilityStore probabilities, string identity,
        string datasetHash, string splitHash, string questionsHash, ExtractionCost? extraction)
    {
        if (metadata.Count != probabilities.Count || string.IsNullOrWhiteSpace(identity))
            throw new InvalidDataException("Completed compact data requires matching metadata/count/identity.");
        Metadata = metadata;
        Probabilities = probabilities;
        FeatureFingerprint = identity;
        DatasetManifestSha256 = datasetHash;
        SplitSha256 = splitHash;
        QuestionsSha256 = questionsHash;
        Extraction = extraction;
    }

    public static StudyData FromLegacy(ImportedStudy study, long numericCapBytes = ProbabilityStore.DefaultNumericCapBytes)
    {
        StudyWorkflow.Validate(study);
        var partitions = study.Split.Rows.ToDictionary(r => r.RowId, r => r.Split);
        var metadata = new StudyMetadata(study.Rows.Select(r =>
            new StudyRowMetadata(r.RowId, r.GroupId, r.Label, r.Text, partitions[r.RowId])));
        var store = new ProbabilityStore(metadata.Count, numericCapBytes);
        try
        {
            for (int i = 0; i < study.Rows.Length; i++)
            {
                study.Rows[i].Semantic.AsSpan().CopyTo(store.WritableSemantic(i));
                store.SetDirect(i, study.Rows[i].SpamBaseline);
            }
            return new(metadata, store, study.FeatureFingerprint, study.DatasetManifestSha256,
                study.SplitSha256, study.QuestionsSha256, study.Extraction);
        }
        catch
        {
            store.Dispose();
            throw;
        }
    }

    public RowSelection All()
    {
        RequireOpen();
        return new(this, Enumerable.Range(0, Metadata.Count));
    }

    public RowSelection Partition(string name)
    {
        RequireOpen();
        if (!SplitManifest.Names.Contains(name, StringComparer.Ordinal)) throw new ArgumentException("Unknown partition.", nameof(name));
        return new(this, Enumerable.Range(0, Metadata.Count).Where(o => Metadata[o].Partition == name));
    }

    internal void RequireOpen()
    {
        lock (gate) ObjectDisposedException.ThrowIf(disposed, this);
    }

    internal void AcquireCursor()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cursors = checked(cursors + 1);
        }
    }

    internal void ReleaseCursor()
    {
        lock (gate)
        {
            if (--cursors == 0 && disposed) Probabilities.Dispose();
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            if (cursors == 0) Probabilities.Dispose();
        }
    }
}

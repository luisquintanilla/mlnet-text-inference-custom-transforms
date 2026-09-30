namespace DecisionArrowPredictor;

public sealed class RowSelection
{
    private readonly int[] ordinals;
    internal StudyData Owner { get; }
    public int Count => ordinals.Length;
    public int this[int index] => ordinals[index];

    internal RowSelection(StudyData owner, IEnumerable<int> source)
    {
        owner.RequireOpen();
        Owner = owner;
        ordinals = source.ToArray();
        if (ordinals.Any(o => (uint)o >= (uint)owner.Metadata.Count) || ordinals.Distinct().Count() != ordinals.Length)
            throw new InvalidDataException("Selection requires unique valid canonical ordinals.");
    }

    public RowSelection Subset(int target)
    {
        Owner.RequireOpen();
        if (target <= 0 || !ordinals.Any(o => Owner.Metadata[o].Label) ||
            !ordinals.Any(o => !Owner.Metadata[o].Label))
            throw new InvalidDataException("Learning subsets need a positive target, unique rows, and both classes.");
        if (target >= Count) return Sorted(ordinals);
        var result = new List<int>();
        bool ham = false, spam = false;
        foreach (var group in ordinals.GroupBy(o => Owner.Metadata[o].GroupId)
            .OrderBy(g => SplitManifest.OrderKey(g.Key, PredictorTraining.Seed), StringComparer.Ordinal))
        {
            foreach (int ordinal in group)
            {
                result.Add(ordinal);
                spam |= Owner.Metadata[ordinal].Label;
                ham |= !Owner.Metadata[ordinal].Label;
            }
            if (result.Count >= target && ham && spam) break;
        }
        return Sorted(result);
    }

    private RowSelection Sorted(IEnumerable<int> source) =>
        new(Owner, source.OrderBy(o => Owner.Metadata[o].RowId));

    public long[] SourceIds() => ordinals.Select(o => Owner.Metadata[o].RowId).ToArray();
    public double Prevalence() => Count == 0 ? throw new InvalidDataException("Empty prevalence selection.") :
        (double)ordinals.Count(o => Owner.Metadata[o].Label) / Count;
    public StudyDataView View() => new(this);
    public StudyDataView View(Microsoft.ML.MLContext context) => new(this, context);
    public StudyDataView PartitionedPredictionControlView() => new(this, partitionedPredictionControl: true);
}

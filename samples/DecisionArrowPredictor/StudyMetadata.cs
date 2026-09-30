namespace DecisionArrowPredictor;

public readonly record struct StudyRowMetadata(long RowId, long GroupId, bool Label, string Text, string Partition);

public sealed class StudyMetadata
{
    private readonly StudyRowMetadata[] rows;
    private readonly Dictionary<long, int> ordinals;
    public int Count => rows.Length;
    public StudyRowMetadata this[int ordinal] => rows[ordinal];

    public StudyMetadata(IEnumerable<StudyRowMetadata> source)
    {
        rows = source.ToArray();
        ordinals = new(rows.Length);
        for (int i = 0; i < rows.Length; i++)
        {
            var row = rows[i];
            if (row.Text is null || !SplitManifest.Names.Contains(row.Partition, StringComparer.Ordinal) ||
                !ordinals.TryAdd(row.RowId, i))
                throw new InvalidDataException("Metadata requires unique source IDs, original text and a declared partition.");
        }
        if (rows.GroupBy(r => r.GroupId).Any(g => g.Select(r => r.Partition).Distinct().Count() != 1))
            throw new InvalidDataException("A metadata group crosses partitions.");
    }

    public int Ordinal(long rowId) => ordinals.TryGetValue(rowId, out int ordinal) ? ordinal :
        throw new InvalidDataException($"Arrow source ID {rowId} is not in the frozen metadata.");
}

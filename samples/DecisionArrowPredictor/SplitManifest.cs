using System.Security.Cryptography;
using System.Text;

namespace DecisionArrowPredictor;

public sealed record SplitRow(long RowId, long GroupId, bool Label, string Split);
public sealed record PartitionCount(string Split, int Rows, int Ham, int Spam, int Groups);
public sealed record SplitManifest(int Version, int Seed, string CorpusSha256, string QuestionsSha256,
    string GroupDiagnosticsSha256, string Algorithm, PartitionCount[] Counts, SplitRow[] Rows)
{
    public const int SplitSeed = 1729;
    public static readonly string[] Names = ["train", "validation", "holdout"];

    public static SplitManifest Create(GroupDiagnostics groups, string corpusHash, string questionsHash, string groupsHash,
        IReadOnlyList<CorpusRow> source)
    {
        var byId = source.ToDictionary(r => r.RowId);
        if (groups.Rows != source.Count || !groups.Membership.SelectMany(g => g.RowIds).Order().SequenceEqual(byId.Keys.Order()))
            throw new InvalidDataException("Group membership must cover each source ID exactly once.");
        double[] fractions = [0.6, 0.2, 0.2];
        int totalSpam = source.Count(r => r.Label), totalHam = source.Count - totalSpam;
        var counts = new int[3, 2];
        var assignments = new List<SplitRow>();
        var ordered = groups.Membership.OrderByDescending(g => g.RowIds.Length)
            .ThenBy(g => OrderKey(g.GroupId, SplitSeed), StringComparer.Ordinal).ThenBy(g => g.GroupId);
        foreach (var group in ordered)
        {
            int[] addition = [group.Ham, group.Spam];
            double Cost(int split)
            {
                double cost = 0;
                for (int s = 0; s < 3; s++)
                    for (int c = 0; c < 2; c++)
                    {
                        double target = fractions[s] * (c == 0 ? totalHam : totalSpam);
                        double delta = counts[s, c] + (s == split ? addition[c] : 0) - target;
                        cost += delta * delta / Math.Max(1, target);
                    }
                return cost;
            }
            int selected = Enumerable.Range(0, 3).OrderBy(Cost).ThenBy(s => s).First();
            counts[selected, 0] += group.Ham; counts[selected, 1] += group.Spam;
            assignments.AddRange(group.RowIds.Select(id => new SplitRow(id, group.GroupId, byId[id].Label, Names[selected])));
        }
        var rows = assignments.OrderBy(r => r.RowId).ToArray();
        var summary = Names.Select(n =>
        {
            var partition = rows.Where(r => r.Split == n).ToArray();
            return new PartitionCount(n, partition.Length, partition.Count(r => !r.Label),
                partition.Count(r => r.Label), partition.Select(r => r.GroupId).Distinct().Count());
        }).ToArray();
        if (summary.Any(c => c.Ham == 0 || c.Spam == 0))
            throw new InvalidDataException("Grouped partition lacks both classes; do not split groups to force counts.");
        return new(1, SplitSeed, corpusHash, questionsHash, groupsHash,
            "size-desc/sha256-seed-id-tie/greedy-squared-class-deficit-60-20-20-v1", summary, rows);
    }

    public static string OrderKey(long id, int seed) =>
        Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(
            FormattableString.Invariant($"{seed}:{id}"))));

    public void Validate(IReadOnlyList<CorpusRow> source)
    {
        if (Version != 1 || Seed != SplitSeed || Rows.Length != source.Count ||
            Rows.Select(r => r.RowId).Distinct().Count() != Rows.Length ||
            !Rows.Select(r => r.RowId).Order().SequenceEqual(source.Select(r => r.RowId).Order()))
            throw new InvalidDataException("Invalid split version/seed/source IDs.");
        var labels = source.ToDictionary(r => r.RowId, r => r.Label);
        if (Rows.Any(r => !Names.Contains(r.Split, StringComparer.Ordinal) || labels[r.RowId] != r.Label) ||
            Rows.GroupBy(r => r.GroupId).Any(g => g.Select(r => r.Split).Distinct().Count() != 1))
            throw new InvalidDataException("Labels changed or a duplicate group crosses partitions.");
        foreach (var name in Names)
        {
            var actual = Rows.Where(r => r.Split == name).ToArray();
            var expected = new PartitionCount(name, actual.Length, actual.Count(r => !r.Label),
                actual.Count(r => r.Label), actual.Select(r => r.GroupId).Distinct().Count());
            if (expected.Ham == 0 || expected.Spam == 0 || Counts.Single(c => c.Split == name) != expected)
                throw new InvalidDataException($"Invalid partition counts: {name}.");
        }
    }
}

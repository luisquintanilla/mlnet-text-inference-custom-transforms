using System.Text.RegularExpressions;

namespace DecisionArrowPredictor;

public sealed record DuplicateGroup(long GroupId, long[] RowIds, int Ham, int Spam, bool Chained);
public sealed record GroupDiagnostics(string Algorithm, double Threshold, int Rows, int Groups,
    int LargestGroup, int MixedLabelGroups, int ChainedGroups, DuplicateGroup[] Membership);

public static partial class DuplicateGrouping
{
    public const double Threshold = 0.90;
    public const string Algorithm = "lower-invariant-whitespace/url-digit4-template/normalized-char5-jaccard90/components-source-id-v1";

    [GeneratedRegex(@"\s+", RegexOptions.CultureInvariant)]
    private static partial Regex Whitespace();
    [GeneratedRegex(@"(?:https?://|www\.)\S+", RegexOptions.CultureInvariant)]
    private static partial Regex Url();
    [GeneratedRegex(@"\d{4,}", RegexOptions.CultureInvariant)]
    private static partial Regex Digits();

    public static string Normalize(string text) => Whitespace().Replace(text.ToLowerInvariant(), " ").Trim();
    public static string Template(string normalized) => Digits().Replace(Url().Replace(normalized, "<url>"), "<digits>");
    public static HashSet<string> Grams(string normalized) =>
        Enumerable.Range(0, Math.Max(0, normalized.Length - 4))
            .Select(i => normalized.Substring(i, 5)).ToHashSet(StringComparer.Ordinal);

    public static double Jaccard(HashSet<string> a, HashSet<string> b)
    {
        int intersection = a.Count(b.Contains);
        int union = a.Count + b.Count - intersection;
        return union == 0 ? 1 : (double)intersection / union;
    }

    public static GroupDiagnostics Build(IReadOnlyList<CorpusRow> source)
    {
        var rows = source.OrderBy(r => r.RowId).ToArray();
        if (rows.Length == 0 || rows.Select(r => r.RowId).Distinct().Count() != rows.Length)
            throw new InvalidDataException("Grouping requires nonempty unique source IDs.");
        var normalized = rows.Select(r => Normalize(r.Text)).ToArray();
        var templates = normalized.Select(Template).ToArray();
        var grams = normalized.Select(n => n.Length >= 20 ? Grams(n) : null).ToArray();
        var parent = Enumerable.Range(0, rows.Length).ToArray();
        int Root(int n)
        {
            while (parent[n] != n) { parent[n] = parent[parent[n]]; n = parent[n]; }
            return n;
        }
        void Union(int a, int b)
        {
            a = Root(a); b = Root(b);
            parent[Math.Max(a, b)] = Math.Min(a, b);
        }
        bool Direct(int a, int b) =>
            normalized[a] == normalized[b] || templates[a] == templates[b] ||
            (grams[a] is { } x && grams[b] is { } y &&
             Math.Min(x.Count, y.Count) >= Threshold * Math.Max(x.Count, y.Count) &&
             Jaccard(x, y) >= Threshold);

        // The size bound is exact for set Jaccard; it prunes without changing membership.
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < i; j++)
                if (Direct(i, j)) Union(i, j);
        var groups = Enumerable.Range(0, rows.Length).GroupBy(Root).Select(g =>
        {
            var indices = g.ToArray();
            bool chained = indices.Any(i => indices.Any(j => j < i && !Direct(i, j)));
            int spam = indices.Count(i => rows[i].Label);
            return new DuplicateGroup(rows[indices[0]].RowId, indices.Select(i => rows[i].RowId).ToArray(),
                indices.Length - spam, spam, chained);
        }).OrderBy(g => g.GroupId).ToArray();
        return new(Algorithm, Threshold, rows.Length, groups.Length, groups.Max(g => g.RowIds.Length),
            groups.Count(g => g.Ham > 0 && g.Spam > 0), groups.Count(g => g.Chained), groups);
    }
}

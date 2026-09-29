using System.Globalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class DuplicateGroupingTests
{
    [TestMethod]
    [DataRow("en-US")]
    [DataRow("tr-TR")]
    public void Normalize_InvariantComparisonPreservesCorpusText(string culture)
    {
        var originalCulture = CultureInfo.CurrentCulture;
        try
        {
            // CurrentCulture is execution-context local; never set DefaultThreadCurrentCulture.
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(culture);
            CorpusRow[] rows = [new(8, false, " \tI  FOO\r\nBar "), new(2, true, "i foo bar")];
            var snapshot = rows.Select(r => r.Text).ToArray();

            Assert.AreEqual("i foo bar", DuplicateGrouping.Normalize(rows[0].Text));
            var groups = DuplicateGrouping.Build(rows);

            Assert.AreEqual(1, groups.Groups);
            Assert.AreEqual(2L, groups.Membership[0].GroupId);
            CollectionAssert.AreEqual(new long[] { 2, 8 }, groups.Membership[0].RowIds);
            Assert.AreEqual(1, groups.Membership[0].Ham);
            Assert.AreEqual(1, groups.Membership[0].Spam);
            CollectionAssert.AreEqual(snapshot, rows.Select(r => r.Text).ToArray());
        }
        finally
        {
            CultureInfo.CurrentCulture = originalCulture;
        }
        Assert.AreSame(originalCulture, CultureInfo.CurrentCulture);
    }

    [TestMethod]
    public void Template_UrlAndFourDigitRunsMergeButThreeDigitsDoNot()
    {
        string first = DuplicateGrouping.Normalize("Call 1234 at https://a.example/x");
        string second = DuplicateGrouping.Normalize("Call 98765 at www.b.example/y");

        Assert.AreEqual("call <digits> at <url>", DuplicateGrouping.Template(first));
        Assert.AreEqual("call <digits> at <url>", DuplicateGrouping.Template(second));
        Assert.AreEqual("id<digits>", DuplicateGrouping.Template("id1234"));
        Assert.AreEqual("id<digits>", DuplicateGrouping.Template("id9876"));
        Assert.AreEqual("id123", DuplicateGrouping.Template("id123"));
        Assert.AreEqual("id987", DuplicateGrouping.Template("id987"));
        var urlGroups = DuplicateGrouping.Build([new(1, false, first), new(2, true, second)]);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, urlGroups.Membership.Single().RowIds);
        var longDigits = DuplicateGrouping.Build([new(1, false, "id1234"), new(2, true, "id9876")]);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, longDigits.Membership.Single().RowIds);
        var shortDigits = DuplicateGrouping.Build([new(1, false, "id123"), new(2, true, "id987")]);
        Assert.AreEqual(2, shortDigits.Groups);
        Assert.AreEqual(0, shortDigits.MixedLabelGroups);
    }

    [TestMethod]
    public void Template_UnicodeDigitRunsAtLeastFourAreReplaced()
    {
        Assert.AreEqual("id<digits>", DuplicateGrouping.Template("id١٢٣٤"));
        Assert.AreEqual("id<digits>", DuplicateGrouping.Template("id１２３４５"));
        Assert.AreEqual("id١٢٣", DuplicateGrouping.Template("id١٢٣"));

        var groups = DuplicateGrouping.Build([new(7, false, "id١٢٣٤"), new(3, true, "id１２３４５")]);

        CollectionAssert.AreEqual(new long[] { 3, 7 }, groups.Membership.Single().RowIds);
        Assert.AreEqual(1, groups.MixedLabelGroups);
        Assert.IsFalse(groups.Membership[0].Chained);
    }

    [TestMethod]
    public void Grams_DistinctUtf16Char5SetsAndEmptyJaccard()
    {
        Assert.AreEqual(0, DuplicateGrouping.Grams("abcd").Count);
        CollectionAssert.AreEquivalent(new[] { "abcde" }, DuplicateGrouping.Grams("abcde").ToArray());
        CollectionAssert.AreEquivalent(new[] { "aaaaa" }, DuplicateGrouping.Grams("aaaaaaaa").ToArray());
        // Five UTF16 code units, including a surrogate pair, not five Unicode scalars.
        CollectionAssert.AreEquivalent(new[] { "a😀bc", "😀bcd" }, DuplicateGrouping.Grams("a😀bcd").ToArray());
        var empty = DuplicateGrouping.Grams("");
        var nonempty = DuplicateGrouping.Grams("abcde");

        Assert.AreEqual(1d, DuplicateGrouping.Jaccard(empty, new HashSet<string>()));
        Assert.AreEqual(0d, DuplicateGrouping.Jaccard(empty, nonempty));
        Assert.AreEqual(0d, DuplicateGrouping.Jaccard(nonempty, empty));
        CollectionAssert.AreEquivalent(new[] { "abcde" }, nonempty.ToArray());
        Assert.AreEqual(0, empty.Count);
    }

    [TestMethod]
    public void Jaccard_InclusivePointNineEdgeAndBelowThreshold()
    {
        const string a = "abcdefghijklmnopqrstuvw";
        const string boundary = "abcdefghijklmnopqrstuvx";
        const string below = "abcdefghijklmnopqrstuxy";
        var x = DuplicateGrouping.Grams(a);
        var y = DuplicateGrouping.Grams(boundary);
        var z = DuplicateGrouping.Grams(below);

        Assert.AreEqual(19, x.Count);
        Assert.AreEqual(19, y.Count);
        Assert.AreEqual(18, x.Intersect(y).Count());
        Assert.AreEqual(20, x.Union(y).Count());
        Assert.IsTrue(x.Contains("stuvw"));
        Assert.IsTrue(y.Contains("stuvx"));
        Assert.AreEqual(.9, DuplicateGrouping.Jaccard(x, y));
        Assert.AreEqual(17d / 21, DuplicateGrouping.Jaccard(x, z), 1e-12);
        var together = DuplicateGrouping.Build([new(1, false, a), new(2, true, boundary)]);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, together.Membership.Single().RowIds);
        var separate = DuplicateGrouping.Build([new(1, false, a), new(2, true, below)]);
        Assert.AreEqual(2, separate.Groups);
        Assert.AreEqual(.90, together.Threshold);
        Assert.IsFalse(together.Membership[0].Chained);
        // Unequal set sizes hit the separate pruning bound at equality, too.
        const string subsetText = "abcdefghijklmnopqrstuv";
        const string supersetText = "abcdefghijklmnopqrstuvwx";
        var subset = DuplicateGrouping.Grams(subsetText);
        var superset = DuplicateGrouping.Grams(supersetText);
        Assert.AreEqual(18, subset.Count);
        Assert.AreEqual(20, superset.Count);
        Assert.AreEqual(.9, DuplicateGrouping.Jaccard(subset, superset));
        var sizeBoundary = DuplicateGrouping.Build([new(7, false, subsetText), new(9, true, supersetText)]);
        CollectionAssert.AreEqual(new long[] { 7, 9 }, sizeBoundary.Membership.Single().RowIds);
        Assert.AreEqual(1, sizeBoundary.MixedLabelGroups);
    }

    [TestMethod]
    [DataRow("both-nineteen", "abababababababababa", "bababababababababab", 2)]
    [DataRow("both-twenty", "abababababababababab", "babababababababababa", 1)]
    [DataRow("mixed-lengths", "abababababababababa", "babababababababababa", 2)]
    [DataRow("raw-padding-does-not-enable-fuzzy", " abababababababababa ", "babababababababababa", 2)]
    public void Build_FuzzyEdgesRequireBothLengthsAtLeastTwenty(string caseId, string a, string b, int expectedGroups)
    {
        string normalizedA = DuplicateGrouping.Normalize(a), normalizedB = DuplicateGrouping.Normalize(b);
        Assert.AreEqual(1d, DuplicateGrouping.Jaccard(DuplicateGrouping.Grams(normalizedA), DuplicateGrouping.Grams(normalizedB)));
        Assert.AreNotEqual(DuplicateGrouping.Template(normalizedA), DuplicateGrouping.Template(normalizedB));
        CorpusRow[] rows = [new(1, false, a), new(2, true, b)];

        var actual = DuplicateGrouping.Build(rows);

        Assert.AreEqual(expectedGroups, actual.Groups, caseId);
        Assert.AreEqual(expectedGroups == 1 ? 1 : 0, actual.MixedLabelGroups);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, actual.Membership.SelectMany(g => g.RowIds).Order().ToArray());
        CollectionAssert.AreEqual(new[] { a, b }, rows.Select(r => r.Text).ToArray());
    }

    [TestMethod]
    public void Build_ShortMessagesOnlyUseExactOrTemplateEdges()
    {
        CorpusRow[] rows =
        [
            new(1, false, " Hi  THERE "), new(2, true, "hi there"),
            new(3, false, "id1234"), new(4, true, "id9876"),
            new(5, false, "abababababababababa"), new(6, true, "bababababababababab")
        ];
        var snapshot = rows.ToArray();

        var actual = DuplicateGrouping.Build(rows);

        Assert.AreEqual(4, actual.Groups);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, actual.Membership[0].RowIds);
        CollectionAssert.AreEqual(new long[] { 3, 4 }, actual.Membership[1].RowIds);
        CollectionAssert.AreEqual(new long[] { 5 }, actual.Membership[2].RowIds);
        CollectionAssert.AreEqual(new long[] { 6 }, actual.Membership[3].RowIds);
        Assert.AreEqual(2, actual.MixedLabelGroups);
        Assert.AreEqual(0, actual.ChainedGroups);
        CollectionAssert.AreEqual(snapshot, rows);
    }

    [TestMethod]
    public void Build_TransitiveChainHasMinimumIdOrderedMembershipAndMixedLabels()
    {
        var rows = CorpusFixtures.Chain();
        var a = DuplicateGrouping.Grams(rows[0].Text);
        var b = DuplicateGrouping.Grams(rows[1].Text);
        var c = DuplicateGrouping.Grams(rows[2].Text);
        Assert.AreEqual(116, b.Count);
        Assert.AreEqual(111, a.Intersect(b).Count());
        Assert.AreEqual(121, a.Union(b).Count());
        Assert.AreEqual(111d / 121, DuplicateGrouping.Jaccard(a, b), 1e-12);
        Assert.AreEqual(111d / 121, DuplicateGrouping.Jaccard(b, c), 1e-12);
        Assert.AreEqual(106d / 126, DuplicateGrouping.Jaccard(a, c), 1e-12);

        var actual = DuplicateGrouping.Build(rows);

        Assert.AreEqual("lower-invariant-whitespace/url-digit4-template/normalized-char5-jaccard90/components-source-id-v1", actual.Algorithm);
        Assert.AreEqual(.9, actual.Threshold);
        Assert.AreEqual(3, actual.Rows);
        Assert.AreEqual(1, actual.Groups);
        Assert.AreEqual(3, actual.LargestGroup);
        Assert.AreEqual(1, actual.MixedLabelGroups);
        Assert.AreEqual(1, actual.ChainedGroups);
        var group = actual.Membership.Single();
        Assert.AreEqual(10L, group.GroupId);
        CollectionAssert.AreEqual(new long[] { 10, 20, 30 }, group.RowIds);
        Assert.AreEqual(2, group.Ham);
        Assert.AreEqual(1, group.Spam);
        Assert.IsTrue(group.Chained);
        CollectionAssert.AreEqual(new[] { false, true, false }, rows.Select(r => r.Label).ToArray());
    }

    [TestMethod]
    public void Build_InputPermutationPreservesAllDiagnostics()
    {
        CorpusRow[] rows = [.. CorpusFixtures.Chain(), new(50, false, "standalone")];
        var first = DuplicateGrouping.Build(rows);

        var second = DuplicateGrouping.Build([rows[3], rows[2], rows[0], rows[1]]);

        Assert.AreEqual(4, first.Rows);
        Assert.AreEqual(2, first.Groups);
        Assert.AreEqual(first.Algorithm, second.Algorithm);
        Assert.AreEqual(first.Threshold, second.Threshold);
        Assert.AreEqual(first.Rows, second.Rows);
        Assert.AreEqual(first.Groups, second.Groups);
        Assert.AreEqual(first.LargestGroup, second.LargestGroup);
        Assert.AreEqual(first.MixedLabelGroups, second.MixedLabelGroups);
        Assert.AreEqual(first.ChainedGroups, second.ChainedGroups);
        Assert.AreEqual(first.Membership.Length, second.Membership.Length);
        for (int i = 0; i < first.Membership.Length; i++)
        {
            var expected = first.Membership[i];
            var actual = second.Membership[i];
            Assert.AreEqual(expected.GroupId, actual.GroupId);
            CollectionAssert.AreEqual(expected.RowIds, actual.RowIds);
            Assert.AreEqual(expected.Ham, actual.Ham);
            Assert.AreEqual(expected.Spam, actual.Spam);
            Assert.AreEqual(expected.Chained, actual.Chained);
        }
        CollectionAssert.AreEqual(new long[] { 10, 50 }, second.Membership.Select(g => g.GroupId).ToArray());
    }

    [TestMethod]
    [DataRow("empty")]
    [DataRow("duplicate-id")]
    public void Build_EmptyOrDuplicateSourceIdsAreRejected(string caseId)
    {
        CorpusRow[] rows = caseId == "empty" ? [] : [new(1, false, "original ham"), new(1, true, "different spam")];
        var snapshot = rows.ToArray();

        var error = Assert.ThrowsExactly<InvalidDataException>(() => DuplicateGrouping.Build(rows));

        Assert.AreEqual("Grouping requires nonempty unique source IDs.", error.Message);
        CollectionAssert.AreEqual(snapshot, rows);
        Assert.AreEqual(caseId == "empty" ? 0 : 2, rows.Length);
    }
}

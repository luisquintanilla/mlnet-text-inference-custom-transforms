using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class SplitManifestTests
{
    [TestMethod]
    [DataRow(1L, 1729)]
    [DataRow(59L, 1729)]
    [DataRow(-27L, 42)]
    [DataRow(9223372036854775807L, -1)]
    public void OrderKey_MatchesIndependentInvariantSeededSha256(long id, int seed)
    {
        string expected = SplitExpectations.OrderKey(id, seed);

        string actual = SplitManifest.OrderKey(id, seed);

        Assert.AreEqual(expected, actual);
        Assert.AreEqual(64, actual.Length);
        Assert.AreEqual(actual.ToLowerInvariant(), actual);
        Assert.AreNotEqual(SplitExpectations.OrderKey(id, seed + 1), actual);
    }

    [TestMethod]
    public void Create_BalancedMixedGroupsProduceDeterministicThirtySixTwelveTwelveRows()
    {
        var source = CorpusFixtures.Balanced();
        var snapshot = source.ToArray();
        var groups = DuplicateGrouping.Build(source);

        var manifest = SplitExpectations.Create(source, groups);
        var reordered = SplitExpectations.Create(source.Reverse().ToArray(),
            groups with { Membership = groups.Membership.Reverse().ToArray() });

        Assert.AreEqual(30, groups.Groups);
        Assert.AreEqual(30, groups.MixedLabelGroups);
        Assert.AreEqual(0, groups.ChainedGroups);
        SplitExpectations.Complete(manifest, source, groups);
        CollectionAssert.AreEqual(new[]
        {
            new PartitionCount("train", 36, 18, 18, 18),
            new PartitionCount("validation", 12, 6, 6, 6),
            new PartitionCount("holdout", 12, 6, 6, 6)
        }, manifest.Counts);
        AssertGroupIds(manifest, "train", [1, 3, 7, 11, 15, 21, 23, 27, 29, 31, 33, 35, 37, 47, 51, 55, 57, 59]);
        AssertGroupIds(manifest, "validation", [9, 13, 25, 41, 45, 53]);
        AssertGroupIds(manifest, "holdout", [5, 17, 19, 39, 43, 49]);
        // The second hash-ordered balanced group has equal candidate costs:
        // train wins by partition index, not by source ID or arbitrary enumeration.
        Assert.AreEqual("train", manifest.Rows.Single(r => r.RowId == 21).Split);
        SplitExpectations.Equal(manifest, reordered);
        CollectionAssert.AreEqual(snapshot, source);
    }

    [TestMethod]
    public void Create_UnbalancedGroupsPreserveRealRowsLabelsAndClassSupport()
    {
        var source = CorpusFixtures.Unbalanced();
        var groups = DuplicateGrouping.Build(source);
        long[] order = groups.Membership.OrderByDescending(g => g.RowIds.Length)
            .ThenBy(g => SplitExpectations.OrderKey(g.GroupId, 1729), StringComparer.Ordinal)
            .ThenBy(g => g.GroupId).Select(g => g.GroupId).ToArray();
        CollectionAssert.AreEqual(new long[] { 1, 5, 21, 13, 19, 10, 8, 9, 14, 23, 12, 24, 20, 15, 17, 22, 11, 25, 16, 18 }, order);

        var actual = SplitExpectations.Create(source, groups);

        SplitExpectations.Complete(actual, source, groups);
        CollectionAssert.AreEqual(new[]
        {
            new PartitionCount("train", 14, 7, 7, 9),
            new PartitionCount("validation", 6, 3, 3, 6),
            new PartitionCount("holdout", 5, 3, 2, 5)
        }, actual.Counts);
        AssertGroupIds(actual, "train", [1, 5, 11, 14, 15, 17, 19, 20, 22]);
        AssertGroupIds(actual, "validation", [9, 10, 12, 13, 16, 25]);
        AssertGroupIds(actual, "holdout", [8, 18, 21, 23, 24]);
        Assert.AreEqual(13, actual.Counts.Sum(c => c.Ham));
        Assert.AreEqual(12, actual.Counts.Sum(c => c.Spam));
        // Independent objective checkpoints, with fixed actual class totals 13/12.
        CollectionAssert.AreEqual(new[] { 4, 3, 1, 1, 1 }, order.Take(5)
            .Select(id => groups.Membership.Single(g => g.GroupId == id).RowIds.Length).ToArray());
        var emptyCosts = Costs(new int[3, 2], 3, 1);
        EvaluationGoldens.Near(18.292735042735043, emptyCosts[0]);
        EvaluationGoldens.Near(20.87820512820513, emptyCosts[1]);
        EvaluationGoldens.Near(20.87820512820513, emptyCosts[2]);
        var secondCosts = Costs(new int[,] { { 3, 1 }, { 0, 0 }, { 0, 0 } }, 1, 2);
        EvaluationGoldens.Near(14.301282051282051, secondCosts[0]);
        EvaluationGoldens.Near(14.344017094017094, secondCosts[1]);
        Assert.IsTrue(secondCosts[0] < secondCosts[1]);
        Assert.AreEqual("train", actual.Rows.Single(r => r.RowId == 5).Split);
        SplitExpectations.Equal(actual, SplitExpectations.Create(source.Reverse().ToArray(),
            groups with { Membership = groups.Membership.Reverse().ToArray() }));
    }

    [TestMethod]
    public void Create_TwoBalancedGroupsCannotPopulateAllPartitions()
    {
        CorpusRow[] rows =
        [
            new(1, false, "pair-one"), new(2, true, "pair-one"),
            new(3, false, "pair-two"), new(4, true, "pair-two")
        ];
        var groups = DuplicateGrouping.Build(rows);
        var snapshot = rows.ToArray();

        var error = Assert.ThrowsExactly<InvalidDataException>(() => SplitExpectations.Create(rows, groups));

        Assert.AreEqual("Grouped partition lacks both classes; do not split groups to force counts.", error.Message);
        Assert.AreEqual(2, groups.Groups);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, groups.Membership[0].RowIds);
        CollectionAssert.AreEqual(new long[] { 3, 4 }, groups.Membership[1].RowIds);
        CollectionAssert.AreEqual(snapshot, rows);
    }

    [TestMethod]
    [DataRow("missing")]
    [DataRow("extra")]
    [DataRow("duplicate-within")]
    [DataRow("duplicate-across")]
    [DataRow("diagnostic-row-count")]
    public void Create_InvalidMembershipCoverageIsRejected(string mutation)
    {
        var source = CorpusFixtures.Balanced();
        var original = DuplicateGrouping.Build(source);
        var membership = original.Membership.ToArray();
        var changed = mutation switch
        {
            "missing" => membership[0] with { RowIds = [membership[0].RowIds[0]] },
            "extra" => membership[0] with { RowIds = [.. membership[0].RowIds, 999] },
            "duplicate-within" => membership[0] with { RowIds = [1, 1] },
            "duplicate-across" => membership[0] with { RowIds = [1, 3] },
            "diagnostic-row-count" => membership[0],
            _ => throw new ArgumentOutOfRangeException(nameof(mutation))
        };
        membership[0] = changed;
        var invalid = original with { Membership = membership, Rows = mutation == "diagnostic-row-count" ? 59 : 60 };

        var error = Assert.ThrowsExactly<InvalidDataException>(() => SplitExpectations.Create(source, invalid));

        Assert.AreEqual("Group membership must cover each source ID exactly once.", error.Message);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, original.Membership[0].RowIds);
        Assert.AreEqual(60, source.Length);
        CollectionAssert.AreEqual(Enumerable.Range(1, 60).Select(i => (long)i).ToArray(), source.Select(r => r.RowId).ToArray());
    }

    [TestMethod]
    public void Create_DuplicateSourceIdsThrowArgumentException()
    {
        var source = CorpusFixtures.Balanced();
        var groups = DuplicateGrouping.Build(source);
        source[1] = source[1] with { RowId = 1 };
        var snapshot = source.ToArray();

        Assert.ThrowsExactly<ArgumentException>(() => SplitExpectations.Create(source, groups));

        CollectionAssert.AreEqual(snapshot, source);
        Assert.AreEqual(60, groups.Rows);
        CollectionAssert.AreEqual(new long[] { 1, 2 }, groups.Membership[0].RowIds);
    }

    [TestMethod]
    [DataRow("version")]
    [DataRow("seed")]
    [DataRow("duplicate-row")]
    [DataRow("missing-row")]
    [DataRow("extra-row")]
    [DataRow("label")]
    [DataRow("split-case")]
    [DataRow("split-unknown")]
    [DataRow("absent-class")]
    [DataRow("count-rows")]
    [DataRow("count-ham")]
    [DataRow("count-spam")]
    [DataRow("count-groups")]
    public void Validate_TamperedManifestFieldsAreRejected(string mutation)
    {
        var source = CorpusFixtures.Balanced();
        var valid = SplitExpectations.Create(source);
        var rows = valid.Rows.ToArray();
        var counts = valid.Counts.ToArray();
        var invalid = valid;
        switch (mutation)
        {
            case "version": invalid = valid with { Version = 2 }; break;
            case "seed": invalid = valid with { Seed = 1730 }; break;
            case "duplicate-row": rows[1] = rows[1] with { RowId = rows[0].RowId }; break;
            case "missing-row": rows = rows[..^1]; break;
            case "extra-row": rows = [.. rows, new(999, 999, false, "train")]; break;
            case "label": rows[0] = rows[0] with { Label = !rows[0].Label }; break;
            case "split-case": rows[0] = rows[0] with { Split = "Train" }; break;
            case "split-unknown": rows[0] = rows[0] with { Split = "test" }; break;
            case "absent-class":
                rows = rows.Select(r => r.Split == "train" ? r with { Split = "validation" } : r).ToArray();
                counts = SplitExpectations.Counts(rows);
                break;
            case "count-rows": counts[0] = counts[0] with { Rows = 37 }; break;
            case "count-ham": counts[0] = counts[0] with { Ham = 19 }; break;
            case "count-spam": counts[0] = counts[0] with { Spam = 19 }; break;
            case "count-groups": counts[0] = counts[0] with { Groups = 19 }; break;
            default: throw new ArgumentOutOfRangeException(nameof(mutation));
        }
        invalid = invalid with { Rows = rows, Counts = counts };

        var error = Assert.ThrowsExactly<InvalidDataException>(() => invalid.Validate(source));

        Assert.AreEqual(mutation is "version" or "seed" or "duplicate-row" or "missing-row" or "extra-row"
            ? "Invalid split version/seed/source IDs."
            : mutation is "label" or "split-case" or "split-unknown"
                ? "Labels changed or a duplicate group crosses partitions." : "Invalid partition counts: train.",
            error.Message);
        Assert.AreEqual(60, valid.Rows.Length);
        Assert.AreEqual(36, valid.Counts[0].Rows);
        valid.Validate(source);
        Assert.AreEqual(source[0].Label, valid.Rows[0].Label);
    }

    [TestMethod]
    public void Validate_DeclaredGroupCannotSpanPartitions()
    {
        var source = CorpusFixtures.Balanced();
        var valid = SplitExpectations.Create(source);
        var rows = valid.Rows.ToArray();
        int index = Array.FindIndex(rows, r => r.GroupId == 1);
        rows[index] = rows[index] with { Split = "validation" };
        var invalid = valid with { Rows = rows, Counts = SplitExpectations.Counts(rows) };

        var error = Assert.ThrowsExactly<InvalidDataException>(() => invalid.Validate(source));

        Assert.AreEqual("Labels changed or a duplicate group crosses partitions.", error.Message);
        Assert.AreEqual(2, rows.Where(r => r.GroupId == 1).Select(r => r.Split).Distinct().Count());
        Assert.AreEqual(1, valid.Rows.Where(r => r.GroupId == 1).Select(r => r.Split).Distinct().Count());
        CollectionAssert.AreEqual(source.Select(r => r.Label).ToArray(), rows.Select(r => r.Label).ToArray());
        valid.Validate(source);
    }

    [TestMethod]
    [DataRow("missing-count")]
    [DataRow("duplicate-count")]
    public void Validate_CountMultiplicityRejectionUsesCurrentExactExceptionType(string caseId)
    {
        var source = CorpusFixtures.Balanced();
        var valid = SplitExpectations.Create(source);
        var invalid = valid with
        {
            Counts = caseId == "missing-count" ? valid.Counts[1..] : [.. valid.Counts, valid.Counts[0]]
        };

        Assert.ThrowsExactly<InvalidOperationException>(() => invalid.Validate(source));

        CollectionAssert.AreEqual(new[] { "train", "validation", "holdout" }, valid.Counts.Select(c => c.Split).ToArray());
        Assert.AreEqual(caseId == "missing-count" ? 2 : 4, invalid.Counts.Length);
        valid.Validate(source);
    }

    private static void AssertGroupIds(SplitManifest manifest, string split, long[] expected) =>
        CollectionAssert.AreEqual(expected, manifest.Rows.Where(r => r.Split == split)
            .Select(r => r.GroupId).Distinct().Order().ToArray());

    private static double[] Costs(int[,] counts, int ham, int spam)
    {
        double[] targetsHam = [7.8, 2.6, 2.6], targetsSpam = [7.2, 2.4, 2.4];
        return Enumerable.Range(0, 3).Select(candidate => Enumerable.Range(0, 3).Sum(partition =>
            Math.Pow(counts[partition, 0] + (partition == candidate ? ham : 0) - targetsHam[partition], 2) / targetsHam[partition] +
            Math.Pow(counts[partition, 1] + (partition == candidate ? spam : 0) - targetsSpam[partition], 2) / targetsSpam[partition])).ToArray();
    }
}

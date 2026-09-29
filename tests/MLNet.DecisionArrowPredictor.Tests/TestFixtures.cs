using System.Globalization;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

// All file fixtures are owned by this test project, independent of the runner's cwd.
internal sealed class TempDirectory : IDisposable
{
    public string Root { get; }

    public TempDirectory()
    {
        string projectRoot = Path.GetDirectoryName(SourcePath())!;
        if (Path.GetFileName(projectRoot) != "MLNet.DecisionArrowPredictor.Tests")
            throw new InvalidOperationException("Cannot resolve the authorized test fixture root.");
        Root = Path.Combine(projectRoot, ".artifacts", "fixtures", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Root);
    }

    private static string SourcePath([CallerFilePath] string path = "") => path;

    public string FilePath(string name)
    {
        string path = Path.GetFullPath(Path.Combine(Root, name));
        if (!path.StartsWith(Root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Fixture path escaped its owned directory.");
        return path;
    }

    public string Put(string name, byte[] bytes)
    {
        string path = FilePath(name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, bytes);
        return path;
    }

    public string PutText(string name, string text) => Put(name, new UTF8Encoding(false, true).GetBytes(text));

    public void Dispose()
    {
        // Only this instance's new, untracked fixture tree is removed.
        // Cleanup failure must not mask an assertion/production exception.
        try { Directory.Delete(Root, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}

internal sealed record SerializationFixture(int Count, string Text, long[] Ids, bool[] Labels);

internal static class ArtifactExpectations
{
    public const string EmptyHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    public const string AbcHash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    public static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    public static string HashFile(string path) => Hash(File.ReadAllBytes(path));
    public static void Bytes(byte[] expected, string path) => CollectionAssert.AreEqual(expected, File.ReadAllBytes(path));
}

internal static class QuestionFixtures
{
    public static string[] Projection =>
    [
        "commercial_solicitation", "requested_contact_action",
        "time_pressure[0]", "time_pressure[1]", "time_pressure[2]",
        "message_purpose[0]", "message_purpose[1]", "message_purpose[2]",
        "message_purpose[3]", "message_purpose[4]"
    ];

    public static JsonObject Document() => JsonNode.Parse("""
        {
          "schemaVersion": 1,
          "featureProjection": [
            "commercial_solicitation", "requested_contact_action",
            "time_pressure[0]", "time_pressure[1]", "time_pressure[2]",
            "message_purpose[0]", "message_purpose[1]", "message_purpose[2]",
            "message_purpose[3]", "message_purpose[4]"
          ],
          "questions": [
            {"id":"commercial_solicitation","kind":"binary"},
            {"id":"requested_contact_action","kind":"binary"},
            {"id":"time_pressure","kind":"score","rubric":["none","mild","explicit urgent deadline"]},
            {"id":"message_purpose","kind":"choice","candidates":[
              {"id":"personal"},{"id":"service_notice"},{"id":"promotion"},
              {"id":"financial_offer"},{"id":"other"}]},
            {"id":"spam_baseline","kind":"binary"}
          ]
        }
        """)!.AsObject();

    public static byte[] Bytes(JsonObject? document = null) =>
        Encoding.UTF8.GetBytes((document ?? Document()).ToJsonString());

    public static JsonObject Mutate(string mutation)
    {
        var doc = Document();
        var questions = doc["questions"]!.AsArray();
        var projection = doc["featureProjection"]!.AsArray();
        switch (mutation)
        {
            case "schema-version": doc["schemaVersion"] = 2; break;
            case "question-count": questions.RemoveAt(4); break;
            case "projection-count": projection.RemoveAt(9); break;
            case "projection-order":
                projection[0] = "requested_contact_action";
                projection[1] = "commercial_solicitation";
                break;
            case "baseline-intrusion": projection[9] = "spam_baseline"; break;
            case "question-id": questions[0]!["id"] = "commercial_SOLICITATION"; break;
            case "question-kind": questions[0]!["kind"] = "score"; break;
            case "question-order":
                var first = questions[0]!.DeepClone();
                questions[0] = questions[1]!.DeepClone();
                questions[1] = first;
                break;
            case "rubric-order":
                questions[2]!["rubric"] = new JsonArray("mild", "none", "explicit urgent deadline");
                break;
            case "candidate-order":
                questions[3]!["candidates"]![0]!["id"] = "service_notice";
                questions[3]!["candidates"]![1]!["id"] = "personal";
                break;
            default: throw new ArgumentOutOfRangeException(nameof(mutation));
        }
        return doc;
    }
}

internal static class CorpusFixtures
{
    public const string FirstTab = "ham\t  Keep\tinner tab  \nspam\tOffer \t now  \n";

    public static string BalancedText() => string.Concat(Enumerable.Range(0, 30).Select(j =>
        FormattableString.Invariant($"ham\ttopic-{j:00}\nspam\ttopic-{j:00}\n")));

    public static CorpusRow[] Balanced() => Enumerable.Range(0, 30).SelectMany(j => new[]
    {
        new CorpusRow(2 * j + 1, false, FormattableString.Invariant($"topic-{j:00}")),
        new CorpusRow(2 * j + 2, true, FormattableString.Invariant($"topic-{j:00}"))
    }).ToArray();

    public static CorpusRow[] Unbalanced() =>
    [
        new(1, false, "group-one"), new(2, false, "group-one"),
        new(3, false, "group-one"), new(4, true, "group-one"),
        new(5, false, "group-five"), new(6, true, "group-five"), new(7, true, "group-five"),
        .. Enumerable.Range(8, 18).Select(i => new CorpusRow(i, i % 2 == 1,
            FormattableString.Invariant($"unique-{i:00}")))
    ];

    public static CorpusRow[] Chain()
    {
        string middle = string.Concat(Enumerable.Range(0, 60).Select(i =>
            new string([(char)('a' + i / 26), (char)('a' + i % 26)])));
        var a = middle.ToCharArray(); a[10] = 'z';
        var c = middle.ToCharArray(); c[90] = 'z';
        return [new(30, false, new string(a)), new(10, true, middle), new(20, false, new string(c))];
    }
}

// Synthetic, hash-consistent local evidence, NOT authenticated UCI acquisition.
internal sealed class LocalAcquisitionFixture : IDisposable
{
    public TempDirectory Directory { get; } = new();
    public string CorpusPath { get; }
    public string ReceiptPath { get; }
    public AcquisitionReceipt Receipt { get; }
    public static DateTimeOffset Timestamp => new(2025, 1, 2, 3, 4, 5, TimeSpan.Zero);

    public LocalAcquisitionFixture()
    {
        byte[] corpus = Encoding.UTF8.GetBytes(CorpusFixtures.BalancedText());
        byte[] readme = Encoding.UTF8.GetBytes("Authored offline fixture; not a downloaded corpus.\n");
        Directory.Put("SMSSpamCollection", corpus);
        Directory.Put("readme", readme);
        Directory.PutText("uci-page.html", "<html>Local evidence: https://creativecommons.org/licenses/by/4.0/</html>");
        Directory.PutText("uci-metadata.json", """{"data":{"dataset_doi":"10.24432/C5CC84","num_instances":999}}""");
        using (var stream = new MemoryStream())
        {
            using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
            {
                foreach (var (name, bytes) in new[] { ("SMSSpamCollection", corpus), ("readme", readme) })
                {
                    var entry = zip.CreateEntry(name);
                    entry.LastWriteTime = Timestamp;
                    using var output = entry.Open();
                    output.Write(bytes);
                }
            }
            Directory.Put("sms-spam-collection.zip", stream.ToArray());
        }
        CorpusPath = Directory.FilePath("SMSSpamCollection");
        ReceiptPath = Directory.FilePath("acquisition.json");
        Receipt = new(1, "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
            "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
            "10.24432/C5CC84", "Authored local test evidence", "Local CC BY 4.0 evidence",
            "uci-page.html", ArtifactExpectations.HashFile(Directory.FilePath("uci-page.html")),
            "uci-metadata.json", ArtifactExpectations.HashFile(Directory.FilePath("uci-metadata.json")),
            "sms-spam-collection.zip", ArtifactExpectations.HashFile(Directory.FilePath("sms-spam-collection.zip")),
            "SMSSpamCollection", ArtifactExpectations.Hash(corpus), "readme", ArtifactExpectations.Hash(readme),
            999, "first-tab-strict-utf8-original-lines-v1", Timestamp);
        WriteReceipt(Receipt);
    }

    public void WriteReceipt(AcquisitionReceipt value) =>
        File.WriteAllBytes(ReceiptPath, JsonSerializer.SerializeToUtf8Bytes(value,
            new JsonSerializerOptions(JsonSerializerDefaults.Web)));

    public void Dispose() => Directory.Dispose();
}

internal static class SplitExpectations
{
    public static string[] Names => ["train", "validation", "holdout"];

    public static string OrderKey(long id, int seed) => ArtifactExpectations.Hash(
        Encoding.UTF8.GetBytes(seed.ToString(CultureInfo.InvariantCulture) + ":" +
            id.ToString(CultureInfo.InvariantCulture)));

    public static SplitManifest Create(CorpusRow[] rows, GroupDiagnostics? groups = null) =>
        SplitManifest.Create(groups ?? DuplicateGrouping.Build(rows),
            new string('a', 64), new string('b', 64), new string('c', 64), rows);

    public static void Complete(SplitManifest manifest, CorpusRow[] source, GroupDiagnostics diagnostics)
    {
        Assert.AreEqual(1, manifest.Version);
        Assert.AreEqual(1729, manifest.Seed);
        Assert.AreEqual(new string('a', 64), manifest.CorpusSha256);
        Assert.AreEqual(new string('b', 64), manifest.QuestionsSha256);
        Assert.AreEqual(new string('c', 64), manifest.GroupDiagnosticsSha256);
        Assert.AreEqual("size-desc/sha256-seed-id-tie/greedy-squared-class-deficit-60-20-20-v1", manifest.Algorithm);
        CollectionAssert.AreEqual(source.Select(r => r.RowId).Order().ToArray(), manifest.Rows.Select(r => r.RowId).ToArray());
        Assert.AreEqual(source.Length, manifest.Rows.Select(r => r.RowId).Distinct().Count());
        foreach (var row in source)
            Assert.AreEqual(row.Label, manifest.Rows.Single(r => r.RowId == row.RowId).Label);
        foreach (var group in diagnostics.Membership)
        {
            var assigned = manifest.Rows.Where(r => r.GroupId == group.GroupId).ToArray();
            CollectionAssert.AreEqual(group.RowIds, assigned.Select(r => r.RowId).ToArray());
            Assert.AreEqual(1, assigned.Select(r => r.Split).Distinct().Count());
        }
        CollectionAssert.AreEqual(Names, manifest.Counts.Select(c => c.Split).ToArray());
        foreach (var count in manifest.Counts)
        {
            var actual = manifest.Rows.Where(r => r.Split == count.Split).ToArray();
            Assert.AreEqual(actual.Length, count.Rows);
            Assert.AreEqual(actual.Count(r => !r.Label), count.Ham);
            Assert.AreEqual(actual.Count(r => r.Label), count.Spam);
            Assert.AreEqual(actual.Select(r => r.GroupId).Distinct().Count(), count.Groups);
            Assert.IsTrue(count.Ham > 0 && count.Spam > 0);
        }
        Assert.AreEqual(source.Length, manifest.Counts.Sum(c => c.Rows));
        manifest.Validate(source);
    }

    public static PartitionCount[] Counts(SplitRow[] rows) => Names.Select(n =>
    {
        var selected = rows.Where(r => r.Split == n).ToArray();
        return new PartitionCount(n, selected.Length, selected.Count(r => !r.Label),
            selected.Count(r => r.Label), selected.Select(r => r.GroupId).Distinct().Count());
    }).ToArray();

    public static void Equal(SplitManifest expected, SplitManifest actual)
    {
        Assert.AreEqual(expected.Version, actual.Version);
        Assert.AreEqual(expected.Seed, actual.Seed);
        Assert.AreEqual(expected.CorpusSha256, actual.CorpusSha256);
        Assert.AreEqual(expected.QuestionsSha256, actual.QuestionsSha256);
        Assert.AreEqual(expected.GroupDiagnosticsSha256, actual.GroupDiagnosticsSha256);
        Assert.AreEqual(expected.Algorithm, actual.Algorithm);
        CollectionAssert.AreEqual(expected.Counts, actual.Counts);
        CollectionAssert.AreEqual(expected.Rows, actual.Rows);
    }
}

internal static class EvaluationGoldens
{
    public static Prediction[] Tied =>
    [
        new(1, 10, true, .9), new(2, 20, false, .9),
        new(3, 30, true, .5), new(4, 40, false, .1)
    ];

    public static Prediction[] Grouped =>
    [
        new(1, 10, false, .2), new(2, 10, false, .4), new(3, 20, true, .8)
    ];

    public static string[] MetricNames => ["auprc", "rocAuc", "logLoss", "brier", "recall", "falsePositiveRate"];
    public static int[] Defined => [505, 505, 1000, 1000, 767, 738];

    public static void Near(double expected, double actual) => Assert.AreEqual(expected, actual, 1e-12);
    public static void NullableNear(double? expected, double? actual)
    {
        if (expected is null) Assert.IsNull(actual);
        else
        {
            Assert.IsTrue(actual.HasValue);
            Near(expected.Value, actual.Value);
        }
    }

    public static void MetricsEqual(Metrics expected, Metrics actual)
    {
        Assert.AreEqual(expected.Rows, actual.Rows);
        Assert.AreEqual(expected.Spam, actual.Spam);
        Assert.AreEqual(expected.Ham, actual.Ham);
        NullableNear(expected.Auprc, actual.Auprc);
        NullableNear(expected.RocAuc, actual.RocAuc);
        Near(expected.LogLoss, actual.LogLoss);
        Near(expected.Brier, actual.Brier);
        Assert.AreEqual(expected.Threshold, actual.Threshold);
        Assert.AreEqual(expected.TruePositive, actual.TruePositive);
        Assert.AreEqual(expected.FalsePositive, actual.FalsePositive);
        Assert.AreEqual(expected.TrueNegative, actual.TrueNegative);
        Assert.AreEqual(expected.FalseNegative, actual.FalseNegative);
        NullableNear(expected.Precision, actual.Precision);
        NullableNear(expected.Recall, actual.Recall);
        NullableNear(expected.FalsePositiveRate, actual.FalsePositiveRate);
    }

    public static void Report(BootstrapReport actual, int[] defined, double?[] lower, double?[] upper, bool paired = false)
    {
        Assert.AreEqual(1729, actual.Seed);
        Assert.AreEqual(1000, actual.Resamples);
        Assert.AreEqual("non-stratified duplicate-group resampling; whole groups with replacement", actual.Unit);
        CollectionAssert.AreEqual(MetricNames.Select(n => paired ? "pairedDelta:" + n : n).ToArray(),
            actual.Intervals.Select(i => i.Metric).ToArray());
        for (int i = 0; i < 6; i++)
        {
            Assert.AreEqual(defined[i], actual.Intervals[i].Defined);
            Assert.AreEqual(1000 - defined[i], actual.Intervals[i].Undefined);
            Assert.AreEqual(1000, actual.Intervals[i].Defined + actual.Intervals[i].Undefined);
            NullableNear(lower[i], actual.Intervals[i].Lower);
            NullableNear(upper[i], actual.Intervals[i].Upper);
        }
    }

    // Draw-table oracle uses only System.Random, not production evaluation/bootstrap.
    public static int[] DrawOutcomes()
    {
        var random = new Random(1729);
        var counts = new int[3];
        for (int i = 0; i < 1000; i++)
            counts[random.Next(2) + random.Next(2)]++;
        return counts;
    }

    public static double[] FloorBounds(double hamOnly, double mixed, double spamOnly)
    {
        var draws = DrawOutcomes();
        var values = Enumerable.Repeat(hamOnly, draws[0])
            .Concat(Enumerable.Repeat(mixed, draws[1]))
            .Concat(Enumerable.Repeat(spamOnly, draws[2])).Order().ToArray();
        return [values[(int)Math.Floor(.025 * 999)], values[(int)Math.Floor(.975 * 999)]];
    }
}

internal static class LearningRowFixtures
{
    public const string Identity = "authored-tiny-feature-contract-v1";
    public static string DatasetHash => new('d', 64);
    public static string SplitHash => new('e', 64);
    public static string QuestionsHash => new('f', 64);

    public static LearningRow[] Grouped600() => Enumerable.Range(0, 600).Select(i => new LearningRow
    {
        RowId = i + 1,
        GroupId = 3 * (i / 3) + 1,
        Label = i % 3 == 2,
        Text = FormattableString.Invariant($"authored group {i / 3} message {i % 3}"),
        Semantic = Enumerable.Range(0, 10).Select(j => (float)(.1 + .01 * j)).ToArray(),
        SpamBaseline = i % 3 == 2 ? .8 : .2
    }).ToArray();

    public static LearningRow[] Training() => Enumerable.Range(0, 12)
        .Select(i => Tiny(i + 1, i + 1, i % 2 == 1, i, false)).ToArray();

    public static LearningRow[] Validation() => Enumerable.Range(0, 4)
        .Select(i => Tiny(101 + i, 201 + i, i % 2 == 1, i, true)).ToArray();

    private static LearningRow Tiny(long id, long group, bool spam, int variant, bool heldout) => new()
    {
        RowId = id, GroupId = group, Label = spam,
        Text = (spam ? "claim prize promotion urgent offer" : "family dinner meeting personal notice") +
            (heldout ? " zzheldoutuniqueq" : " orchardtrainonly") +
            (variant % 3 == 0 ? " today" : " tomorrow"),
        Semantic = Enumerable.Range(0, 10)
            .Select(j => (float)((spam ? .85 : .10) + .005 * j + .002 * (variant % 3))).ToArray(),
        // Deliberately misleading: a learned arm must not sneak this label proxy into Features.
        SpamBaseline = spam ? .01 : .99
    };

    public static LearningRow Copy(LearningRow row, double? baseline = null) => new()
    {
        RowId = row.RowId, GroupId = row.GroupId, Label = row.Label, Text = row.Text,
        Semantic = row.Semantic.ToArray(), SpamBaseline = baseline ?? row.SpamBaseline
    };

    public static void Unchanged(LearningRow[] before, LearningRow[] after)
    {
        Assert.AreEqual(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.AreEqual(before[i].RowId, after[i].RowId);
            Assert.AreEqual(before[i].GroupId, after[i].GroupId);
            Assert.AreEqual(before[i].Label, after[i].Label);
            Assert.AreEqual(before[i].Text, after[i].Text);
            Assert.AreEqual(before[i].SpamBaseline, after[i].SpamBaseline);
            CollectionAssert.AreEqual(before[i].Semantic, after[i].Semantic);
        }
    }
}

// Exactly one successful Fit per arm, with no model shared outside this sealed test class.
internal sealed class TinyModelFixture : IDisposable
{
    public TempDirectory Directory { get; } = new();
    public LearningRow[] Training { get; } = LearningRowFixtures.Training();
    public LearningRow[] Validation { get; } = LearningRowFixtures.Validation();
    private readonly Lazy<TrainedArm> text;
    private readonly Lazy<TrainedArm> semantic;
    private readonly Lazy<TrainedArm> combined;

    public TinyModelFixture()
    {
        text = new(() => Fit("text"));
        semantic = new(() => Fit("semantic"));
        combined = new(() => Fit("combined"));
    }

    public TrainedArm Arm(string arm) => arm switch
    {
        "text" => text.Value, "semantic" => semantic.Value, "combined" => combined.Value,
        _ => throw new ArgumentOutOfRangeException(nameof(arm))
    };

    public string Output(string arm) => Directory.FilePath(arm);
    public string ModelPath(string arm) => Path.Combine(Output(arm), arm + "-12.mlnet");

    private TrainedArm Fit(string arm)
    {
        try
        {
            return PredictorTraining.Fit(arm, 12, Training, Validation, LearningRowFixtures.Identity,
                LearningRowFixtures.DatasetHash, LearningRowFixtures.SplitHash,
                LearningRowFixtures.QuestionsHash, Output(arm));
        }
        catch
        {
            // A failed arm must not discard another successfully cached arm's originals.
            string output = Output(arm);
            if (System.IO.Directory.Exists(output)) System.IO.Directory.Delete(output, true);
            throw;
        }
    }

    public void Dispose() => Directory.Dispose();
}

internal static class ModelExpectations
{
    public static void Predictions(LearningRow[] input, Prediction[] actual)
    {
        Assert.AreEqual(input.Length, actual.Length);
        CollectionAssert.AreEqual(input.Select(r => r.RowId).ToArray(), actual.Select(r => r.RowId).ToArray());
        CollectionAssert.AreEqual(input.Select(r => r.GroupId).ToArray(), actual.Select(r => r.GroupId).ToArray());
        CollectionAssert.AreEqual(input.Select(r => r.Label).ToArray(), actual.Select(r => r.Label).ToArray());
        foreach (var p in actual)
        {
            Assert.IsTrue(double.IsFinite(p.Probability), $"Row {p.RowId} probability must be finite.");
            Assert.IsTrue(p.Probability >= 0 && p.Probability <= 1, $"Row {p.RowId} probability must be in [0,1].");
        }
    }

    public static void Replay(Prediction[] expected, Prediction[] actual)
    {
        CollectionAssert.AreEqual(expected.Select(r => (r.RowId, r.GroupId, r.Label)).ToArray(),
            actual.Select(r => (r.RowId, r.GroupId, r.Label)).ToArray());
        for (int i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i].Probability, actual[i].Probability, 1e-6, $"Row {expected[i].RowId}");
    }

    private static (int TP, int FP, int TN, int FN) Confusion(Prediction[] rows, double threshold) =>
        (rows.Count(p => p.Label && p.Probability >= threshold),
         rows.Count(p => !p.Label && p.Probability >= threshold),
         rows.Count(p => !p.Label && p.Probability < threshold),
         rows.Count(p => p.Label && p.Probability < threshold));

    // Independent exhaustive four-row threshold oracle: no production evaluation calls.
    public static double Threshold(Prediction[] rows, bool budget)
    {
        var candidates = rows.Select(p => p.Probability).Distinct().Append(Math.BitIncrement(1d))
            .Select(t =>
            {
                var c = Confusion(rows, t);
                double recall = (double)c.TP / rows.Count(p => p.Label);
                double f1 = c.TP == 0 ? 0 : 2d * c.TP / (2 * c.TP + c.FP + c.FN);
                return (Threshold: t, c.FP, Recall: recall, F1: f1);
            });
        if (budget)
            candidates = candidates.Where(c => (double)c.FP / rows.Count(p => !p.Label) <= .01);
        return candidates.OrderByDescending(c => budget ? c.Recall : c.F1)
            .ThenBy(c => c.FP).ThenByDescending(c => c.Threshold).First().Threshold;
    }

    public static void ValidationMetrics(Prediction[] rows, double threshold, Metrics actual)
    {
        int spam = rows.Count(p => p.Label), ham = rows.Length - spam;
        var c = Confusion(rows, threshold);
        Assert.AreEqual(rows.Length, actual.Rows);
        Assert.AreEqual(spam, actual.Spam);
        Assert.AreEqual(ham, actual.Ham);
        Assert.AreEqual(threshold, actual.Threshold);
        Assert.AreEqual(c.TP, actual.TruePositive);
        Assert.AreEqual(c.FP, actual.FalsePositive);
        Assert.AreEqual(c.TN, actual.TrueNegative);
        Assert.AreEqual(c.FN, actual.FalseNegative);
        EvaluationGoldens.NullableNear(c.TP + c.FP == 0 ? null : (double)c.TP / (c.TP + c.FP), actual.Precision);
        EvaluationGoldens.NullableNear((double)c.TP / spam, actual.Recall);
        EvaluationGoldens.NullableNear((double)c.FP / ham, actual.FalsePositiveRate);
        EvaluationGoldens.Near(rows.Average(p => Math.Pow(p.Probability - (p.Label ? 1 : 0), 2)), actual.Brier);
        EvaluationGoldens.Near(rows.Average(p =>
        {
            double q = Math.Clamp(p.Probability, 1e-15, 1 - 1e-15);
            return -Math.Log(p.Label ? q : 1 - q);
        }), actual.LogLoss);
        double wins = rows.Where(p => p.Label).Sum(p => rows.Where(n => !n.Label)
            .Sum(n => p.Probability > n.Probability ? 1d : p.Probability == n.Probability ? .5 : 0));
        EvaluationGoldens.NullableNear(wins / (spam * ham), actual.RocAuc);
        double ap = 0;
        foreach (double score in rows.Select(p => p.Probability).Distinct())
        {
            int positivesAtScore = rows.Count(p => p.Label && p.Probability == score);
            int positivesAboveOrAt = rows.Count(p => p.Label && p.Probability >= score);
            int rowsAboveOrAt = rows.Count(p => p.Probability >= score);
            ap += (double)positivesAtScore / spam * positivesAboveOrAt / rowsAboveOrAt;
        }
        EvaluationGoldens.NullableNear(ap, actual.Auprc);
    }
}

internal sealed class WorkflowFixture : IDisposable
{
    public LocalAcquisitionFixture Acquisition { get; } = new();
    public TempDirectory Directory => Acquisition.Directory;
    public CorpusRow[] Source { get; }
    public byte[] QuestionsBytes { get; }
    public string QuestionsPath { get; }
    public string ReviewOutput => Directory.FilePath("review");
    public string Output => Directory.FilePath("frozen");
    public string DiagnosticsPath => Path.Combine(ReviewOutput, "group-diagnostics.json");
    public GroupDiagnostics Groups { get; }
    public string ReviewedHash { get; }

    public WorkflowFixture()
    {
        try
        {
            Source = Enumerable.Range(0, 30).SelectMany(j => new CorpusRow[]
            {
                new(2 * j + 1, false, FormattableString.Invariant($"  ToPiC-{j:00}\t ")),
                new(2 * j + 2, true, FormattableString.Invariant($"topic-{j:00}"))
            }).ToArray();
            byte[] corpus = Encoding.UTF8.GetBytes(string.Concat(Source.Select(r =>
                (r.Label ? "spam\t" : "ham\t") + r.Text + "\n")));
            File.WriteAllBytes(Acquisition.CorpusPath, corpus);
            byte[] readme = File.ReadAllBytes(Directory.FilePath("readme"));
            using var memory = new MemoryStream();
            using (var zip = new ZipArchive(memory, ZipArchiveMode.Create, true))
            {
                foreach (var (name, bytes) in new[] { ("SMSSpamCollection", corpus), ("readme", readme) })
                {
                    var entry = zip.CreateEntry(name);
                    entry.LastWriteTime = LocalAcquisitionFixture.Timestamp;
                    using var stream = entry.Open();
                    stream.Write(bytes);
                }
            }
            byte[] archive = memory.ToArray();
            File.WriteAllBytes(Directory.FilePath("sms-spam-collection.zip"), archive);
            Acquisition.WriteReceipt(Acquisition.Receipt with
            {
                CorpusSha256 = ArtifactExpectations.Hash(corpus), ArchiveSha256 = ArtifactExpectations.Hash(archive)
            });
            // Freeze now authenticates the exact frozen document bytes, not just its shape.
            // The parent's prebuilt sample has already copied this inert local artifact here.
            QuestionsBytes = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "questions.v1.json"));
            QuestionsPath = Directory.Put("questions-input.json", QuestionsBytes);
            Groups = PreparationWorkflow.Review(Acquisition.ReceiptPath, Acquisition.CorpusPath, ReviewOutput);
            ReviewedHash = ArtifactExpectations.HashFile(DiagnosticsPath);
        }
        catch
        {
            Acquisition.Dispose();
            throw;
        }
    }

    public PreparationReceipt Freeze(string? questions = null, string? diagnostics = null,
        string? reviewedHash = null, string? output = null) =>
        PreparationWorkflow.Freeze(Acquisition.ReceiptPath, Acquisition.CorpusPath,
            questions ?? QuestionsPath, diagnostics ?? DiagnosticsPath, reviewedHash ?? ReviewedHash, output ?? Output);

    public Dictionary<string, byte[]> SnapshotEvidence() => new[]
    {
        "acquisition.json", "SMSSpamCollection", "readme", "uci-page.html",
        "uci-metadata.json", "sms-spam-collection.zip", "questions-input.json"
    }.ToDictionary(n => n, n => File.ReadAllBytes(Directory.FilePath(n)), StringComparer.Ordinal);

    public void AssertEvidence(Dictionary<string, byte[]> before)
    {
        foreach (var (name, bytes) in before) ArtifactExpectations.Bytes(bytes, Directory.FilePath(name));
    }

    public void Dispose() => Acquisition.Dispose();
}

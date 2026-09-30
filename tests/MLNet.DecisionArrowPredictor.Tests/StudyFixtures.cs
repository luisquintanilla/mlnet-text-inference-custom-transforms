using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

// Authored DTO fixtures only: no exporter metadata, real corpus, or extraction.
internal sealed class StudyFixture : IDisposable
{
    public const string Identity = "authored-workflow-contract-v1";
    public static string[] ArmNames => ["combined", "direct", "prior", "semantic", "text"];
    public static int[] Targets => [100, 500, 600];
    public static JsonSerializerOptions Json => new(JsonSerializerDefaults.Web) { WriteIndented = true };
    public TempDirectory Directory { get; } = new();
    public ImportedStudy Study { get; } = Authored();
    public string TrainingOutput => Directory.FilePath("training");
    public string FreezePath => Path.Combine(TrainingOutput, "training.freeze.json");
    public string EvaluationOutput => Directory.FilePath("evaluation");
    private readonly Lazy<TrainingFreeze> training;
    private readonly Lazy<StudyEvaluation> evaluation;

    public StudyFixture()
    {
        training = new(() => StudyWorkflow.Train(Study, TrainingOutput));
        evaluation = new(() =>
        {
            _ = Training;
            return StudyWorkflow.Evaluate(Study, FreezePath, EvaluationOutput);
        });
    }

    public TrainingFreeze Training => training.Value;
    public StudyEvaluation Evaluation => evaluation.Value;

    public static ImportedStudy Authored()
    {
        var rows = Enumerable.Range(0, 600)
            .Select(i => Row(i + 1, 3 * (i / 3) + 1, i / 3 % 3 == 2, "train", i / 3))
            .Concat(Enumerable.Range(0, 4).Select(i => Row(1001 + i, 1001 + 2 * (i / 2), i >= 2, "validation", i / 2)))
            .Concat(Enumerable.Range(0, 4).Select(i => Row(2001 + i, 2001 + 2 * (i / 2), i >= 2, "holdout", i / 2)))
            .ToArray();
        var assignments = rows.Select(r => new SplitRow(r.RowId, r.GroupId, r.Label,
            r.RowId < 1000 ? "train" : r.RowId < 2000 ? "validation" : "holdout")).ToArray();
        var split = new SplitManifest(1, 1729, new string('a', 64), new string('f', 64), new string('c', 64),
            "size-desc/sha256-seed-id-tie/greedy-squared-class-deficit-60-20-20-v1",
            SplitExpectations.Counts(assignments), assignments);
        return new(Identity, new string('d', 64), SplitHash(split), split.QuestionsSha256, rows, split);
    }

    private static LearningRow Row(long id, long group, bool spam, string partition, int variant) => new()
    {
        RowId = id, GroupId = group, Label = spam,
        Text = (spam ? "claim prize urgent offer " : "family dinner personal notice ") +
            (partition == "train" ? "orchardtrainonly" : partition == "validation" ? "zzvalidationonlyq" : "zzheldoutstudyq"),
        Semantic = Enumerable.Range(0, 10)
            .Select(j => (float)((spam ? .85 : .10) + .005 * j + .001 * (variant % 5))).ToArray(),
        SpamBaseline = partition == "train" ? spam ? .01 : .99 :
            partition == "validation" ? spam ? .8 : .2 : spam ? .7 : .3
    };

    public static string SplitHash(SplitManifest split) =>
        ArtifactExpectations.Hash(JsonSerializer.SerializeToUtf8Bytes(split, Json));

    public static ImportedStudy Copy(ImportedStudy study) => study with
    {
        Rows = study.Rows.Select(r => LearningRowFixtures.Copy(r)).ToArray(),
        Split = study.Split with { Rows = study.Split.Rows.ToArray(), Counts = study.Split.Counts.ToArray() }
    };

    public static TrainingFreeze Copy(TrainingFreeze freeze) => freeze with
    {
        Arms = freeze.Arms.Select(a => a with { TrainingRowIds = a.TrainingRowIds.ToArray() }).ToArray()
    };

    // Independent seeded-hash ordering, never calls production Subset.
    public LearningRow[] ExpectedSubset(int target)
    {
        var train = Study.Rows.Where(r => r.RowId < 1000).ToArray();
        int groupsNeeded = target >= train.Length ? 200 : (target + 2) / 3;
        long[] ids = train.Select(r => r.GroupId).Distinct()
            .OrderBy(g => SplitExpectations.OrderKey(g, 1), StringComparer.Ordinal).Take(groupsNeeded).ToArray();
        return train.Where(r => ids.Contains(r.GroupId)).OrderBy(r => r.RowId).ToArray();
    }

    public string CopyTraining(TempDirectory destination)
    {
        _ = Training;
        string root = destination.FilePath("training-copy");
        foreach (string path in System.IO.Directory.GetFiles(TrainingOutput))
            destination.Put("training-copy/" + Path.GetFileName(path), File.ReadAllBytes(path));
        return root;
    }

    public static void WriteFreeze(string root, TrainingFreeze freeze) =>
        File.WriteAllBytes(Path.Combine(root, "training.freeze.json"), JsonSerializer.SerializeToUtf8Bytes(freeze, Json));

    public static ModelReceipt Receipt(string root, ArmFreeze arm) =>
        JsonSerializer.Deserialize<ModelReceipt>(File.ReadAllBytes(Path.Combine(root, arm.ModelReceiptFile!)), Json)!;

    public static Dictionary<string, byte[]> Snapshot(string root) =>
        System.IO.Directory.GetFiles(root).ToDictionary(p => Path.GetFileName(p)!, File.ReadAllBytes, StringComparer.Ordinal);

    public static void Unchanged(string root, Dictionary<string, byte[]> before)
    {
        foreach (var (name, bytes) in before) ArtifactExpectations.Bytes(bytes, Path.Combine(root, name));
    }

    public static void ReportEqual(BootstrapReport expected, BootstrapReport actual)
    {
        Assert.AreEqual(expected.Seed, actual.Seed);
        Assert.AreEqual(expected.Resamples, actual.Resamples);
        Assert.AreEqual(expected.Unit, actual.Unit);
        CollectionAssert.AreEqual(expected.Intervals, actual.Intervals);
    }

    public void Dispose() => Directory.Dispose();
}

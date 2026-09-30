using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class StudyWorkflowTests
{
    private static StudyFixture fixture = null!;

    [ClassInitialize]
    public static void Initialize(TestContext _) => fixture = new();

    [ClassCleanup]
    public static void Cleanup() => fixture.Dispose();

    [TestMethod]
    public void Validate_AuthoredStudyRetainsExactRowsAndInclusiveProbabilities()
    {
        var study = StudyFixture.Copy(fixture.Study);
        study.Rows[0].Semantic[0] = 0;
        study.Rows[0].Semantic[9] = 1;
        study.Rows[0].SpamBaseline = 0;
        study.Rows[1].SpamBaseline = 1;
        var before = study.Rows.Select(r => LearningRowFixtures.Copy(r)).ToArray();

        StudyWorkflow.Validate(study);

        Assert.AreEqual(608, study.Rows.Length);
        Assert.AreEqual(600, study.Split.Counts[0].Rows);
        Assert.AreEqual(198, study.Split.Counts[0].Spam);
        Assert.AreEqual(402, study.Split.Counts[0].Ham);
        Assert.AreEqual(200, study.Split.Counts[0].Groups);
        Assert.AreEqual(4, study.Split.Counts[1].Rows);
        Assert.AreEqual(4, study.Split.Counts[2].Rows);
        Assert.IsNull(study.Extraction);
        LearningRowFixtures.Unchanged(before, study.Rows);
    }

    [TestMethod]
    [DataRow("identity-empty")]
    [DataRow("identity-blank")]
    [DataRow("missing-row")]
    [DataRow("duplicate-id")]
    [DataRow("unknown-id")]
    [DataRow("label")]
    [DataRow("group")]
    [DataRow("semantic-nine")]
    [DataRow("semantic-eleven")]
    [DataRow("semantic-nan")]
    [DataRow("semantic-positive-infinity")]
    [DataRow("semantic-negative-infinity")]
    [DataRow("semantic-negative")]
    [DataRow("semantic-above-one")]
    [DataRow("baseline-nan")]
    [DataRow("baseline-positive-infinity")]
    [DataRow("baseline-negative-infinity")]
    [DataRow("baseline-negative")]
    [DataRow("baseline-above-one")]
    [DataRow("questions")]
    [DataRow("split-version")]
    [DataRow("split-seed")]
    public void Validate_ImportedRowAndContractMutationsAreRejected(string mutation)
    {
        var original = fixture.Study.Rows.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        var study = StudyFixture.Copy(fixture.Study);
        switch (mutation)
        {
            case "identity-empty": study = study with { FeatureFingerprint = "" }; break;
            case "identity-blank": study = study with { FeatureFingerprint = " \t" }; break;
            case "missing-row": study = study with { Rows = study.Rows[..^1] }; break;
            case "duplicate-id": study.Rows[1].RowId = study.Rows[0].RowId; break;
            case "unknown-id": study.Rows[0].RowId = 9999; break;
            case "label": study.Rows[0].Label = !study.Rows[0].Label; break;
            case "group": study.Rows[0].GroupId = 9999; break;
            case "semantic-nine": study.Rows[0].Semantic = new float[9]; break;
            case "semantic-eleven": study.Rows[0].Semantic = new float[11]; break;
            case "questions": study = study with { QuestionsSha256 = new string('0', 64) }; break;
            case "split-version": study = study with { Split = study.Split with { Version = 2 } }; break;
            case "split-seed": study = study with { Split = study.Split with { Seed = 1730 } }; break;
            default:
                double number = mutation[(mutation.IndexOf('-') + 1)..] switch
                {
                    "nan" => double.NaN,
                    "positive-infinity" => double.PositiveInfinity,
                    "negative-infinity" => double.NegativeInfinity,
                    "negative" => -float.Epsilon,
                    "above-one" => MathF.BitIncrement(1f),
                    _ => throw new ArgumentOutOfRangeException(nameof(mutation))
                };
                if (mutation.StartsWith("semantic", StringComparison.Ordinal)) study.Rows[0].Semantic[4] = (float)number;
                else study.Rows[0].SpamBaseline = number;
                break;
        }
        var before = study.Rows.Select(r => LearningRowFixtures.Copy(r)).ToArray();

        Assert.ThrowsExactly<InvalidDataException>(() => StudyWorkflow.Validate(study));

        LearningRowFixtures.Unchanged(before, study.Rows);
        LearningRowFixtures.Unchanged(original, fixture.Study.Rows);
        Assert.AreEqual(608, fixture.Study.Rows.Length);
        StudyWorkflow.Validate(fixture.Study);
    }

    [TestMethod]
    public void Train_FiveArmsUseMatchedIntactTrainingOnlySubsetsAndPublishFreeze()
    {
        var before = fixture.Study.Rows.Select(r => LearningRowFixtures.Copy(r)).ToArray();

        var freeze = fixture.Training;

        Assert.AreEqual(1, freeze.Version);
        Assert.AreEqual("complete", freeze.Status);
        Assert.AreEqual(StudyFixture.Identity, freeze.FeatureFingerprint);
        Assert.AreEqual(fixture.Study.DatasetManifestSha256, freeze.DatasetManifestSha256);
        Assert.AreEqual(fixture.Study.SplitSha256, freeze.SplitSha256);
        Assert.AreEqual(fixture.Study.QuestionsSha256, freeze.QuestionsSha256);
        Assert.AreEqual(FeatureContract.Conversion, freeze.Conversion);
        Assert.AreEqual(15, freeze.Arms.Length);
        CollectionAssert.AreEqual(StudyFixture.Targets, freeze.Arms.Select(a => a.TargetRows).Distinct().Order().ToArray());
        foreach (var (target, expectedCount) in new[] { (100, 102), (500, 501), (600, 600) })
        {
            var subset = fixture.ExpectedSubset(target);
            Assert.AreEqual(expectedCount, subset.Length);
            Assert.IsTrue(subset.Any(r => r.Label) && subset.Any(r => !r.Label));
            double prior = (double)subset.Count(r => r.Label) / subset.Length;
            if (target == 600)
                Assert.AreEqual(.33, prior, "Prior must not be manufactured from balanced validation/holdout labels.");
            var arms = freeze.Arms.Where(a => a.TargetRows == target).ToArray();
            CollectionAssert.AreEqual(StudyFixture.ArmNames, arms.Select(a => a.Arm).Order(StringComparer.Ordinal).ToArray());
            foreach (var arm in arms)
            {
                Assert.AreEqual(expectedCount, arm.ActualRows);
                CollectionAssert.AreEqual(subset.Select(r => r.RowId).ToArray(), arm.TrainingRowIds);
                Assert.IsTrue(arm.TrainingRowIds.All(id => id < 1000));
                Assert.AreEqual(expectedCount / 3, subset.Select(r => r.GroupId).Distinct().Count());
                foreach (var group in subset.GroupBy(r => r.GroupId)) Assert.AreEqual(3, group.Count());
                if (arm.Arm is "prior" or "direct")
                {
                    Assert.IsNull(arm.ModelReceiptFile);
                    Assert.IsNull(arm.ModelReceiptSha256);
                    if (arm.Arm == "prior")
                    {
                        Assert.AreEqual(prior, arm.Prior!.Value);
                        Assert.AreEqual(prior, arm.Threshold);
                        Assert.AreEqual(Math.BitIncrement(1d), arm.BudgetThreshold);
                    }
                    else
                    {
                        Assert.IsNull(arm.Prior);
                        Assert.AreEqual(.8, arm.Threshold);
                        Assert.AreEqual(.8, arm.BudgetThreshold);
                    }
                }
                else
                {
                    Assert.IsNull(arm.Prior);
                    Assert.AreEqual($"{arm.Arm}-{target}.receipt.json", arm.ModelReceiptFile);
                    Assert.AreEqual(ArtifactExpectations.HashFile(Path.Combine(fixture.TrainingOutput, arm.ModelReceiptFile!)),
                        arm.ModelReceiptSha256);
                    var receipt = StudyFixture.Receipt(fixture.TrainingOutput, arm);
                    Assert.AreEqual(arm.Arm, receipt.Arm);
                    Assert.AreEqual(target, receipt.TargetRows);
                    Assert.AreEqual(expectedCount, receipt.ActualRows);
                    Assert.AreEqual(arm.Threshold, receipt.Threshold);
                    Assert.AreEqual(arm.BudgetThreshold, receipt.BudgetThreshold);
                    Assert.AreEqual(4, receipt.Validation.Rows);
                    Assert.AreEqual(2, receipt.Validation.Ham);
                    Assert.AreEqual(2, receipt.Validation.Spam);
                    Assert.AreEqual(ArtifactExpectations.HashFile(Path.Combine(fixture.TrainingOutput, receipt.ModelFile)), receipt.ModelSha256);
                    Assert.IsTrue(receipt.ModelBytes > 0);
                }
            }
        }
        var published = JsonSerializer.Deserialize<TrainingFreeze>(File.ReadAllBytes(fixture.FreezePath), StudyFixture.Json)!;
        Assert.AreEqual("complete", published.Status);
        Assert.AreEqual(15, published.Arms.Length);
        Assert.AreEqual(freeze.SplitSha256, published.SplitSha256);
        foreach (var arm in freeze.Arms)
        {
            var saved = published.Arms.Single(a => a.Arm == arm.Arm && a.TargetRows == arm.TargetRows);
            CollectionAssert.AreEqual(arm.TrainingRowIds, saved.TrainingRowIds);
            Assert.AreEqual(arm.Threshold, saved.Threshold);
            Assert.AreEqual(arm.BudgetThreshold, saved.BudgetThreshold);
            Assert.AreEqual(arm.ModelReceiptSha256, saved.ModelReceiptSha256);
        }
        Assert.AreEqual(19, Directory.GetFiles(fixture.TrainingOutput).Length);
        Assert.IsTrue(freeze.Selection.Contains("No holdout selection.", StringComparison.Ordinal));
        LearningRowFixtures.Unchanged(before, fixture.Study.Rows);
    }

    [TestMethod]
    public void Train_HoldoutChangesDoNotAffectModelsOrFrozenThresholds()
    {
        using var temp = new TempDirectory();
        var original = fixture.Training;
        var files = StudyFixture.Snapshot(fixture.TrainingOutput);
        var modified = StudyFixture.Copy(fixture.Study);
        foreach (var row in modified.Rows.Where(r => r.RowId >= 2000))
        {
            row.Label = !row.Label;
            row.Text = "different authored holdout never fit";
            row.Semantic = row.Semantic.Select(p => 1 - p).ToArray();
            row.SpamBaseline = 1 - row.SpamBaseline;
        }
        var splitRows = modified.Split.Rows.Select(r => r.Split == "holdout" ? r with { Label = !r.Label } : r).ToArray();
        modified = modified with { Split = modified.Split with { Rows = splitRows } };
        modified = modified with { SplitSha256 = StudyFixture.SplitHash(modified.Split) };
        string output = temp.FilePath("altered-holdout-training");

        var second = StudyWorkflow.Train(modified, output);

        Assert.AreNotEqual(original.SplitSha256, second.SplitSha256);
        Assert.AreEqual(15, second.Arms.Length);
        var validation = fixture.Study.Rows.Where(r => r.RowId is >= 1000 and < 2000).ToArray();
        foreach (var first in original.Arms)
        {
            var other = second.Arms.Single(a => a.Arm == first.Arm && a.TargetRows == first.TargetRows);
            CollectionAssert.AreEqual(first.TrainingRowIds, other.TrainingRowIds);
            Assert.AreEqual(first.Prior, other.Prior);
            Assert.AreEqual(first.Threshold, other.Threshold);
            Assert.AreEqual(first.BudgetThreshold, other.BudgetThreshold);
            Assert.AreEqual(first.ActualRows, other.ActualRows);
            if (first.ModelReceiptFile is null) continue;
            var receipt = StudyFixture.Receipt(fixture.TrainingOutput, first);
            var otherReceipt = StudyFixture.Receipt(output, other);
            Assert.AreEqual(receipt.L2, otherReceipt.L2);
            var model = PredictorTraining.Load(Path.Combine(fixture.TrainingOutput, receipt.ModelFile), receipt, StudyFixture.Identity);
            var otherModel = PredictorTraining.Load(Path.Combine(output, otherReceipt.ModelFile), otherReceipt, StudyFixture.Identity);
            ModelExpectations.Replay(PredictorTraining.Predict(model, validation), PredictorTraining.Predict(otherModel, validation));
        }
        StudyFixture.Unchanged(fixture.TrainingOutput, files);
        Assert.AreEqual(2, fixture.Study.Rows.Where(r => r.RowId >= 2000).Count(r => r.Label));
        Assert.AreEqual(198, fixture.Study.Rows.Where(r => r.RowId < 1000).Count(r => r.Label));
    }

    [TestMethod]
    public void Train_ExistingOutputRetainsFrozenArtifacts()
    {
        _ = fixture.Training;
        var before = StudyFixture.Snapshot(fixture.TrainingOutput);

        var error = Assert.ThrowsExactly<IOException>(() => StudyWorkflow.Train(fixture.Study, fixture.TrainingOutput));

        Assert.AreEqual("Training output must be empty; models/freezes are immutable.", error.Message);
        StudyFixture.Unchanged(fixture.TrainingOutput, before);
        Assert.AreEqual(19, Directory.GetFiles(fixture.TrainingOutput).Length);
        Assert.IsFalse(Directory.GetFiles(fixture.TrainingOutput).Any(p => p.EndsWith(".partial", StringComparison.Ordinal)));
    }

    [TestMethod]
    public void Evaluate_AllArmsUseFrozenModelsThresholdsAndSavedPredictionBootstrap()
    {
        _ = fixture.Training;
        var models = StudyFixture.Snapshot(fixture.TrainingOutput);
        var inputBefore = fixture.Study.Rows.Select(r => LearningRowFixtures.Copy(r)).ToArray();

        var report = fixture.Evaluation;

        Assert.AreEqual(1, report.Version);
        Assert.AreEqual("complete", report.Status);
        Assert.AreEqual(ArtifactExpectations.HashFile(fixture.FreezePath), report.TrainingFreezeSha256);
        Assert.AreEqual(StudyFixture.Identity, report.FeatureFingerprint);
        Assert.AreEqual(fixture.Study.DatasetManifestSha256, report.DatasetManifestSha256);
        Assert.AreEqual(fixture.Study.SplitSha256, report.SplitSha256);
        Assert.AreEqual(15, report.Arms.Length);
        Assert.IsNull(report.Extraction);
        var holdout = fixture.Study.Rows.Where(r => r.RowId >= 2000).ToArray();
        foreach (var arm in fixture.Training.Arms)
        {
            var result = report.Arms.Single(a => a.Arm == arm.Arm && a.TargetRows == arm.TargetRows);
            Assert.AreEqual(arm.ActualRows, result.ActualTrainingRows);
            Assert.AreEqual($"{arm.Arm}-{arm.TargetRows}.holdout.json", result.SavedPredictionsFile);
            string path = Path.Combine(fixture.EvaluationOutput, result.SavedPredictionsFile);
            var saved = JsonSerializer.Deserialize<Prediction[]>(File.ReadAllBytes(path), StudyFixture.Json)!;
            ModelExpectations.Predictions(holdout, saved);
            Assert.AreEqual(ArtifactExpectations.HashFile(path), result.SavedPredictionsSha256);
            Assert.AreEqual(arm.Threshold, result.Holdout.Threshold);
            Assert.AreEqual(arm.BudgetThreshold, result.AtValidationFprBudget.Threshold);
            ModelExpectations.ValidationMetrics(saved, arm.Threshold, result.Holdout);
            ModelExpectations.ValidationMetrics(saved, arm.BudgetThreshold, result.AtValidationFprBudget);
            if (arm.Arm == "prior")
            {
                double prior = arm.Prior!.Value;
                CollectionAssert.AreEqual(Enumerable.Repeat(prior, 4).ToArray(), saved.Select(p => p.Probability).ToArray());
                Assert.AreEqual(2, result.Holdout.TruePositive);
                Assert.AreEqual(2, result.Holdout.FalsePositive);
                Assert.AreEqual(.5, result.Holdout.Auprc);
                Assert.AreEqual(.5, result.Holdout.RocAuc);
                Assert.AreEqual(0, result.AtValidationFprBudget.TruePositive);
                Assert.AreEqual(0, result.AtValidationFprBudget.FalsePositive);
            }
            else if (arm.Arm == "direct")
            {
                CollectionAssert.AreEqual(new double[] { .3, .3, .7, .7 }, saved.Select(p => p.Probability).ToArray());
                // Holdout would prefer .7; the frozen validation threshold remains .8.
                Assert.AreEqual(.8, result.Holdout.Threshold);
                Assert.AreEqual(0, result.Holdout.TruePositive);
                Assert.AreEqual(0, result.Holdout.FalsePositive);
                Assert.AreEqual(2, result.Holdout.TrueNegative);
                Assert.AreEqual(2, result.Holdout.FalseNegative);
                Assert.IsNull(result.Holdout.Precision);
                Assert.AreEqual(0d, result.Holdout.Recall);
                Assert.AreEqual(1d, result.Holdout.Auprc);
                Assert.AreEqual(1d, result.Holdout.RocAuc);
                EvaluationGoldens.Near(-Math.Log(.7), result.Holdout.LogLoss);
                EvaluationGoldens.Near(.09, result.Holdout.Brier);
            }
            else
            {
                var receipt = StudyFixture.Receipt(fixture.TrainingOutput, arm);
                var model = PredictorTraining.Load(Path.Combine(fixture.TrainingOutput, receipt.ModelFile), receipt, StudyFixture.Identity);
                ModelExpectations.Replay(PredictorTraining.Predict(model, holdout), saved);
                Assert.AreEqual(receipt.Threshold, result.Holdout.Threshold);
                Assert.AreEqual(receipt.BudgetThreshold, result.AtValidationFprBudget.Threshold);
            }
            StudyFixture.ReportEqual(PredictorEvaluation.Bootstrap(saved, arm.Threshold), result.Uncertainty);
            var text = fixture.Training.Arms.Single(a => a.Arm == "text" && a.TargetRows == arm.TargetRows);
            var reference = JsonSerializer.Deserialize<Prediction[]>(
                File.ReadAllBytes(Path.Combine(fixture.EvaluationOutput, $"text-{arm.TargetRows}.holdout.json")), StudyFixture.Json)!;
            StudyFixture.ReportEqual(PredictorEvaluation.Bootstrap(saved, arm.Threshold, reference, text.Threshold),
                result.PairedDeltaFromText);
            Assert.AreEqual(1000, result.Uncertainty.Resamples);
            Assert.AreEqual(505, result.Uncertainty.Intervals.Single(i => i.Metric == "auprc").Defined);
            Assert.AreEqual(495, result.Uncertainty.Intervals.Single(i => i.Metric == "rocAuc").Undefined);
            foreach (var interval in result.PairedDeltaFromText.Intervals)
                Assert.AreEqual(1000, interval.Defined + interval.Undefined);
            if (arm.Arm == "text")
                foreach (var interval in result.PairedDeltaFromText.Intervals)
                {
                    Assert.AreEqual(0d, interval.Lower);
                    Assert.AreEqual(0d, interval.Upper);
                }
        }
        var published = JsonSerializer.Deserialize<StudyEvaluation>(
            File.ReadAllBytes(Path.Combine(fixture.EvaluationOutput, "evaluation.json")), StudyFixture.Json)!;
        Assert.AreEqual("complete", published.Status);
        Assert.AreEqual(report.TrainingFreezeSha256, published.TrainingFreezeSha256);
        Assert.AreEqual(15, published.Arms.Length);
        foreach (var arm in report.Arms)
        {
            var persisted = published.Arms.Single(a => a.Arm == arm.Arm && a.TargetRows == arm.TargetRows);
            Assert.AreEqual(arm.Holdout, persisted.Holdout);
            Assert.AreEqual(arm.AtValidationFprBudget, persisted.AtValidationFprBudget);
            Assert.AreEqual(arm.SavedPredictionsSha256, persisted.SavedPredictionsSha256);
            StudyFixture.ReportEqual(arm.Uncertainty, persisted.Uncertainty);
            StudyFixture.ReportEqual(arm.PairedDeltaFromText, persisted.PairedDeltaFromText);
        }
        Assert.AreEqual(16, Directory.GetFiles(fixture.EvaluationOutput).Length);
        Assert.IsFalse(Directory.GetFiles(fixture.EvaluationOutput).Any(p => p.EndsWith(".partial", StringComparison.Ordinal)));
        StudyFixture.Unchanged(fixture.TrainingOutput, models);
        LearningRowFixtures.Unchanged(inputBefore, fixture.Study.Rows);
        // Source assurance: Evaluate's second loop reads Prediction[] artifacts for
        // Bootstrap; it contains no model Load/Predict. This is not a dynamic counter.
    }

    [TestMethod]
    public void Evaluate_FrozenBaselineThresholdsAreNotRetunedAndTextPairUsesOwnThreshold()
    {
        using var temp = new TempDirectory();
        string root = fixture.CopyTraining(temp);
        var freeze = StudyFixture.Copy(fixture.Training);
        freeze = freeze with
        {
            Arms = freeze.Arms.Select(a => a.Arm == "direct"
                ? a with { Threshold = 0, BudgetThreshold = Math.BitIncrement(1d) } : a).ToArray()
        };
        StudyFixture.WriteFreeze(root, freeze);
        var original = StudyFixture.Snapshot(fixture.TrainingOutput);
        string output = temp.FilePath("deliberately-frozen-thresholds");

        var report = StudyWorkflow.Evaluate(fixture.Study, Path.Combine(root, "training.freeze.json"), output);

        foreach (int target in StudyFixture.Targets)
        {
            var arm = report.Arms.Single(a => a.Arm == "direct" && a.TargetRows == target);
            Assert.AreEqual(0d, arm.Holdout.Threshold);
            Assert.AreEqual(2, arm.Holdout.TruePositive);
            Assert.AreEqual(2, arm.Holdout.FalsePositive);
            Assert.AreEqual(1d, arm.Holdout.Recall);
            Assert.AreEqual(1d, arm.Holdout.FalsePositiveRate);
            Assert.AreEqual(Math.BitIncrement(1d), arm.AtValidationFprBudget.Threshold);
            Assert.AreEqual(0, arm.AtValidationFprBudget.TruePositive);
            Assert.AreEqual(2, arm.AtValidationFprBudget.FalseNegative);
            var saved = JsonSerializer.Deserialize<Prediction[]>(File.ReadAllBytes(Path.Combine(output, arm.SavedPredictionsFile)), StudyFixture.Json)!;
            var textSaved = JsonSerializer.Deserialize<Prediction[]>(File.ReadAllBytes(Path.Combine(output, $"text-{target}.holdout.json")), StudyFixture.Json)!;
            double textThreshold = freeze.Arms.Single(a => a.Arm == "text" && a.TargetRows == target).Threshold;
            Assert.IsTrue(textThreshold > 0);
            var right = PredictorEvaluation.Bootstrap(saved, 0, textSaved, textThreshold);
            var wrong = PredictorEvaluation.Bootstrap(saved, 0, textSaved, 0);
            StudyFixture.ReportEqual(right, arm.PairedDeltaFromText);
            Assert.IsFalse(right.Intervals.SequenceEqual(wrong.Intervals),
                "The fixture must distinguish text's frozen threshold from incorrectly reusing direct's threshold.");
            CollectionAssert.AreEqual(new double[] { .3, .3, .7, .7 }, saved.Select(p => p.Probability).ToArray());
        }
        StudyFixture.Unchanged(fixture.TrainingOutput, original);
        Assert.AreEqual(16, Directory.GetFiles(output).Length);
    }

    [TestMethod]
    [DataRow("null")]
    [DataRow("empty")]
    [DataRow("missing-arm")]
    [DataRow("unknown-arm")]
    [DataRow("duplicate-arm")]
    [DataRow("missing-curve")]
    [DataRow("missing-curve-500")]
    [DataRow("missing-curve-full")]
    [DataRow("extra-target")]
    [DataRow("duplicate-curve")]
    [DataRow("wrong-full-target")]
    public void Evaluate_ExactFiveArmsAndDeclaredTargetsAreRequired(string mutation)
    {
        using var temp = new TempDirectory();
        string root = fixture.CopyTraining(temp);
        var freeze = StudyFixture.Copy(fixture.Training);
        var arms = freeze.Arms;
        switch (mutation)
        {
            case "null": arms = null!; break;
            case "empty": arms = []; break;
            case "missing-arm": arms = arms.Where(a => !(a.TargetRows == 100 && a.Arm == "semantic")).ToArray(); break;
            case "unknown-arm": arms[0] = arms[0] with { Arm = "unknown" }; break;
            case "duplicate-arm": arms = [.. arms, arms[0]]; break;
            case "missing-curve": arms = arms.Where(a => a.TargetRows != 100).ToArray(); break;
            case "missing-curve-500": arms = arms.Where(a => a.TargetRows != 500).ToArray(); break;
            case "missing-curve-full": arms = arms.Where(a => a.TargetRows != 600).ToArray(); break;
            case "extra-target": arms = [.. arms, arms[0] with { TargetRows = 999 }]; break;
            case "duplicate-curve": arms = [.. arms, .. arms.Where(a => a.TargetRows == 500)]; break;
            case "wrong-full-target": arms = arms.Select(a => a.TargetRows == 600 ? a with { TargetRows = 601 } : a).ToArray(); break;
        }
        StudyFixture.WriteFreeze(root, freeze with { Arms = arms });
        string output = temp.FilePath("rejected");
        Assert.IsFalse(Directory.Exists(output), "Each malformed freeze starts without evaluation output.");

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            StudyWorkflow.Evaluate(fixture.Study, Path.Combine(root, "training.freeze.json"), output));

        Assert.AreEqual("Expected all declared learning curves with exactly five frozen arms each.", error.Message);
        Assert.IsFalse(Directory.Exists(output), "Reject an incomplete freeze before creating any evaluation output.");
        Assert.AreEqual(15, fixture.Training.Arms.Length);
        CollectionAssert.AreEqual(StudyFixture.ArmNames, fixture.Training.Arms.Where(a => a.TargetRows == 100)
            .Select(a => a.Arm).Order(StringComparer.Ordinal).ToArray());
    }

    [TestMethod]
    [DataRow("mismatched-order")]
    [DataRow("mismatched-subset")]
    [DataRow("actual-count")]
    [DataRow("duplicate-id")]
    [DataRow("validation-id")]
    [DataRow("holdout-id")]
    [DataRow("unknown-id")]
    [DataRow("split-group")]
    public void Evaluate_TrainingOnlyMatchedWholeGroupsAreRequired(string mutation)
    {
        using var temp = new TempDirectory();
        string root = fixture.CopyTraining(temp);
        var freeze = StudyFixture.Copy(fixture.Training);
        freeze = freeze with { Arms = freeze.Arms.Select(a =>
        {
            if (a.TargetRows != 100) return a;
            if (mutation == "mismatched-order")
                return a.Arm == "direct" ? a with { TrainingRowIds = a.TrainingRowIds.Reverse().ToArray() } : a;
            if (mutation == "mismatched-subset")
                return a.Arm == "direct" ? a with { TrainingRowIds = a.TrainingRowIds[3..], ActualRows = a.ActualRows - 3 } : a;
            if (mutation == "actual-count")
                return a.Arm == "prior" ? a with { ActualRows = a.ActualRows + 1 } : a;
            long[] ids = mutation == "split-group" ? a.TrainingRowIds[1..] :
                [.. a.TrainingRowIds, mutation == "duplicate-id" ? a.TrainingRowIds[0] :
                    mutation == "validation-id" ? 1001 : mutation == "holdout-id" ? 2001 : 9999];
            return a with { TrainingRowIds = ids, ActualRows = ids.Length };
        }).ToArray() };
        StudyFixture.WriteFreeze(root, freeze);
        var originals = StudyFixture.Snapshot(fixture.TrainingOutput);
        string output = temp.FilePath("rejected");

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            StudyWorkflow.Evaluate(fixture.Study, Path.Combine(root, "training.freeze.json"), output));

        Assert.AreEqual(mutation == "split-group" ? "Frozen learning subset splits a duplicate group." :
            "Frozen arms do not use matching training-only source rows.", error.Message);
        Assert.IsFalse(File.Exists(Path.Combine(output, "evaluation.json")));
        StudyFixture.Unchanged(fixture.TrainingOutput, originals);
        Assert.AreEqual(102, fixture.Training.Arms.Single(a => a.TargetRows == 100 && a.Arm == "text").TrainingRowIds.Length);
    }

    [TestMethod]
    [DataRow("identity")]
    [DataRow("identity-case")]
    [DataRow("identity-space")]
    [DataRow("identity-blank")]
    [DataRow("version")]
    [DataRow("status")]
    [DataRow("dataset")]
    [DataRow("split")]
    [DataRow("questions")]
    [DataRow("conversion")]
    public void Evaluate_FreezeIdentityAndProvenanceMutationsAreRejected(string mutation)
    {
        using var temp = new TempDirectory();
        string root = fixture.CopyTraining(temp);
        var original = fixture.Training;
        var freeze = StudyFixture.Copy(original);
        freeze = mutation switch
        {
            "identity" => freeze with { FeatureFingerprint = "different-contract" },
            "identity-case" => freeze with { FeatureFingerprint = freeze.FeatureFingerprint.ToUpperInvariant() },
            "identity-space" => freeze with { FeatureFingerprint = " " + freeze.FeatureFingerprint },
            "identity-blank" => freeze with { FeatureFingerprint = "" },
            "version" => freeze with { Version = 2 },
            "status" => freeze with { Status = "partial" },
            "dataset" => freeze with { DatasetManifestSha256 = new string('0', 64) },
            "split" => freeze with { SplitSha256 = new string('0', 64) },
            "questions" => freeze with { QuestionsSha256 = new string('0', 64) },
            "conversion" => freeze with { Conversion = "unvalidated" },
            _ => throw new ArgumentOutOfRangeException(nameof(mutation))
        };
        StudyFixture.WriteFreeze(root, freeze);
        string output = temp.FilePath("rejected");

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            StudyWorkflow.Evaluate(fixture.Study, Path.Combine(root, "training.freeze.json"), output));

        Assert.AreEqual(mutation.StartsWith("identity", StringComparison.Ordinal)
            ? "External producer feature-contract identity mismatch." : "Training freeze does not match imported study.", error.Message);
        Assert.IsFalse(Directory.Exists(output));
        Assert.AreEqual("complete", original.Status);
        Assert.AreEqual(StudyFixture.Identity, original.FeatureFingerprint);
    }

    [TestMethod]
    [DataRow("missing-file")]
    [DataRow("traversal")]
    [DataRow("rooted")]
    [DataRow("hash")]
    [DataRow("arm")]
    [DataRow("target")]
    [DataRow("actual")]
    [DataRow("threshold")]
    [DataRow("budget")]
    [DataRow("split")]
    [DataRow("dataset")]
    [DataRow("questions")]
    [DataRow("model-path")]
    [DataRow("model-length")]
    [DataRow("model-same-length")]
    [DataRow("conversion")]
    [DataRow("projection")]
    public void Evaluate_ModelReceiptHashAndFrozenFieldsMustMatch(string mutation)
    {
        using var temp = new TempDirectory();
        string root = fixture.CopyTraining(temp);
        var freeze = StudyFixture.Copy(fixture.Training);
        int index = Array.FindIndex(freeze.Arms, a => a.TargetRows == 100 && a.Arm == "text");
        var arm = freeze.Arms[index];
        var receipt = StudyFixture.Receipt(root, arm);
        string receiptPath = Path.Combine(root, arm.ModelReceiptFile!);
        string expectedMessage;
        if (mutation is "missing-file" or "traversal" or "rooted" or "hash")
        {
            arm = mutation switch
            {
                "missing-file" => arm with { ModelReceiptFile = null },
                "traversal" => arm with { ModelReceiptFile = "../text-100.receipt.json" },
                "rooted" => arm with { ModelReceiptFile = receiptPath },
                _ => arm with { ModelReceiptSha256 = new string('0', 64) }
            };
            expectedMessage = mutation == "missing-file" ? "Missing model receipt." :
                mutation == "hash" ? "SHA-256 mismatch: " + receiptPath : "Receipt must be a sibling filename.";
        }
        else if (mutation.StartsWith("model-", StringComparison.Ordinal) && mutation != "model-path")
        {
            string path = Path.Combine(root, receipt.ModelFile);
            byte[] bytes = File.ReadAllBytes(path);
            if (mutation == "model-length") bytes = [.. bytes, 0x21];
            else bytes[bytes.Length / 2] ^= 1;
            File.WriteAllBytes(path, bytes);
            expectedMessage = mutation == "model-length" ? "Saved predictor feature projection/receipt mismatch." :
                "SHA-256 mismatch: " + path;
        }
        else
        {
            receipt = mutation switch
            {
                "arm" => receipt with { Arm = "semantic" },
                "target" => receipt with { TargetRows = 101 },
                "actual" => receipt with { ActualRows = receipt.ActualRows + 1 },
                "threshold" => receipt with { Threshold = Math.BitIncrement(receipt.Threshold) },
                "budget" => receipt with { BudgetThreshold = Math.BitIncrement(receipt.BudgetThreshold) },
                "split" => receipt with { SplitSha256 = new string('0', 64) },
                "dataset" => receipt with { DatasetManifestSha256 = new string('0', 64) },
                "questions" => receipt with { QuestionsSha256 = new string('0', 64) },
                "model-path" => receipt with { ModelFile = "../" + receipt.ModelFile },
                "conversion" => receipt with { Conversion = "unvalidated" },
                "projection" => receipt with { Projection = receipt.Projection[..^1] },
                _ => throw new ArgumentOutOfRangeException(nameof(mutation))
            };
            File.WriteAllBytes(receiptPath, JsonSerializer.SerializeToUtf8Bytes(receipt, StudyFixture.Json));
            arm = arm with { ModelReceiptSha256 = ArtifactExpectations.HashFile(receiptPath) };
            expectedMessage = mutation is "conversion" or "projection"
                ? "Saved predictor feature projection/receipt mismatch." : "Frozen model receipt mismatch.";
        }
        freeze.Arms[index] = arm;
        StudyFixture.WriteFreeze(root, freeze);
        var originals = StudyFixture.Snapshot(fixture.TrainingOutput);
        string output = temp.FilePath("rejected");

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            StudyWorkflow.Evaluate(fixture.Study, Path.Combine(root, "training.freeze.json"), output));

        Assert.AreEqual(expectedMessage, error.Message);
        Assert.IsFalse(File.Exists(Path.Combine(output, "evaluation.json")));
        StudyFixture.Unchanged(fixture.TrainingOutput, originals);
        Assert.AreEqual(19, Directory.GetFiles(fixture.TrainingOutput).Length);
    }

    [TestMethod]
    [DataRow("missing")]
    [DataRow("zero")]
    [DataRow("one")]
    [DataRow("negative")]
    [DataRow("above-one")]
    public void Evaluate_InvalidFrozenTrainingPrevalenceIsRejected(string mutation)
    {
        using var temp = new TempDirectory();
        string root = fixture.CopyTraining(temp);
        var freeze = StudyFixture.Copy(fixture.Training);
        double? prior = mutation switch
        {
            "missing" => null,
            "zero" => 0, "one" => 1, "negative" => -.1, "above-one" => 1.1,
            _ => throw new ArgumentOutOfRangeException(nameof(mutation))
        };
        freeze = freeze with { Arms = freeze.Arms.Select(a => a.Arm == "prior" && a.TargetRows == 100
            ? a with { Prior = prior } : a).ToArray() };
        StudyFixture.WriteFreeze(root, freeze);
        string output = temp.FilePath("rejected");

        var failure = Assert.ThrowsExactly<InvalidDataException>(() =>
            StudyWorkflow.Evaluate(fixture.Study, Path.Combine(root, "training.freeze.json"), output));

        Assert.AreEqual(mutation == "missing" ? "Missing frozen training prevalence." :
            "Frozen training prevalence must lie strictly between zero and one.", failure.Message);
        Assert.IsFalse(File.Exists(Path.Combine(output, "evaluation.json")));
        Assert.IsTrue(fixture.Training.Arms.Where(a => a.Arm == "prior").All(a => a.Prior is > 0 and < 1));
    }

    [TestMethod]
    public void Evaluate_OutputImmutableAfterFirstPublication()
    {
        _ = fixture.Evaluation;
        var before = StudyFixture.Snapshot(fixture.EvaluationOutput);
        var training = StudyFixture.Snapshot(fixture.TrainingOutput);

        var error = Assert.ThrowsExactly<IOException>(() =>
            StudyWorkflow.Evaluate(fixture.Study, fixture.FreezePath, fixture.EvaluationOutput));

        Assert.AreEqual("Evaluation output must be empty; saved holdout predictions are immutable.", error.Message);
        StudyFixture.Unchanged(fixture.EvaluationOutput, before);
        StudyFixture.Unchanged(fixture.TrainingOutput, training);
        Assert.AreEqual(16, Directory.GetFiles(fixture.EvaluationOutput).Length);
        Assert.IsFalse(Directory.GetFiles(fixture.EvaluationOutput).Any(p => p.EndsWith(".partial", StringComparison.Ordinal)));
    }
}

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DecisionArrowPredictor.Tests;

[TestClass]
public sealed class PredictorTrainingTests
{
    private static TinyModelFixture models = null!;

    [ClassInitialize]
    public static void Initialize(TestContext _) => models = new TinyModelFixture();

    [ClassCleanup]
    public static void Cleanup() => models.Dispose();

    [TestMethod]
    public void Subset_Targets100500FullPreserveNestedIntactOriginalGroups()
    {
        var source = LearningRowFixtures.Grouped600();
        var before = source.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        var shuffled = source.OrderBy(r => r.RowId % 17).ThenByDescending(r => r.RowId).ToArray();
        long[] groups = source.Select(r => r.GroupId).Distinct()
            .OrderBy(g => SplitExpectations.OrderKey(g, 1), StringComparer.Ordinal).ToArray();
        LearningRow[]? previous = null;

        foreach (var (target, actualCount, groupCount) in new[] { (100, 102, 34), (500, 501, 167), (600, 600, 200) })
        {
            var selected = PredictorTraining.Subset(source, target);
            var replayed = PredictorTraining.Subset(shuffled, target);
            var expectedIds = source.Where(r => groups.Take(groupCount).Contains(r.GroupId))
                .Select(r => r.RowId).Order().ToArray();
            Assert.AreEqual(actualCount, selected.Length);
            Assert.AreEqual(groupCount, selected.Select(r => r.GroupId).Distinct().Count());
            Assert.AreEqual(groupCount, selected.Count(r => r.Label));
            Assert.AreEqual(2 * groupCount, selected.Count(r => !r.Label));
            CollectionAssert.AreEqual(expectedIds, selected.Select(r => r.RowId).ToArray());
            CollectionAssert.AreEqual(expectedIds, replayed.Select(r => r.RowId).ToArray());
            CollectionAssert.AreEqual(selected.Select(r => (r.RowId, r.GroupId, r.Label)).ToArray(),
                replayed.Select(r => (r.RowId, r.GroupId, r.Label)).ToArray());
            foreach (var row in selected)
            {
                Assert.AreSame(source.Single(r => r.RowId == row.RowId), row);
                Assert.AreEqual(3, selected.Count(r => r.GroupId == row.GroupId));
            }
            if (previous is not null)
                CollectionAssert.IsSubsetOf(previous.Select(r => r.RowId).ToArray(), expectedIds);
            previous = selected;
        }
        LearningRowFixtures.Unchanged(before, source);
    }

    [TestMethod]
    [DataRow(600)]
    [DataRow(1000)]
    public void Subset_FullAndOversizedTargetsReturnAllOriginalRowsInSourceIdOrder(int target)
    {
        var source = LearningRowFixtures.Grouped600().Reverse().ToArray();
        var before = source.Select(r => LearningRowFixtures.Copy(r)).ToArray();

        var selected = PredictorTraining.Subset(source, target);

        CollectionAssert.AreEqual(Enumerable.Range(1, 600).Select(i => (long)i).ToArray(),
            selected.Select(r => r.RowId).ToArray());
        Assert.AreEqual(600, selected.Length);
        Assert.AreEqual(200, selected.Count(r => r.Label));
        Assert.AreEqual(200, selected.Select(r => r.GroupId).Distinct().Count());
        foreach (var row in selected) Assert.AreSame(source.Single(r => r.RowId == row.RowId), row);
        LearningRowFixtures.Unchanged(before, source);
    }

    [TestMethod]
    public void Subset_DelayedSecondClassCausesHonestOvershoot()
    {
        long[] groupOrder = new long[] { 10, 20, 30, 40 }
            .OrderBy(g => SplitExpectations.OrderKey(g, 1), StringComparer.Ordinal).ToArray();
        var source = groupOrder.SelectMany((g, i) => Enumerable.Range(1, 3).Select(j => new LearningRow
        {
            RowId = g + j, GroupId = g, Label = i == 3, Text = $"authored pure group {g}",
            Semantic = Enumerable.Repeat(i == 3 ? .8f : .2f, 10).ToArray()
        })).Reverse().ToArray();
        var before = source.Select(r => LearningRowFixtures.Copy(r)).ToArray();

        var selected = PredictorTraining.Subset(source, 4);

        Assert.AreEqual(12, selected.Length, "Meeting target alone must not end selection before the second class.");
        Assert.AreEqual(9, selected.Count(r => !r.Label));
        Assert.AreEqual(3, selected.Count(r => r.Label));
        CollectionAssert.AreEqual(source.Select(r => r.RowId).Order().ToArray(), selected.Select(r => r.RowId).ToArray());
        foreach (var group in selected.GroupBy(r => r.GroupId)) Assert.AreEqual(3, group.Count());
        foreach (var row in selected) Assert.AreSame(source.Single(r => r.RowId == row.RowId), row);
        LearningRowFixtures.Unchanged(before, source);
    }

    [TestMethod]
    [DataRow("zero")]
    [DataRow("negative")]
    [DataRow("empty")]
    [DataRow("ham-only")]
    [DataRow("spam-only")]
    [DataRow("duplicate-id")]
    public void Subset_InvalidTargetClassSupportOrDuplicateIdsIsRejected(string mutation)
    {
        var source = LearningRowFixtures.Training();
        int target = mutation == "zero" ? 0 : mutation == "negative" ? -1 : 5;
        if (mutation == "empty") source = [];
        if (mutation == "ham-only") source = source.Where(r => !r.Label).ToArray();
        if (mutation == "spam-only") source = source.Where(r => r.Label).ToArray();
        if (mutation == "duplicate-id") source[1].RowId = source[0].RowId;
        var before = source.Select(r => LearningRowFixtures.Copy(r)).ToArray();

        var error = Assert.ThrowsExactly<InvalidDataException>(() => PredictorTraining.Subset(source, target));

        Assert.AreEqual("Learning subsets need a positive target, unique rows, and both classes.", error.Message);
        LearningRowFixtures.Unchanged(before, source);
    }

    [TestMethod]
    public void Fit_TextArmTrainsOfflineAndPublishesVerifiedReceipt() => AssertReceipt("text");

    [TestMethod]
    public void Fit_SemanticArmTrainsOfflineWithExactlyTenFeatures()
    {
        AssertReceipt("semantic");
        var view = Transform("semantic", models.Validation);
        Assert.AreEqual(10, ((VectorDataViewType)view.Schema["Features"].Type).Size);
        var features = Vectors(view, "Features");
        for (int i = 0; i < features.Length; i++)
            CollectionAssert.AreEqual(models.Validation[i].Semantic, features[i]);
        CollectionAssert.AreEqual(QuestionFixtures.Projection, models.Arm("semantic").Receipt.Projection);
        Assert.IsFalse(models.Arm("semantic").Receipt.Projection.Contains("spam_baseline"));
    }

    [TestMethod]
    public void Fit_CombinedArmTrainsOfflineWithTextThenSemanticFeatures()
    {
        AssertReceipt("combined");
        var view = Transform("combined", models.Validation);
        int textWidth = ((VectorDataViewType)view.Schema["TextFeatures"].Type).Size;
        Assert.IsTrue(textWidth > 10, "The authored text should yield a real vocabulary, not an empty block.");
        Assert.AreEqual(textWidth + 10, ((VectorDataViewType)view.Schema["Features"].Type).Size);
        var text = Vectors(view, "TextFeatures");
        var combined = Vectors(view, "Features");
        for (int i = 0; i < combined.Length; i++)
        {
            CollectionAssert.AreEqual(text[i], combined[i].Take(textWidth).ToArray());
            CollectionAssert.AreEqual(models.Validation[i].Semantic, combined[i].Skip(textWidth).ToArray());
            Assert.AreEqual(textWidth + 10, combined[i].Length);
        }
    }

    [TestMethod]
    [DataRow("text")]
    [DataRow("combined")]
    public void Fit_TextAndCombinedVocabularyExcludesValidationOnlyToken(string arm)
    {
        var view = Transform(arm, models.Validation);
        var column = view.Schema[arm == "text" ? "Features" : "TextFeatures"];
        Assert.IsTrue(column.Annotations.Schema.GetColumnOrNull("SlotNames").HasValue,
            "M04/RD03: learned vocabulary annotations must be observable.");
        VBuffer<ReadOnlyMemory<char>> slots = default;
        column.Annotations.GetValue("SlotNames", ref slots);
        string[] names = slots.DenseValues().Select(s => s.ToString()).ToArray();
        // ML.NET prefixes unigram slots with "Word.". Exact names distinguish words from char ngrams.
        CollectionAssert.Contains(names, "Word.orchardtrainonly", string.Join(" | ", names));
        CollectionAssert.DoesNotContain(names, "Word.zzheldoutuniqueq");
        CollectionAssert.Contains(names, "Word.claim");
        CollectionAssert.Contains(names, "Word.family");
        Assert.AreEqual(((VectorDataViewType)column.Type).Size, slots.Length);
        Assert.IsTrue(models.Training.All(r => !r.Text.Contains("zzheldoutuniqueq", StringComparison.Ordinal)));
        Assert.IsTrue(models.Validation.All(r => !r.Text.Contains("orchardtrainonly", StringComparison.Ordinal)));
        // Source evidence: PredictorTraining.Fit creates view from training only, calls estimator.Fit(view),
        // then Predict(fitted, validation). This test checks that path's fitted word-unigram annotations.
    }

    [TestMethod]
    [DataRow("text")]
    [DataRow("semantic")]
    [DataRow("combined")]
    public void Fit_AllArmsIgnoreSpamBaseline(string arm)
    {
        var before = models.Validation.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        var modified = models.Validation.Select(r => LearningRowFixtures.Copy(r, r.Label ? .999 : .001)).ToArray();
        var model = models.Arm(arm).Model;

        var original = PredictorTraining.Predict(model, models.Validation);
        var changed = PredictorTraining.Predict(model, modified);

        ModelExpectations.Predictions(modified, changed);
        ModelExpectations.Replay(original, changed);
        CollectionAssert.AreEqual(Vectors(Transform(arm, models.Validation), "Features").SelectMany(x => x).ToArray(),
            Vectors(Transform(arm, modified), "Features").SelectMany(x => x).ToArray());
        Assert.IsTrue(modified.Zip(before).All(p => p.First.SpamBaseline != p.Second.SpamBaseline));
        LearningRowFixtures.Unchanged(before, models.Validation);
    }

    [TestMethod]
    [DataRow("text")]
    [DataRow("semantic")]
    [DataRow("combined")]
    public void Predict_AndRealLoadPreserveIdentitiesAndReplayProbabilities(string arm)
    {
        var fitted = models.Arm(arm);
        byte[] originalModel = File.ReadAllBytes(models.ModelPath(arm));
        var receipt = fitted.Receipt with { Projection = fitted.Receipt.Projection.ToArray() };

        var loaded = PredictorTraining.Load(models.ModelPath(arm), receipt, LearningRowFixtures.Identity);
        // Non-source-id order checks that Transform and Predict do not sort or invent identities.
        var input = models.Validation.Reverse().Select(r => LearningRowFixtures.Copy(r)).ToArray();
        var expected = PredictorTraining.Predict(fitted.Model, input);
        var replayed = PredictorTraining.Predict(loaded, input);

        Assert.AreEqual(4, input.Length);
        Assert.AreEqual(input.Length, expected.Length);
        Assert.AreEqual(expected.Length, replayed.Length, "Saved replay must contain exactly every validation row.");
        ModelExpectations.Predictions(input, expected);
        ModelExpectations.Predictions(input, replayed);
        ModelExpectations.Replay(expected, replayed);
        // Independent ML.NET scoring, not another call to production Predict:
        // a constant/default probability mapping would otherwise replay itself.
        var context = new MLContext(1);
        var direct = context.Data.CreateEnumerable<ScoredRow>(
            fitted.Model.Transform(context.Data.LoadFromEnumerable(input)), reuseRowObject: false).ToArray();
        Assert.AreEqual(input.Length, direct.Length);
        Assert.IsTrue(direct.Select(r => r.Probability).Distinct().Count() > 1,
            "The authored fitted model must expose nonconstant scores.");
        for (int i = 0; i < direct.Length; i++)
        {
            Assert.AreEqual(input[i].RowId, direct[i].RowId);
            Assert.AreEqual((double)direct[i].Probability, expected[i].Probability);
            Assert.AreEqual((double)direct[i].Probability, replayed[i].Probability, 1e-6);
        }
        var validation = PredictorTraining.Predict(fitted.Model, models.Validation);
        Assert.AreEqual(ModelExpectations.Threshold(validation, false), receipt.Threshold);
        Assert.AreEqual(ModelExpectations.Threshold(validation, true), receipt.BudgetThreshold);
        ModelExpectations.ValidationMetrics(validation, receipt.Threshold, receipt.Validation);
        string receiptPath = Path.Combine(models.Output(arm), arm + "-12.receipt.json");
        var finalized = System.Text.Json.JsonSerializer.Deserialize<ModelReceipt>(
            File.ReadAllBytes(receiptPath),
            new System.Text.Json.JsonSerializerOptions(System.Text.Json.JsonSerializerDefaults.Web))!;
        Assert.AreEqual(1, finalized.Version);
        Assert.AreEqual(receipt.ModelSha256, finalized.ModelSha256);
        Assert.AreEqual(ArtifactExpectations.Hash(originalModel), finalized.ModelSha256);
        Assert.AreEqual((long)originalModel.Length, finalized.ModelBytes);
        Assert.AreEqual(replayed.Length, finalized.Validation.Rows);
        Assert.AreEqual(receipt.Threshold, finalized.Threshold);
        Assert.AreEqual(receipt.BudgetThreshold, finalized.BudgetThreshold);
        Assert.IsFalse(File.Exists(receiptPath + ".partial"), "Successful verification must leave a finalized receipt.");
        ArtifactExpectations.Bytes(originalModel, models.ModelPath(arm));
        CollectionAssert.AreEqual(QuestionFixtures.Projection, fitted.Receipt.Projection);
    }

    [TestMethod]
    [DataRow("text")]
    [DataRow("semantic")]
    [DataRow("combined")]
    public void Fit_FailedSavedModelVerificationDoesNotFinalizeReceipt(string arm)
    {
        using var temp = new TempDirectory();
        var training = LearningRowFixtures.Training();
        var validation = LearningRowFixtures.Validation();
        var beforeTraining = training.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        var beforeValidation = validation.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        string output = temp.FilePath("failed-verification");

        // In the current API, blank identity deterministically fails the saved
        // model's Load verification after Save. No race, watcher or source seam.
        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            PredictorTraining.Fit(arm, 12, training, validation, "",
                LearningRowFixtures.DatasetHash, LearningRowFixtures.SplitHash,
                LearningRowFixtures.QuestionsHash, output));

        Assert.AreEqual("External producer feature-contract identity mismatch.", error.Message);
        string modelPath = Path.Combine(output, arm + "-12.mlnet");
        string receiptPath = Path.Combine(output, arm + "-12.receipt.json");
        Assert.IsFalse(File.Exists(receiptPath), "A failed verification must never publish a completion receipt.");
        Assert.IsFalse(File.Exists(receiptPath + ".partial"));
        CollectionAssert.AreEqual(new[] { arm + "-12.mlnet" },
            Directory.GetFiles(output).Select(Path.GetFileName).ToArray());
        byte[] savedModel = File.ReadAllBytes(modelPath);
        Assert.IsTrue(savedModel.Length > 0);
        Assert.IsFalse(File.Exists(modelPath + ".partial"));

        // Independently prove Save completed a valid ordinary ML.NET model:
        // the failed stage was contract verification, not Fit or serialization.
        var context = new MLContext(1);
        var physicalModel = context.Model.Load(modelPath, out var schema);
        Assert.AreEqual(10, ((VectorDataViewType)schema[nameof(LearningRow.Semantic)].Type).Size);
        var scored = context.Data.CreateEnumerable<ScoredRow>(
            physicalModel.Transform(context.Data.LoadFromEnumerable(validation)), reuseRowObject: false).ToArray();
        Assert.AreEqual(4, scored.Length);
        CollectionAssert.AreEqual(validation.Select(r => r.RowId).ToArray(), scored.Select(r => r.RowId).ToArray());
        CollectionAssert.AreEqual(validation.Select(r => r.GroupId).ToArray(), scored.Select(r => r.GroupId).ToArray());
        CollectionAssert.AreEqual(validation.Select(r => r.Label).ToArray(), scored.Select(r => r.Label).ToArray());
        Assert.IsTrue(scored.All(r => float.IsFinite(r.Probability) && r.Probability >= 0 && r.Probability <= 1));
        Assert.IsTrue(scored.Select(r => r.Probability).Distinct().Count() > 1);
        ArtifactExpectations.Bytes(savedModel, modelPath);
        LearningRowFixtures.Unchanged(beforeTraining, training);
        LearningRowFixtures.Unchanged(beforeValidation, validation);
    }

    [TestMethod]
    [DataRow("identity-different")]
    [DataRow("identity-case")]
    [DataRow("identity-whitespace")]
    [DataRow("identity-empty")]
    [DataRow("identity-blank")]
    [DataRow("receipt-identity")]
    [DataRow("version")]
    [DataRow("conversion")]
    [DataRow("projection-order")]
    [DataRow("projection-length")]
    [DataRow("projection-baseline")]
    public void Load_ContractAndReceiptMutationsAreRejected(string mutation)
    {
        var original = models.Arm("semantic").Receipt;
        var receipt = original with { Projection = original.Projection.ToArray() };
        string identity = LearningRowFixtures.Identity;
        switch (mutation)
        {
            case "identity-different": identity = "another-contract-v1"; break;
            case "identity-case": identity = identity.ToUpperInvariant(); break;
            case "identity-whitespace": identity = " " + identity + " "; break;
            case "identity-empty": identity = ""; break;
            case "identity-blank": identity = " \t"; break;
            case "receipt-identity": receipt = receipt with { FeatureFingerprint = identity + "-changed" }; break;
            case "version": receipt = receipt with { Version = 2 }; break;
            case "conversion": receipt = receipt with { Conversion = "unvalidated conversion" }; break;
            case "projection-order": (receipt.Projection[0], receipt.Projection[1]) = (receipt.Projection[1], receipt.Projection[0]); break;
            case "projection-length": receipt = receipt with { Projection = receipt.Projection.Take(9).ToArray() }; break;
            case "projection-baseline": receipt.Projection[9] = "spam_baseline"; break;
        }
        byte[] modelBytes = File.ReadAllBytes(models.ModelPath("semantic"));
        byte[] receiptBytes = File.ReadAllBytes(Path.Combine(models.Output("semantic"), "semantic-12.receipt.json"));

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            PredictorTraining.Load(models.ModelPath("semantic"), receipt, identity));

        if (mutation is "version" or "conversion" || mutation.StartsWith("projection", StringComparison.Ordinal))
            Assert.AreEqual("Saved predictor feature projection/receipt mismatch.", error.Message);
        else
            Assert.IsTrue(error.Message.Contains("identity", StringComparison.OrdinalIgnoreCase), error.Message);
        ArtifactExpectations.Bytes(modelBytes, models.ModelPath("semantic"));
        ArtifactExpectations.Bytes(receiptBytes, Path.Combine(models.Output("semantic"), "semantic-12.receipt.json"));
        CollectionAssert.AreEqual(QuestionFixtures.Projection, original.Projection);
        Assert.AreEqual(LearningRowFixtures.Identity, original.FeatureFingerprint);
    }

    [TestMethod]
    [DataRow("changed-length")]
    [DataRow("same-length")]
    public void Load_ChangedLengthAndSameLengthTamperingAreRejected(string mutation)
    {
        using var temp = new TempDirectory();
        var fitted = models.Arm("semantic");
        byte[] original = File.ReadAllBytes(models.ModelPath("semantic"));
        byte[] changed = mutation == "changed-length" ? [.. original, 0x21] : original.ToArray();
        if (mutation == "same-length") changed[changed.Length / 2] ^= 0x01;
        string path = temp.Put("tampered-copy.mlnet", changed);
        var receipt = fitted.Receipt with { Projection = fitted.Receipt.Projection.ToArray() };

        var error = Assert.ThrowsExactly<InvalidDataException>(() =>
            PredictorTraining.Load(path, receipt, LearningRowFixtures.Identity));

        Assert.AreEqual(mutation == "changed-length" ? "Saved predictor feature projection/receipt mismatch." :
            "SHA-256 mismatch: " + path, error.Message);
        Assert.AreEqual(original.Length + (mutation == "changed-length" ? 1 : 0), changed.Length);
        Assert.AreNotEqual(receipt.ModelSha256, ArtifactExpectations.HashFile(path));
        ArtifactExpectations.Bytes(changed, path);
        ArtifactExpectations.Bytes(original, models.ModelPath("semantic"));
        CollectionAssert.AreEqual(QuestionFixtures.Projection, fitted.Receipt.Projection);
    }

    [TestMethod]
    [DataRow("unknown-arm")]
    [DataRow("arm-case")]
    [DataRow("train-ham-only")]
    [DataRow("train-spam-only")]
    [DataRow("validation-ham-only")]
    [DataRow("validation-spam-only")]
    [DataRow("empty-training")]
    [DataRow("empty-validation")]
    [DataRow("intersecting-row")]
    [DataRow("intersecting-group")]
    public void Fit_UnknownArmMissingClassOrIntersectingRowsIsRejected(string mutation)
    {
        using var temp = new TempDirectory();
        var training = LearningRowFixtures.Training();
        var validation = LearningRowFixtures.Validation();
        string arm = mutation == "unknown-arm" ? "baseline" : mutation == "arm-case" ? "Text" : "semantic";
        if (mutation == "train-ham-only") training = training.Where(r => !r.Label).ToArray();
        if (mutation == "train-spam-only") training = training.Where(r => r.Label).ToArray();
        if (mutation == "validation-ham-only") validation = validation.Where(r => !r.Label).ToArray();
        if (mutation == "validation-spam-only") validation = validation.Where(r => r.Label).ToArray();
        if (mutation == "empty-training") training = [];
        if (mutation == "empty-validation") validation = [];
        if (mutation == "intersecting-row") validation[0].RowId = training[0].RowId;
        if (mutation == "intersecting-group") validation[0].GroupId = training[0].GroupId;
        var beforeTrain = training.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        var beforeValidation = validation.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        string output = temp.FilePath("must-not-publish");

        if (arm != "semantic")
        {
            var error = Assert.ThrowsExactly<ArgumentException>(() => PredictorTraining.Fit(arm, 12, training, validation,
                LearningRowFixtures.Identity, LearningRowFixtures.DatasetHash, LearningRowFixtures.SplitHash,
                LearningRowFixtures.QuestionsHash, output));
            Assert.AreEqual("arm", error.ParamName);
        }
        else
        {
            var error = Assert.ThrowsExactly<InvalidDataException>(() => PredictorTraining.Fit(arm, 12, training, validation,
                LearningRowFixtures.Identity, LearningRowFixtures.DatasetHash, LearningRowFixtures.SplitHash,
                LearningRowFixtures.QuestionsHash, output));
            Assert.AreEqual("Fit needs disjoint train/validation rows with both classes.", error.Message);
        }
        Assert.IsFalse(Directory.Exists(output));
        LearningRowFixtures.Unchanged(beforeTrain, training);
        LearningRowFixtures.Unchanged(beforeValidation, validation);
    }

    private static IDataView Transform(string arm, LearningRow[] input)
    {
        var context = new MLContext(1);
        return models.Arm(arm).Model.Transform(context.Data.LoadFromEnumerable(input));
    }

    private static float[][] Vectors(IDataView view, string column)
    {
        var values = new List<float[]>();
        using var cursor = view.GetRowCursor([view.Schema[column]]);
        var getter = cursor.GetGetter<VBuffer<float>>(view.Schema[column]);
        VBuffer<float> buffer = default;
        while (cursor.MoveNext())
        {
            getter(ref buffer);
            values.Add(buffer.DenseValues().ToArray());
        }
        return values.ToArray();
    }

    private static void AssertReceipt(string arm)
    {
        var beforeTraining = models.Training.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        var beforeValidation = models.Validation.Select(r => LearningRowFixtures.Copy(r)).ToArray();
        Assert.AreEqual(12, models.Training.Length);
        Assert.AreEqual(4, models.Validation.Length);
        Assert.AreEqual(6, models.Training.Count(r => r.Label));
        Assert.AreEqual(2, models.Validation.Count(r => r.Label));
        Assert.AreEqual(0, models.Training.Select(r => r.RowId).Intersect(models.Validation.Select(r => r.RowId)).Count());
        Assert.AreEqual(0, models.Training.Select(r => r.GroupId).Intersect(models.Validation.Select(r => r.GroupId)).Count());
        var fitted = models.Arm(arm);
        var receipt = fitted.Receipt;
        string receiptPath = Path.Combine(models.Output(arm), arm + "-12.receipt.json");
        var published = System.Text.Json.JsonSerializer.Deserialize<ModelReceipt>(File.ReadAllBytes(receiptPath),
            new System.Text.Json.JsonSerializerOptions(System.Text.Json.JsonSerializerDefaults.Web))!;

        Assert.AreEqual(1, receipt.Version);
        Assert.AreEqual(arm, receipt.Arm);
        Assert.AreEqual(12, receipt.TargetRows);
        Assert.AreEqual(12, receipt.ActualRows);
        Assert.AreEqual(LearningRowFixtures.Identity, receipt.FeatureFingerprint);
        Assert.AreEqual(LearningRowFixtures.DatasetHash, receipt.DatasetManifestSha256);
        Assert.AreEqual(LearningRowFixtures.SplitHash, receipt.SplitSha256);
        Assert.AreEqual(LearningRowFixtures.QuestionsHash, receipt.QuestionsSha256);
        Assert.AreEqual("Arrow float64 probabilities -> ML.NET float32, checked finite [0,1], original Arrow unchanged",
            receipt.Conversion);
        CollectionAssert.AreEqual(QuestionFixtures.Projection, receipt.Projection);
        CollectionAssert.Contains(new double[] { .0001, .001, .01 }, receipt.L2);
        Assert.AreEqual(1, receipt.Seed);
        Assert.AreEqual(100, receipt.MaxIterations);
        Assert.AreEqual(arm + "-12.mlnet", receipt.ModelFile);
        Assert.AreEqual(ArtifactExpectations.HashFile(models.ModelPath(arm)), receipt.ModelSha256);
        Assert.AreEqual((long)File.ReadAllBytes(models.ModelPath(arm)).Length, receipt.ModelBytes);
        Assert.IsTrue(receipt.ModelBytes > 0);
        CollectionAssert.AreEqual(new[] { arm + "-12.mlnet", arm + "-12.receipt.json" },
            Directory.GetFiles(models.Output(arm)).Select(Path.GetFileName).Order(StringComparer.Ordinal).ToArray());
        Assert.AreEqual(receipt.ModelSha256, published.ModelSha256);
        Assert.AreEqual(receipt.FeatureFingerprint, published.FeatureFingerprint);
        CollectionAssert.AreEqual(QuestionFixtures.Projection, published.Projection);
        var predictions = PredictorTraining.Predict(fitted.Model, models.Validation);
        ModelExpectations.Predictions(models.Validation, predictions);
        Assert.AreEqual(ModelExpectations.Threshold(predictions, false), receipt.Threshold);
        Assert.AreEqual(ModelExpectations.Threshold(predictions, true), receipt.BudgetThreshold);
        ModelExpectations.ValidationMetrics(predictions, receipt.Threshold, receipt.Validation);
        ModelExpectations.ValidationMetrics(predictions, published.Threshold, published.Validation);
        LearningRowFixtures.Unchanged(beforeTraining, models.Training);
        LearningRowFixtures.Unchanged(beforeValidation, models.Validation);
    }
}
